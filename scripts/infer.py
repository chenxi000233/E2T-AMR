#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 train.pkl 读取一条样本 -> 打印数据形状/真实标签
用训练同样的数据处理链 (RML2016ModalDataset + collate_modal)
加载 best.pt 做单样本推理 -> 输出预测类别/标签/置信度
"""

import os
import sys
import json
import argparse
import pickle
import inspect

import numpy as np
import torch
import torch.nn.functional as F

# 保证能 import 到 src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# 依赖：和训练一致
from e2tamr.data.datasets.rml2016 import RML2016ModalDataset, collate_modal
from e2tamr.models.e2t_amr import E2TAMR


def load_yaml_or_json(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".yaml", ".yml"]:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config extension: {ext}")


def build_model_from_cfg(model_cfg: dict, num_classes: int, device: torch.device):
    """根据 E2TAMR 的 __init__ 签名安全构建，避免多传参数导致报错"""
    sig = inspect.signature(E2TAMR.__init__)
    valid = set(sig.parameters.keys()) - {"self"}

    # 可能用到的键（和你的训练脚本一致）
    cand = {}
    for k in [
        "num_classes", "hidden", "dropout", "seq_len", "in_ch",
        "spec_dim", "tf_shape", "use_iq", "use_spec", "use_tf",
        "cget_cfg", "emix_cfg", "utrans_cfg", "heads_cfg",
        "T", "input_channels"
    ]:
        if k in model_cfg:
            cand[k] = model_cfg[k]

    if "num_classes" not in cand:
        cand["num_classes"] = num_classes
    if "seq_len" not in cand and "T" in cand:
        cand["seq_len"] = cand["T"]
    if "in_ch" not in cand and "input_channels" in cand:
        cand["in_ch"] = cand["input_channels"]

    kwargs = {k: v for k, v in cand.items() if k in valid}
    model = E2TAMR(**kwargs).to(device)
    return model


def load_ckpt(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    new_state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[info] loaded ckpt: missing={len(missing)}, unexpected={len(unexpected)}")
    return ckpt.get("class_names", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-cfg", required=True, help="configs/data/rml2016_10a.yaml")
    ap.add_argument("--model-cfg", required=True, help="configs/model/e2t_amr.yaml")
    ap.add_argument("--ckpt", required=True, help="outputs/checkpoints/best.pt")
    ap.add_argument("--train-pkl", required=True, help="data/processed/rml2016_10a/train.pkl")
    ap.add_argument("--index", type=int, default=0, help="取第几条样本")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    # 读取配置
    data_cfg = load_yaml_or_json(args.data_cfg)["data"]
    model_cfg = load_yaml_or_json(args.model_cfg)

    class_names = data_cfg["classes"]
    num_classes = len(class_names)

    # ===== 1) 读取 train.pkl 的一条（原始调试信息用） =====
    raw = pickle.load(open(args.train_pkl, "rb"))
    # 打印真实标签（不依赖 dataset），兼容三种结构
    def peek_one(raw_obj, idx):
        if isinstance(raw_obj, list):
            it = raw_obj[idx]
            if isinstance(it, dict):
                y = it.get("y", it.get("label"))
                snr = it.get("snr", it.get("SNR", it.get("snr_db")))
                x = it.get("iq", it.get("x", {"I": it.get("I"), "Q": it.get("Q")}))
                return x, y, snr
            elif isinstance(it, (list, tuple)) and len(it) >= 3:
                return it[0], it[1], it[2]
        elif isinstance(raw_obj, dict) and all(k in raw_obj for k in ["x","y","snr"]):
            return raw_obj["x"][idx], raw_obj["y"][idx], raw_obj["snr"][idx]
        elif isinstance(raw_obj, dict):
            # dict bucket: {(label,snr): [iq,...]}
            flat = []
            for k, lst in raw_obj.items():
                for x in lst:
                    flat.append((k, x))
            key, x = flat[idx]
            if isinstance(key, tuple) and len(key) == 2:
                y, snr = key
            elif isinstance(key, str) and "|" in key:
                y, snr = key.split("|")
            else:
                y, snr = key, None
            return x, y, snr
        raise ValueError("Unsupported train.pkl structure.")
    x0, y0, snr0 = peek_one(raw, args.index)

    # 标准化真实标签显示
    if isinstance(y0, (int, np.integer)):
        y_id = int(y0)
        y_name = class_names[y_id] if 0 <= y_id < num_classes else f"id={y_id}"
    else:
        y_name = str(y0)
        y_id = class_names.index(y_name) if y_name in class_names else y_name

    print("=" * 72)
    print(f"[RAW] from {args.train_pkl} index={args.index}")
    print(f"  Real Label: id={y_id}  name={y_name}  SNR={snr0}")
    # 不刷屏：只提示原始对象类型
    print(f"  Raw item types: x={type(x0)}, y={type(y0)}")
    print("=" * 72)

    # ===== 2) 用与训练相同的数据处理链取同一条 =====
    ds = RML2016ModalDataset(
        split_pkl=args.train_pkl,
        class_names=class_names,
        use_iq=bool(data_cfg.get("use_iq", True)),
        use_spec=bool(data_cfg.get("use_spec", True)),
        use_tf=bool(data_cfg.get("use_tf", True)),
        spec_nfft=int(data_cfg.get("modal", {}).get("spec_nfft", 256)),
        tf_nfft=int(data_cfg.get("modal", {}).get("tf_nfft", 64)),
        tf_hop=int(data_cfg.get("modal", {}).get("tf_hop", 32)),
    )
    sample = ds[args.index]   # 单条样本（dict）
    # 走一次 collate，形成 batch=1，确保与训练完全一致的张量形状/类型
    batch = collate_modal([sample])

    # 打印数据形状
    print("[BATCH] shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:>4s}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k:>4s}: type={type(v)}")
    # y 是张量
    y_true = int(batch["y"].item())
    y_true_name = class_names[y_true]
    print(f"  -> True Label (after pipeline): id={y_true}, name={y_true_name}")
    print("=" * 72)

    # ===== 3) 构建模型 & 加载权重 =====
    # 注意：模型模态开关/维度应和训练一致（建议从 model_cfg 读取）
    dummy_num_classes = model_cfg.get("num_classes", num_classes)
    model = build_model_from_cfg(model_cfg, num_classes=dummy_num_classes, device=device)
    ckpt_classes = load_ckpt(model, args.ckpt, device)
    # 若 ckpt 内含 class_names，以 ckpt 为准
    if ckpt_classes is not None and len(ckpt_classes) != dummy_num_classes:
        model = build_model_from_cfg(model_cfg, num_classes=len(ckpt_classes), device=device)
        _ = load_ckpt(model, args.ckpt, device)
        class_names = ckpt_classes
        num_classes = len(class_names)
    model.eval()

    # ===== 4) 前向推理 =====
    xdict = {}
    if "iq" in batch:   xdict["iq"]   = batch["iq"].to(device)
    if "spec" in batch: xdict["spec"] = batch["spec"].to(device)
    if "tf" in batch:   xdict["tf"]   = batch["tf"].to(device)

    with torch.no_grad():
        logits = model(xdict)                 # (1, C)
        probs  = F.softmax(logits, dim=-1)[0] # (C,)
        conf, pred = torch.max(probs, dim=-1)
        pred_id = int(pred.item())
        pred_name = class_names[pred_id] if 0 <= pred_id < num_classes else f"id={pred_id}"
        conf_v = float(conf.item())

    # ===== 5) 输出结果 =====
    print("[INFER] predicted:")
    print(f"  class_id={pred_id}, class_name={pred_name}, confidence={conf_v:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
