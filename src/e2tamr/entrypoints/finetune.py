#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Finetune E2T-AMR from a pretrained checkpoint (best.pt) on cleaned splits.

Key features:
- load pretrained (strict=False), optionally reset classification head if #classes mismatch
- freeze/unfreeze (linear probe then full FT) with --freeze-body / --epochs-freeze
- param groups with different LRs: --lr-body / --lr-head
- reuse DataModule-like pipeline (RML2016ModalDataset + collate_modal)
- evaluate each epoch (overall/low/high & SNR table via existing `evaluate` in your train.py style)

Usage example:
python -m e2tamr.entrypoints.finetune \
  --config configs/config.yaml \
  --data-cfg configs/data/rml2016_10a.yaml \
  --model-cfg configs/model/e2t_amr.yaml \
  --train-pkl data/processed/rml2016_10a/train_s.pkl \
  --val-pkl   data/processed/rml2016_10a/val_s.pkl \
  --finetune-ckpt outputs/checkpoints/best.pt \
  --epochs 20 \
  --freeze-body \
  --epochs-freeze 3 \
  --lr-body 1e-4 \
  --lr-head 5e-4 \
  --batch-size 512 \
  --device cuda:0
"""

import os, sys, json, math, argparse, inspect, pickle
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# project import path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from e2tamr.models.e2t_amr import E2TAMR
from e2tamr.data.datasets.rml2016 import RML2016ModalDataset, collate_modal
from e2tamr.entrypoints.train import evaluate, format_table  # 直接复用你的 evaluate

# ----------------------------
# Utils
# ----------------------------
def set_global_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_yaml_or_json(path: str) -> Dict[str, Any]:
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

def build_model_from_cfg(model_cfg: dict, num_classes: int, device: torch.device) -> nn.Module:
    sig = inspect.signature(E2TAMR.__init__)
    valid = set(sig.parameters.keys()) - {"self"}
    cand = {}
    for k in ["num_classes","hidden","dropout","seq_len","in_ch","spec_dim","tf_shape",
              "use_iq","use_spec","use_tf","cget_cfg","emix_cfg","utrans_cfg","heads_cfg",
              "T","input_channels"]:
        if k in model_cfg:
            cand[k] = model_cfg[k]
    if "num_classes" not in cand:
        cand["num_classes"] = num_classes
    if "seq_len" not in cand and "T" in cand:
        cand["seq_len"] = cand["T"]
    if "in_ch" not in cand and "input_channels" in cand:
        cand["in_ch"] = cand["input_channels"]
    kwargs = {k:v for k,v in cand.items() if k in valid}
    model = E2TAMR(**kwargs).to(device)
    return model

def load_pretrained(model: nn.Module, ckpt_path: str, device: torch.device) -> Optional[List[str]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    new_state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[finetune] loaded pretrained: missing={len(missing)}, unexpected={len(unexpected)}")
    return ckpt.get("class_names", None)

def reset_classifier_if_needed(model: nn.Module, num_classes_target: int):
    # 你的 head 是 Sequential: [LayerNorm, Dropout, Linear]
    # 最后一层是 nn.Linear(..., num_classes)
    linear = None
    for m in model.head.modules():
        if isinstance(m, nn.Linear):
            linear = m
    if linear is None:
        raise RuntimeError("Cannot find linear classifier in model.head")
    if linear.out_features != num_classes_target:
        in_f = linear.in_features
        new_fc = nn.Linear(in_f, num_classes_target)
        # 可选：将新层权重随机初始化（默认已是kaiming），或从旧权重子集拷贝（类别重叠场景）
        # 这里直接重置
        with torch.no_grad():
            linear.weight.data = new_fc.weight.data
            linear.bias.data = new_fc.bias.data
        print(f"[finetune] reset classifier: {linear.out_features} -> {num_classes_target}")

def set_requires_grad(mod: nn.Module, flag: bool):
    for p in mod.parameters():
        p.requires_grad = flag

def make_param_groups(model: nn.Module, lr_body: float, lr_head: float, weight_decay: float):
    # 将 head 与 其它部分分开
    head_params = list(model.head.parameters())
    body_params = [p for n,p in model.named_parameters() if ("head." not in n)]
    return [
        {"params": body_params, "lr": lr_body, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
    ]

# ----------------------------
# Data helpers
# ----------------------------
def make_loader(pkl_path: str, class_names: List[str], data_cfg: dict,
                batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    ds = RML2016ModalDataset(
        split_pkl=pkl_path,
        class_names=class_names,
        use_iq=bool(data_cfg.get("use_iq", True)),
        use_spec=bool(data_cfg.get("use_spec", True)),
        use_tf=bool(data_cfg.get("use_tf", True)),
        spec_nfft=int(data_cfg.get("modal", {}).get("spec_nfft", 256)),
        tf_nfft=int(data_cfg.get("modal", {}).get("tf_nfft", 64)),
        tf_hop=int(data_cfg.get("modal", {}).get("tf_hop", 32)),
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_modal
    )

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    ap.add_argument("--data-cfg", required=True)
    ap.add_argument("--model-cfg", required=True)
    ap.add_argument("--train-pkl", required=True)
    ap.add_argument("--val-pkl", required=True)
    ap.add_argument("--finetune-ckpt", required=True, help="pretrained checkpoint (best.pt)")

    # train hparams
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--epochs-freeze", type=int, default=0, help="epochs to freeze body (linear probe)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr-body", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--freeze-body", action="store_true", help="freeze non-head params (can be used with epochs-freeze)")

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    data_cfg  = load_yaml_or_json(args.data_cfg)["data"]
    model_cfg = load_yaml_or_json(args.model_cfg)

    class_names = data_cfg["classes"]
    num_classes = len(class_names)
    snr_list = sorted([int(s) for s in range(-20, 20, 2)]) + [18]  # 与 RML2016.10a 兼容显示

    # loaders
    train_loader = make_loader(args.train_pkl, class_names, data_cfg, args.batch_size, num_workers=4, shuffle=True)
    val_loader   = make_loader(args.val_pkl,   class_names, data_cfg, args.batch_size, num_workers=4, shuffle=False)

    # model
    model = build_model_from_cfg(model_cfg, num_classes=num_classes, device=device)

    # load pretrained
    ckpt_classes = load_pretrained(model, args.finetune-ckpt if hasattr(args, "finetune-ckpt") else args.finetune_ckpt, device)  # noqa
    # ↑ 兼容命名，下面用规范属性名
    finetune_ckpt = getattr(args, "finetune_ckpt", getattr(args, "finetune-ckpt".replace("-", "_"), None))

    # 若 ckpt 带有 class_names 但与当前不一致，重置分类头
    if ckpt_classes is not None and len(ckpt_classes) != num_classes:
        reset_classifier_if_needed(model, num_classes)

    # freeze / param groups
    if args.freeze_body or args.epochs_freeze > 0:
        # 冻结 backbone（非 head）
        for n,p in model.named_parameters():
            if not n.startswith("head."):
                p.requires_grad = False
        print("[finetune] body frozen")

    # optimizer （参数组不同学习率）
    param_groups = make_param_groups(model, args.lr_body, args.lr_head, args.weight_decay)
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # loss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_acc = -1.0
    ckpt_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    out_best = os.path.join(ckpt_dir, "best_finetune.pt")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        # 若线性探测结束，解冻 body
        if args.epochs_freeze > 0 and epoch == (args.epochs_freeze + 1):
            for n,p in model.named_parameters():
                p.requires_grad = True
            # 重新构建优化器（以确保冻结状态改变后生效）
            param_groups = make_param_groups(model, args.lr_body, args.lr_head, args.weight_decay)
            optimizer = optim.AdamW(param_groups)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs - args.epochs_freeze))
            print("[finetune] body unfrozen (full fine-tuning)")

        for bidx, batch in enumerate(train_loader, start=1):
            xdict = {}
            if "iq" in batch:   xdict["iq"]   = batch["iq"].to(device)
            if "spec" in batch: xdict["spec"] = batch["spec"].to(device)
            if "tf" in batch:   xdict["tf"]   = batch["tf"].to(device)
            y = batch["y"].to(device)

            logits = model(xdict)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                running_loss += loss.item() * y.size(0)
                running_correct += (pred == y).sum().item()
                running_total += y.numel()

            global_step += 1
            if bidx % 50 == 0:
                print(f"[Epoch {epoch:03d} | Batch {bidx:05d}] loss={loss.item():.4f}")

        scheduler.step()

        train_loss = running_loss / max(1, running_total)
        train_acc  = running_correct / max(1, running_total)

        # ---- 验证 ----
        eval_res = evaluate(model, val_loader, device, class_names, snr_list)
        overall = eval_res["overall"]
        low_acc = eval_res["low"]
        high_acc = eval_res["high"]
        table = eval_res["table"]

        print("=" * 90)
        print(f"[Epoch {epoch:03d}]")
        print(f"Train Acc = {train_acc:.4f} | Val Acc (Overall) = {overall:.4f} | Low SNR (≤0dB) = {low_acc:.4f} | High SNR (≥10dB) = {high_acc:.4f}")
        print("-" * 90)
        print(table)
        print("=" * 90)

        if overall > best_acc:
            best_acc = overall
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "overall_acc": overall,
                "low_acc": low_acc,
                "high_acc": high_acc,
                "class_names": class_names,
                "note": "finetuned from pretrained"
            }, out_best)
            print(f"  ↳ Saved BEST checkpoint: {out_best} (val_acc={overall:.4f})")

    print(f"[done] Finetune finished. best_val_acc={best_acc:.4f}, best_path={out_best}")


if __name__ == "__main__":
    # 支持 python -m e2tamr.entrypoints.finetune
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)
    main()
