# src/e2tamr/entrypoints/train.py
# -*- coding: utf-8 -*-

import os
import sys
import math
import time
import json
import random
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

# 项目内导入
from e2tamr.data.datamodule import DataModule
from e2tamr.models.e2t_amr import E2TAMR

# -----------------------------
# 基础工具
# -----------------------------
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load YAML: {path}. {e}")


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def resolve_subcfgs(main_cfg_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    cfg = safe_load_yaml(main_cfg_path)
    # 允许在主 config 里写 data/model/train/eval 的名字
    data_name  = cfg.get("data", {}).get("name", "rml2016_10a")
    model_name = cfg.get("model", {}).get("name", "e2t_amr")

    data_cfg  = safe_load_yaml(os.path.join(PROJECT_ROOT, "configs", "data",  f"{data_name}.yaml"))
    model_cfg = safe_load_yaml(os.path.join(PROJECT_ROOT, "configs", "model", f"{model_name}.yaml"))
    train_cfg = safe_load_yaml(os.path.join(PROJECT_ROOT, "configs", "train", "base.yaml"))
    eval_cfg  = safe_load_yaml(os.path.join(PROJECT_ROOT, "configs", "eval", "metrics.yaml"))

    # 允许主 config 覆盖子配置
    def deep_update(base: Dict[str, Any], over: Dict[str, Any]):
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                deep_update(base[k], v)
            else:
                base[k] = v
        return base

    deep_update(data_cfg, cfg.get("data", {}))
    deep_update(model_cfg, cfg.get("model", {}))
    deep_update(train_cfg, cfg.get("train", {}))
    deep_update(eval_cfg,  cfg.get("eval", {}))

    return data_cfg, model_cfg, train_cfg, eval_cfg


# -----------------------------
# 评价/打印
# -----------------------------
def format_table(headers: List[str], rows: List[List[str]]) -> str:
    # 简单等宽表格字符串
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    sep = " | ".join(f"{h:>{widths[i]}}" for i, h in enumerate(headers))
    bar = "-" * len(sep)
    lines = [sep, bar]
    for r in rows:
        lines.append(" | ".join(f"{str(c):>{widths[i]}}" for i, c in enumerate(r)))
    return "\n".join(lines)


# -----------------------------
# 构建损失（含类别权重估计）
# -----------------------------
def _estimate_class_weight(loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for batch in loader:
        # 兼容字典/元组
        if isinstance(batch, dict):
            y = batch["y"]
        else:
            # 旧格式 fallback: (x, y, snr) 或 (x, y)
            if len(batch) >= 2:
                y = batch[1]
            else:
                raise ValueError("Cannot find labels in batch for class weight estimation.")
        counts += torch.bincount(y.view(-1), minlength=num_classes)
    freq = counts.float().clamp_min(1.0)
    w = 1.0 / freq
    w = w / w.mean()
    return w.to(device)


class FocalCrossEntropyLoss(nn.Module):
    """Focal CE，支持 class_weight 与 label_smoothing"""
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ls = label_smoothing

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(
            logits, target, weight=self.weight, reduction="none", label_smoothing=self.ls
        )
        with torch.no_grad():
            pt = torch.softmax(logits, dim=-1).gather(1, target.view(-1, 1)).squeeze(1).clamp_min(1e-6)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def build_criterion(loss_cfg: dict,
                    num_classes: int,
                    train_loader_for_weight: Optional[DataLoader],
                    device: torch.device) -> nn.Module:
    name = (loss_cfg.get("name") or "cross_entropy").lower()
    class_balance = bool(loss_cfg.get("class_balance", False))
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))

    weight = None
    if class_balance and train_loader_for_weight is not None:
        weight = _estimate_class_weight(train_loader_for_weight, num_classes, device)

    if name in ["ce", "cross_entropy", "cross-entropy"]:
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    if name in ["focal", "focal_ce", "focal-cross-entropy"]:
        gamma = float(loss_cfg.get("gamma", 2.0))
        return FocalCrossEntropyLoss(gamma=gamma, weight=weight, label_smoothing=label_smoothing)

    # fallback
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)


# -----------------------------
# 评估：按 SNR 与类别统计 ACC
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             class_names: List[str],
             snr_list: List[int]) -> Dict[str, Any]:
    model.eval()
    num_classes = len(class_names)

    # per snr × class：统计正确个数 / 总数
    correct = {snr: np.zeros(num_classes, dtype=np.int64) for snr in snr_list}
    total   = {snr: np.zeros(num_classes, dtype=np.int64) for snr in snr_list}

    total_correct = 0
    total_samples = 0

    for batch in loader:
        # 兼容字典 batch
        if isinstance(batch, dict):
            x_iq = batch.get("iq", None)
            x_spec = batch.get("spec", None)
            x_tf = batch.get("tf", None)
            y = batch["y"].to(device)
            snr = batch["snr"].cpu().numpy().tolist()
        else:
            # 旧格式 not used in current pipeline
            raise RuntimeError("Expected dict batch with keys [iq/spec/tf/y/snr].")

        # 模型前向：当前 E2TAMR forward(xdict)
        xdict = {}
        if x_iq is not None:   xdict["iq"] = x_iq.to(device)
        if x_spec is not None: xdict["spec"] = x_spec.to(device)
        if x_tf is not None:   xdict["tf"] = x_tf.to(device)

        logits = model(xdict)  # (B, C)
        pred = torch.argmax(logits, dim=-1)

        total_correct += (pred == y).sum().item()
        total_samples += y.numel()

        y_np = y.cpu().numpy()
        pred_np = pred.cpu().numpy()

        for i in range(len(snr)):
            s = int(snr[i])
            gt = int(y_np[i])
            pd = int(pred_np[i])
            total[s][gt] += 1
            if gt == pd:
                correct[s][gt] += 1

    # 计算整体 & 分段
    overall_acc = total_correct / max(1, total_samples)

    # Low / High
    low_mask  = [s for s in snr_list if s <= 0]
    high_mask = [s for s in snr_list if s >= 10]

    def _acc_mask(snrs):
        c = 0
        t = 0
        for s in snrs:
            c += int(sum(correct[s]))
            t += int(sum(total[s]))
        return (c / t) if t > 0 else 0.0

    low_acc  = _acc_mask(low_mask)
    high_acc = _acc_mask(high_mask)

    # 生成可读表格（每行一个 SNR，每列一个类，值为每类的准确率%，以及行均值）
    table_rows = []
    for s in snr_list:
        row = [f"{s:>4d}"]
        per_cls = []
        for c in range(num_classes):
            t = total[s][c]
            acc = 100.0 * (correct[s][c] / t) if t > 0 else 0.0
            per_cls.append(acc)
            row.append(f"{acc:6.1f}")
        mean_acc = np.mean(per_cls) if len(per_cls) > 0 else 0.0
        row.append(f"{mean_acc:6.1f}")
        table_rows.append(row)

    headers = ["SNR(dB)"] + [f"{n:>8s}" for n in class_names] + ["Mean"]
    table_str = format_table(headers, table_rows)

    return {
        "overall": overall_acc,
        "low": low_acc,
        "high": high_acc,
        "table": table_str
    }


# -----------------------------
# 构建模型
# -----------------------------
def build_model(model_cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    # 直接使用 E2TAMR（内部已处理多模态分支）
    mc = model_cfg.get("model", model_cfg)
    hidden = int(mc.get("hidden", 64))
    spec_dim = int(mc.get("spec_dim", 129))
    tf_shape = mc.get("tf_shape", [33, 5])  # (Tf, Ff)
    use_iq = bool(mc.get("use_iq", True))
    use_spec = bool(mc.get("use_spec", True))
    use_tf = bool(mc.get("use_tf", True))

    model = E2TAMR(
        num_classes=num_classes,
        hidden=hidden,
        spec_dim=spec_dim,
        tf_shape=tuple(tf_shape),
        use_iq=use_iq,
        use_spec=use_spec,
        use_tf=use_tf,
    )
    return model


# -----------------------------
# 主流程
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--print_freq", type=int, default=50, help="print batch loss every N steps")
    args = parser.parse_args()

    data_cfg, model_cfg, train_cfg, eval_cfg = resolve_subcfgs(args.config)

    # 设备与随机种
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(train_cfg.get("seed", 42))
    set_global_seed(seed)

    # DataModule
    dm = DataModule(data_cfg, train_cfg)
    dm.setup()
    class_names = dm.class_names
    num_classes = len(class_names)
    snr_list = sorted([int(s) for s in range(-20, 20, 2)]) + [18]  # 与 RML2016.10a 兼容
    # 打印类别
    print(f"[info] classes({num_classes}): {class_names}")

    train_loader = dm.train_loader()
    val_loader   = dm.val_loader()

    # 模型
    model = build_model(model_cfg, num_classes=num_classes).to(device)

    # 优化器 / 调度器
    tcfg = train_cfg.get("train", train_cfg)
    lr = float(tcfg.get("lr", 1e-3))
    wd = float(tcfg.get("weight_decay", 0.0))
    epochs = int(args.epochs or tcfg.get("epochs", 100))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 损失
    loss_cfg = tcfg.get("loss", {"name": "cross_entropy", "label_smoothing": 0.05, "class_balance": True})
    # 用一个小的可迭代器来估计类别权重（不会消耗主训练的 loader）
    train_loader_for_weight = dm.train_loader()
    criterion = build_criterion(loss_cfg, num_classes, train_loader_for_weight, device)

    # 训练目录
    ckpt_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best.pt")
    best_acc = -1.0

    # 训练循环
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for bidx, batch in enumerate(train_loader, start=1):
            # 兼容字典 batch
            if isinstance(batch, dict):
                xdict = {}
                if "iq" in batch:   xdict["iq"]   = batch["iq"].to(device)
                if "spec" in batch: xdict["spec"] = batch["spec"].to(device)
                if "tf" in batch:   xdict["tf"]   = batch["tf"].to(device)
                y = batch["y"].to(device)
            else:
                raise RuntimeError("Expected dict batch with keys [iq/spec/tf/y/snr].")

            logits = model(xdict)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * y.size(0)
            pred = torch.argmax(logits, dim=-1)
            running_correct += (pred == y).sum().item()
            running_total += y.numel()

            global_step += 1
            if bidx % max(1, int(args.print_freq)) == 0:
                print(f"[Epoch {epoch:03d} | Batch {bidx:05d}] loss={loss.item():.4f}")

        scheduler.step()

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)

        # ---- 验证 ----
        eval_res = evaluate(model, val_loader, device, class_names, snr_list)
        overall = eval_res["overall"]
        low_acc = eval_res["low"]
        high_acc = eval_res["high"]
        table = eval_res["table"]

        print("=" * 90)
        print(f"[Epoch {epoch:03d}]")
        print(f"Train Acc = {train_acc:.4f} | Val Acc (Overall) = {overall:.4f} | Low SNR (≤0dB) = {low_acc:.4f} | High SNR (>0dB) = {high_acc:.4f}")
        print("-" * 90)
        print(table)
        print("=" * 90)

        # 保存 best
        if overall > best_acc:
            best_acc = overall
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "overall_acc": overall,
                        "low_acc": low_acc,
                        "high_acc": high_acc,
                        "class_names": class_names},
                       best_path)
            print(f"  ↳ Saved BEST checkpoint: {best_path}")

    print(f"[done] Training finished. best_val_acc={best_acc:.4f}, best_path={best_path}")


if __name__ == "__main__":
    # 允许从项目根目录调用：python train.py --config configs/config.yaml
    # 自动把 src 加到 PYTHONPATH，防止环境问题
    src_path = os.path.join(PROJECT_ROOT, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    main()
