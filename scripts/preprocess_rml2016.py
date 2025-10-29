#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 RML2016.10a_dict.pkl 严格按 (class, SNR) 8:2 切分，保存为 train.pkl / val.pkl
并打印完整的数据概览与校验信息。
"""
import os
import pickle
from collections import defaultdict, Counter
from typing import Dict, Tuple, List

import numpy as np

SRC_PKL = "data/processed/rml2016_10a/RML2016.10a_dict.pkl"
OUT_DIR = "data/processed/rml2016_10a"  # 输出目录
TRAIN_OUT = os.path.join(OUT_DIR, "train.pkl")
VAL_OUT   = os.path.join(OUT_DIR, "val.pkl")

SEED = 42
TRAIN_RATIO = 0.8
SHUFFLE = True

# 如果你想锁定类别顺序（可与 configs/data/rml2016_10a.yaml 保持一致）：
CLASS_ORDER = [
    "8PSK","AM-DSB","AM-SSB","BPSK","CPFSK","GFSK","PAM4","QAM16","QAM64","QPSK","WBFM"
]

def main():
    assert os.path.isfile(SRC_PKL), f"Not found: {SRC_PKL}"
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(SRC_PKL, "rb") as f:
        raw: Dict[Tuple[str, int], np.ndarray] = pickle.load(f, encoding="latin1")

    # 稳定排序：按(调制, SNR) 排序
    keys_sorted = sorted(raw.keys(), key=lambda k: (str(k[0]), int(k[1])))

    rng = np.random.default_rng(SEED)
    train_dict: Dict[Tuple[str, int], np.ndarray] = {}
    val_dict  : Dict[Tuple[str, int], np.ndarray] = {}

    # 统计
    by_cls = Counter()
    by_snr = Counter()
    by_pair_train = Counter()
    by_pair_val = Counter()

    print("==> Splitting per (Class, SNR) with 8:2 ...")
    for (mod, snr) in keys_sorted:
        arr = raw[(mod, snr)]             # (N, 2, 128)
        N = arr.shape[0]
        idx = np.arange(N)
        if SHUFFLE:
            rng.shuffle(idx)
        n_tr = int(round(N * TRAIN_RATIO))
        tr_ids = idx[:n_tr]
        va_ids = idx[n_tr:]

        train_dict[(mod, snr)] = arr[tr_ids]
        val_dict[(mod, snr)]   = arr[va_ids]

        by_cls[mod] += N
        by_snr[int(snr)] += N
        by_pair_train[(mod, int(snr))] = len(tr_ids)
        by_pair_val[(mod, int(snr))]   = len(va_ids)

    with open(TRAIN_OUT, "wb") as f:
        pickle.dump(train_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(VAL_OUT, "wb") as f:
        pickle.dump(val_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 概览打印
    n_tr_total = sum(by_pair_train.values())
    n_va_total = sum(by_pair_val.values())
    print(f"\n==> Saved:")
    print(f"  Train: {TRAIN_OUT}  (samples: {n_tr_total})")
    print(f"  Val  : {VAL_OUT}    (samples: {n_va_total})")

    # 按类打印
    print("\n[Per Class total counts]")
    for c in (CLASS_ORDER if CLASS_ORDER else sorted(by_cls.keys())):
        print(f"  {c:>7s}: {by_cls[c]}")

    # 按 SNR 打印
    print("\n[Per SNR total counts]")
    for s in sorted(by_snr.keys()):
        print(f"  {s:>3d} dB: {by_snr[s]}")

    # 校验若干样本
    any_key = keys_sorted[0]
    print("\n[Sanity Sample]")
    print(f"  Example key: {any_key}")
    print(f"  Train block shape: {train_dict[any_key].shape}  Val block shape: {val_dict[any_key].shape}")
    print("  Example IQ snippet (first sample, first 5 points):")
    ex = train_dict[any_key][0]  # (2,128)
    print("   I[:5] =", np.round(ex[0,:5], 4))
    print("   Q[:5] =", np.round(ex[1,:5], 4))

    # 逐对打印（可选）
    print("\n[Per (Class, SNR) counts  (train | val)]")
    print("  mod      snr   train   val")
    for (mod, snr) in keys_sorted:
        print(f"  {mod:7s}  {snr:>3d}   {by_pair_train[(mod,snr)]:>5d} | {by_pair_val[(mod,snr)]:>5d}")

    print("\nDone.")

if __name__ == "__main__":
    main()
