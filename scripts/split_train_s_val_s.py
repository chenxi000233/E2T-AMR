#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split cleaned train_s.pkl into train_s.pkl (80%) and val_s.pkl (20%)
保持原格式(dictbucket)，并按 (class, SNR) 分桶随机划分。
"""

import os, sys, pickle, random, argparse
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-pkl", required=True, help="data/processed/rml2016_10a/train_s.pkl")
    ap.add_argument("--out-train", required=True, help="data/processed/rml2016_10a/train_s.pkl")
    ap.add_argument("--out-val", required=True, help="data/processed/rml2016_10a/val_s.pkl")
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    print(f"[info] Loading {args.in_pkl} ...")
    d = pickle.load(open(args.in_pkl, "rb"))
    assert isinstance(d, dict), "Expected dictbucket format from cleaning step."

    train_buckets = defaultdict(list)
    val_buckets   = defaultdict(list)

    total_all = 0
    total_train = 0
    total_val = 0

    for key, lst in d.items():
        n = len(lst)
        total_all += n
        n_train = int(n * args.ratio)
        idx = list(range(n))
        random.shuffle(idx)
        sel_train = idx[:n_train]
        sel_val = idx[n_train:]
        train_buckets[key] = [lst[i] for i in sel_train]
        val_buckets[key]   = [lst[i] for i in sel_val]
        total_train += len(sel_train)
        total_val += len(sel_val)

    print("=" * 72)
    print(f"[done] Total samples: {total_all}")
    print(f"       Train_s: {total_train} ({total_train/total_all*100:.2f}%)")
    print(f"       Val_s:   {total_val} ({total_val/total_all*100:.2f}%)")
    print("=" * 72)

    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    pickle.dump(dict(train_buckets), open(args.out_train, "wb"))
    pickle.dump(dict(val_buckets), open(args.out_val, "wb"))
    print(f"[saved] -> {args.out_train}")
    print(f"[saved] -> {args.out_val}")

if __name__ == "__main__":
    main()
