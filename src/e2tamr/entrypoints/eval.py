"""Evaluation entrypoint (skeleton).
Usage:
    python -m e2tamr.entrypoints.eval --ckpt outputs/checkpoints/last.ckpt
"""
import argparse

def get_args():
    p = argparse.ArgumentParser(description="E2T-AMR Evaluation")
    p.add_argument("--ckpt", type=str, required=False, help="Path to checkpoint")
    p.add_argument("--split", type=str, default="test", choices=["val","test"])
    return p.parse_args()

def main():
    args = get_args()
    # TODO: load model checkpoint, run metrics on split, write reports/figures
    print("[eval] ckpt:", args.ckpt, "split:", args.split)

if __name__ == "__main__":
    main()
