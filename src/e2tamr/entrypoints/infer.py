"""Inference entrypoint (skeleton).
Usage:
    python -m e2tamr.entrypoints.infer --ckpt outputs/checkpoints/last.ckpt
"""
import argparse

def get_args():
    p = argparse.ArgumentParser(description="E2T-AMR Inference")
    p.add_argument("--ckpt", type=str, required=False, help="Path to checkpoint")
    p.add_argument("--input", type=str, required=False, help="Path to input file/dir")
    return p.parse_args()

def main():
    args = get_args()
    # TODO: load model checkpoint, run forward on input(s), save outputs
    print("[infer] ckpt:", args.ckpt, "input:", args.input)

if __name__ == "__main__":
    main()
