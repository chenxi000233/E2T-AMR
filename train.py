# E2T-AMR/train.py
import sys, os
# 确保 src 在导入路径前面
SRC_ROOT = os.path.join(os.path.dirname(__file__), "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from e2tamr.entrypoints.train import main

if __name__ == "__main__":
    main()
