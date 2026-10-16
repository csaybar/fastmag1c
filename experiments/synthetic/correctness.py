"""Numerical agreement of MAG1C and FastMAG1C on synthetic CCD columns (Table II).

python correctness.py

One column per seed, s = 50 bands, N in {512, 1024, 2048}, 100 seeds, on
CPU and, when available, CUDA, in float64 and float32. Writes one row per
seed to correctness.csv with the P99, mean and max of |FastMAG1C - MAG1C|
over the N pixels of the column.
"""

import csv
from pathlib import Path

import numpy as np
import torch

from fastmag1c.filters import fastmag1c, mag1c

HERE = Path(__file__).parent
S = 50
NS = [512, 1024, 2048]
SEEDS = 100
K = 30
ALPHA = 1e-4

# float32 products on Ampere GPUs may otherwise use TF32
torch.backends.cuda.matmul.allow_tf32 = False


def column(N, s, seed):
    # Sigma = A A^T + 1e-3 I, N pixels drawn through its Cholesky factor and
    # shifted to positive values, as in the paper
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((s, s))
    L = np.linalg.cholesky(A @ A.T + 1e-3 * np.eye(s))
    x = rng.standard_normal((N, s)) @ L.T
    x += np.abs(x.min()) + 1.0
    return x.astype(np.float32)


def main():
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows = []
    for device in devices:
        for dtype in (torch.float64, torch.float32):
            t = torch.full((S,), S ** -0.5, device=device, dtype=dtype)
            for N in NS:
                for seed in range(SEEDS):
                    x = torch.as_tensor(column(N, S, seed)[None], device=device, dtype=dtype)
                    ref, _ = mag1c(x, t, num_iter=K, alpha=ALPHA)
                    new, _ = fastmag1c(x, t, num_iter=K, alpha=ALPHA)
                    d = (new - ref).abs().flatten().double().cpu().numpy()
                    rows.append([device, str(dtype).removeprefix("torch."), N, seed,
                                 np.percentile(d, 99), d.mean(), d.max()])
                print(device, dtype, N, "done", flush=True)

    with open(HERE / "correctness.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device", "dtype", "N", "seed", "p99", "mean", "max"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
