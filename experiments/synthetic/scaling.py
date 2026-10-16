"""Runtime of MAG1C and FastMAG1C as a function of N and s (Fig. 1).

python scaling.py cpu
python scaling.py cuda

Batches of 10 synthetic columns, N from 512 to 16384 and s in
{50, 100, 150, 200, 285}, float64 and float32. Each configuration runs once
to warm up and then RUNS times. Writes one row per run, in milliseconds, to
scaling_cpu.csv or scaling_cuda.csv.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fastmag1c.filters import fastmag1c, mag1c

HERE = Path(__file__).parent
NS = [512, 1024, 2048, 4096, 8192, 16384]
SS = [50, 100, 150, 200, 285]
BATCH = 10
RUNS = 5
K = 30
ALPHA = 1e-4

torch.backends.cuda.matmul.allow_tf32 = False


def batch(N, s):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((s, s))
    L = np.linalg.cholesky(A @ A.T + 1e-3 * np.eye(s))
    x = rng.standard_normal((BATCH, N, s)) @ L.T
    x += np.abs(x.min()) + 1.0
    return x.astype(np.float32)


def timed(fn, x, t, device):
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    fn(x, t, num_iter=K, alpha=ALPHA)
    times = []
    for _ in range(RUNS):
        sync()
        start = time.perf_counter()
        fn(x, t, num_iter=K, alpha=ALPHA)
        sync()
        times.append((time.perf_counter() - start) * 1000)
    return times


def main(device):
    with open(HERE / f"scaling_{device}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device", "dtype", "s", "N", "run", "mag1c_ms", "fastmag1c_ms"])
        for dtype in (torch.float64, torch.float32):
            name = str(dtype).removeprefix("torch.")
            for s in SS:
                t = torch.full((s,), s ** -0.5, device=device, dtype=dtype)
                for N in NS:
                    x = torch.as_tensor(batch(N, s), device=device, dtype=dtype)
                    ref = timed(mag1c, x, t, device)
                    new = timed(fastmag1c, x, t, device)
                    for run, (a, b) in enumerate(zip(ref, new)):
                        w.writerow([device, name, s, N, run, a, b])
                    f.flush()
                    print(device, name, s, N, f"{np.mean(ref):.1f} -> {np.mean(new):.1f} ms", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
