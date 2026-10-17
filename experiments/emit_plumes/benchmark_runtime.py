"""Time both retrievals on EMIT granules using the paper's GPU settings.

Usage: python benchmark_plume_runtime.py OUTPUT.csv GRANULE.nc [GRANULE.nc ...]
Select one idle physical GPU with CUDA_VISIBLE_DEVICES before running.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

import fastmag1c


WINDOWS = {50: (2122, 2488), 128: (1550, 2500)}
RUNS = 5
BATCH_SIZE = 64
torch.backends.cuda.matmul.allow_tf32 = False


def measure(rad, template, method):
    kwargs = dict(
        template=template,
        method=method,
        device="cuda",
        dtype=torch.float64,
        batch_size=BATCH_SIZE,
    )
    fastmag1c.retrieve(rad[:, :8], **kwargs)
    durations = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fastmag1c.retrieve(rad, **kwargs)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    return durations


def main(output, paths):
    gpu = torch.cuda.get_device_name(0)
    with open(output, "w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ["granule", "downtrack", "crosstrack", "bands", "run", "dtype",
             "batch_size", "mag1c_s", "fastmag1c_s", "gpu"]
        )
        for path in paths:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            for bands, window in WINDOWS.items():
                rad, wavelengths = fastmag1c.read_emit(path, window)
                if rad.shape[2] != bands:
                    raise ValueError(f"{path}: expected {bands} bands, got {rad.shape[2]}")
                template = fastmag1c.load_template(wavelengths)
                reference = measure(rad, template, "mag1c")
                fast = measure(rad, template, "fastmag1c")
                for run, (ref_time, fast_time) in enumerate(zip(reference, fast)):
                    writer.writerow(
                        [path.stem, rad.shape[0], rad.shape[1], bands, run,
                         "float64", BATCH_SIZE, ref_time, fast_time, gpu]
                    )
                stream.flush()
                print(
                    f"{path.stem} s={bands}: "
                    f"{np.mean(reference):.3f}s -> {np.mean(fast):.3f}s "
                    f"({np.mean(reference) / np.mean(fast):.2f}x)",
                    flush=True,
                )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
