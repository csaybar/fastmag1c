"""Runtime of MAG1C and FastMAG1C on the granules in granules.json.

python timing.py DATA_DIR

Each configuration runs once on a few columns to warm up and then REPEATS
times on the full granule. Writes times.csv with the mean and standard
deviation in seconds. Run it on a GPU that nothing else is using.
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import fastmag1c

HERE = Path(__file__).parent
WINDOWS = {"50": (2122, 2488), "128": (1550, 2500)}
REPEATS = 5


def timed(rad, t, method, dtype):
    fastmag1c.retrieve(rad[:, :8], template=t, method=method, dtype=dtype)
    times = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fastmag1c.retrieve(rad, template=t, method=method, dtype=dtype)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)


def main(data):
    granules = json.loads((HERE / "granules.json").read_text())
    gpu = torch.cuda.get_device_name()
    rows = []
    for g in granules:
        path = Path(data) / f"{g['granule']}.nc"
        for window, wl_range in WINDOWS.items():
            rad, wl = fastmag1c.read_emit(path, wl_range)
            t = fastmag1c.load_template(wl)
            for dtype in (torch.float64, torch.float32):
                ref = timed(rad, t, "mag1c", dtype)
                new = timed(rad, t, "fastmag1c", dtype)
                name = str(dtype).removeprefix("torch.")
                rows.append([g["name"], rad.shape[0], rad.shape[1], rad.shape[2], name,
                             *ref, *new, ref[0] / new[0], gpu])
                print(g["name"], rad.shape[2], name, f"{ref[0]:.2f}s -> {new[0]:.2f}s", flush=True)

    with open(HERE / "times.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["granule", "downtrack", "crosstrack", "bands", "dtype",
                    "mag1c_mean_s", "mag1c_std_s", "fastmag1c_mean_s", "fastmag1c_std_s",
                    "speedup", "gpu"])
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1])
