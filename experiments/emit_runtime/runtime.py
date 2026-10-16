"""End-to-end runtime on a full EMIT granule (Table III).

python runtime.py GRANULE.nc cpu
python runtime.py GRANULE.nc cuda

Retrieves the whole granule with MAG1C and FastMAG1C through
fastmag1c.retrieve, with the 50 band (2122 to 2488 nm) and 128 band
(1550 to 2500 nm) windows, in float64 and float32. Each configuration runs
once on a few columns to warm up and then RUNS times on the full granule.
Writes one row per run to runtime_cpu.csv or runtime_cuda.csv, and the
difference between the two maps of the last run to agreement_<device>.csv.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

import fastmag1c
from fastmag1c.scene import NODATA

HERE = Path(__file__).parent
WINDOWS = {50: (2122, 2488), 128: (1550, 2500)}
RUNS = 5
BATCH_SIZE = 64

torch.backends.cuda.matmul.allow_tf32 = False


def timed(rad, t, method, device, dtype):
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    kw = dict(template=t, method=method, device=device, dtype=dtype, batch_size=BATCH_SIZE)
    fastmag1c.retrieve(rad[:, :8], **kw)
    times = []
    for _ in range(RUNS):
        sync()
        start = time.perf_counter()
        mf = fastmag1c.retrieve(rad, **kw)
        sync()
        times.append(time.perf_counter() - start)
    return mf, times


def main(path, device):
    runtime = open(HERE / f"runtime_{device}.csv", "w", newline="")
    agreement = open(HERE / f"agreement_{device}.csv", "w", newline="")
    wr, wa = csv.writer(runtime), csv.writer(agreement)
    wr.writerow(["device", "dtype", "bands", "run", "mag1c_s", "fastmag1c_s"])
    wa.writerow(["device", "dtype", "bands", "max", "mean", "p99"])

    for bands, window in WINDOWS.items():
        rad, wl = fastmag1c.read_emit(path, window)
        assert rad.shape[2] == bands, rad.shape
        t = fastmag1c.load_template(wl)
        for dtype in (torch.float64, torch.float32):
            name = str(dtype).removeprefix("torch.")
            ref, t_ref = timed(rad, t, "mag1c", device, dtype)
            new, t_new = timed(rad, t, "fastmag1c", device, dtype)
            for run, (a, b) in enumerate(zip(t_ref, t_new)):
                wr.writerow([device, name, bands, run, a, b])

            valid = ref != NODATA
            d = np.abs(new[valid].astype(np.float64) - ref[valid])
            wa.writerow([device, name, bands, d.max(), d.mean(), np.percentile(d, 99)])
            runtime.flush()
            agreement.flush()
            print(device, name, bands, f"{np.mean(t_ref):.2f} s -> {np.mean(t_new):.2f} s", flush=True)

    runtime.close()
    agreement.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
