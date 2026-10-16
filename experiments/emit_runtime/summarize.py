"""Table III from runtime_*.csv and agreement_{cpu,cuda}.csv.

python summarize.py

Writes table_runtime.md. Times are the mean ± standard deviation of the
runs, and the speedup error is propagated from both means.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def read(name):
    path = HERE / name
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# the GPU runs are in runtime_cuda_1.csv and runtime_cuda_2.csv
times = defaultdict(lambda: ([], []))
rows = [r for f in sorted(HERE.glob("runtime_*.csv")) for r in read(f.name)]
for r in rows:
    key = (r["device"], int(r["bands"]), r["dtype"])
    times[key][0].append(float(r["mag1c_s"]))
    times[key][1].append(float(r["fastmag1c_s"]))

agree = {(r["device"], int(r["bands"]), r["dtype"]): r
         for r in read("agreement_cpu.csv") + read("agreement_cuda.csv")}

lines = [
    "Full EMIT granule 20240604T140353 (2528 x 1242 pixels), K = 30, batch_size = 64.",
    "Times in seconds, mean ± standard deviation of 5 runs after a warm-up (10 on the GPU).",
    "The last two columns are the max and P99 of |FastMAG1C - MAG1C| in ppm·m.",
    "",
    "| Device | Bands | dtype | MAG1C (s) | FastMAG1C (s) | Speedup | max | P99 |",
    "|---|---|---|---|---|---|---|---|",
]
order = sorted(times, key=lambda k: (k[0], k[1], k[2] != "float64"))
for key in order:
    a, b = map(np.array, times[key])
    sp = a.mean() / b.mean()
    err = sp * np.hypot(a.std() / a.mean(), b.std() / b.mean())
    g = agree.get(key, {})
    lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {a.mean():.2f} ± {a.std():.2f} | "
                 f"{b.mean():.2f} ± {b.std():.2f} | {sp:.2f} ± {err:.2f} | "
                 f"{float(g.get('max', 'nan')):.1e} | {float(g.get('p99', 'nan')):.1e} |")

(HERE / "table_runtime.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
