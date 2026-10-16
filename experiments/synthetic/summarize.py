"""Table II and Fig. 1 from correctness.csv and scaling_{cpu,cuda}.csv.

python summarize.py

Writes table_correctness.md, table_scaling.md and scaling_speedup.pdf/png.
The speedup of a configuration is mean(MAG1C time) / mean(FastMAG1C time)
over the runs, and its error bar is the standard deviation propagated from
both means.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

HERE = Path(__file__).parent
K = 30
COLORS = {50: "#4477AA", 100: "#EE6677", 150: "#228833", 200: "#CCBB44", 285: "#AA3377"}
MARKERS = {50: "o", 100: "s", 150: "D", 200: "^", 285: "v"}


def read(name):
    with open(HERE / name) as f:
        return list(csv.DictReader(f))


def correctness_table():
    groups = defaultdict(list)
    for r in read("correctness.csv"):
        groups[(r["device"], r["dtype"], int(r["N"]))].append(
            [float(r["p99"]), float(r["mean"]), float(r["max"])])

    lines = [
        "Error between MAG1C and FastMAG1C in ppm·m, one column per seed, 100 seeds.",
        "P99 and MAE are the mean ± standard deviation over seeds, max is the largest",
        "value over all seeds.",
        "",
        "| Device | dtype | N | P99 | MAE | max |",
        "|---|---|---|---|---|---|",
    ]
    for (device, dtype, N), v in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] != "float64", kv[0][2])):
        v = np.array(v)
        p99, mae, mx = v[:, 0], v[:, 1], v[:, 2]
        lines.append(f"| {device} | {dtype} | {N} | {p99.mean():.2e} ± {p99.std():.2e} | "
                     f"{mae.mean():.2e} ± {mae.std():.2e} | {mx.max():.2e} |")
    (HERE / "table_correctness.md").write_text("\n".join(lines) + "\n")


def speedups(rows):
    times = defaultdict(lambda: ([], []))
    for r in rows:
        key = (r["device"], r["dtype"], int(r["s"]), int(r["N"]))
        times[key][0].append(float(r["mag1c_ms"]))
        times[key][1].append(float(r["fastmag1c_ms"]))
    out = {}
    for key, (a, b) in times.items():
        a, b = np.array(a), np.array(b)
        sp = a.mean() / b.mean()
        err = sp * np.hypot(a.std() / a.mean(), b.std() / b.mean())
        out[key] = (a.mean(), a.std(), b.mean(), b.std(), sp, err)
    return out


def scaling_table(sp):
    lines = [
        "Runtime in ms (mean ± standard deviation of 5 runs) for a batch of 10 columns.",
        "",
        "| Device | dtype | s | N | MAG1C | FastMAG1C | Speedup | Ks/(s+K) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (device, dtype, s, N), (am, asd, bm, bsd, x, e) in sorted(sp.items()):
        lines.append(f"| {device} | {dtype} | {s} | {N} | {am:.1f} ± {asd:.1f} | "
                     f"{bm:.1f} ± {bsd:.1f} | {x:.2f} ± {e:.2f} | {K * s / (s + K):.1f} |")
    (HERE / "table_scaling.md").write_text("\n".join(lines) + "\n")


def figure(sp, dtype="float64"):
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05), sharey=True)
    for ax, (device, label) in zip(axes, [("cpu", "(a) CPU"), ("cuda", "(b) GPU")]):
        for s in COLORS:
            pts = sorted((N, v[4], v[5]) for (d, t, ss, N), v in sp.items()
                         if d == device and t == dtype and ss == s)
            if not pts:
                continue
            N, y, e = map(np.array, zip(*pts))
            ax.errorbar(N, y, yerr=e, marker=MARKERS[s], color=COLORS[s], label=f"$s={s}$",
                        markersize=2.5, lw=0.9, capsize=1.5, elinewidth=0.6)
            ax.axhline(K * s / (s + K), color=COLORS[s], lw=0.6, ls="--", alpha=0.8)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_yticks([1, 2, 4, 8, 16, 32])
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylim(0.8, 34)
        ax.set_xticks([512, 1024, 2048, 4096, 8192, 16384])
        ax.set_xticklabels(["512", "1K", "2K", "4K", "8K", "16K"])
        ax.tick_params(labelsize=5.5, length=2, pad=1.5)
        ax.axhline(1, color="k", lw=0.4, alpha=0.3)
        ax.grid(True, which="both", alpha=0.15, lw=0.4)
        ax.set_title(label, fontsize=6.5, fontweight="bold", loc="left", pad=2)
        ax.set_xlabel("$N$ (pixels per column)", fontsize=6, labelpad=1)
        ax.text(600, 13, "dashed: $Ks/(s+K)$", fontsize=5, color="0.35")
    axes[0].set_ylabel("Speedup", fontsize=6, labelpad=1)
    handles, labels = max((ax.get_legend_handles_labels() for ax in axes), key=lambda h: len(h[1]))
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=5.5, frameon=False,
               handlelength=1.4, columnspacing=0.8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=0.2, w_pad=0.4, rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"scaling_speedup.{ext}", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    correctness_table()
    rows = []
    for device in ("cpu", "cuda"):
        if (HERE / f"scaling_{device}.csv").exists():
            rows += read(f"scaling_{device}.csv")
    sp = speedups(rows)
    scaling_table(sp)
    figure(sp)
