"""MAG1C vs FastMAG1C on EMIT granules with catalogued CH4 plumes.

python run.py DATA_DIR

For every granule in granules.json and for two spectral windows, runs both
filters in float32 and float64 and writes results.csv. Plume pixels are the
sensor pixels whose orthorectified position falls inside a plume polygon of
the EMIT L2B CH4 plume catalog. Writes errors.csv and one figure per
granule; runtimes are measured separately by timing.py. Granules not yet
downloaded are skipped.
"""

import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.path import Path as Polygon

import fastmag1c
from fastmag1c.scene import NODATA

HERE = Path(__file__).parent
WINDOWS = {"50": (2122, 2488), "128": (1550, 2500)}


def plume_mask(path, plumes):
    """Sensor-geometry mask of the pixels inside the plume polygons."""
    with h5py.File(path, "r") as f:
        gx = f["location/glt_x"][:].astype(np.int64)
        gy = f["location/glt_y"][:].astype(np.int64)
        x0, dx, _, y0, _, dy = f.attrs["geotransform"]
        shape = f["radiance"].shape[:2]
    mask = np.zeros(shape, dtype=bool)
    if not plumes:
        return mask
    rows, cols = np.nonzero((gx > 0) & (gy > 0))
    lonlat = np.column_stack([x0 + (cols + 0.5) * dx, y0 + (rows + 0.5) * dy])
    inside = np.zeros(len(rows), dtype=bool)
    for p in plumes:
        inside |= Polygon(p["polygon"]).contains_points(lonlat)
    r, c = rows[inside], cols[inside]
    mask[gy[r, c] - 1, gx[r, c] - 1] = True
    return mask


def errors(a, b, where):
    d = np.abs(a[where] - b[where])
    if d.size == 0:
        return [np.nan] * 3
    return [d.max(), d.mean(), np.percentile(d, 99)]


def figure(path, maps, mask, name, out):
    """Orthorectified crop around the plumes: maps in float64 and both differences."""
    ortho = {k: fastmag1c.orthorectify(v, path)[0] for k, v in maps.items()}
    m = fastmag1c.orthorectify(mask.astype(np.float32), path, nodata=0)[0] > 0
    if m.any():
        r, c = np.nonzero(m)
        pad = 60
        sl = (slice(max(r.min() - pad, 0), r.max() + pad), slice(max(c.min() - pad, 0), c.max() + pad))
    else:
        sl = (slice(None), slice(None))
    invalid = ortho["mag1c_f64"][sl] == NODATA

    def residual(a, b):
        # signed difference, color limits at the 99th percentile of |d|
        d = ortho[a][sl] - ortho[b][sl]
        lim = np.percentile(np.abs(d[~invalid]), 99)
        return d, "RdBu_r", (-lim, lim)

    panels = [
        ("MAG1C, float64", ortho["mag1c_f64"][sl], "inferno", (0, 1000)),
        ("FastMAG1C, float64", ortho["fast_f64"][sl], "inferno", (0, 1000)),
        ("FastMAG1C - MAG1C, float64", *residual("fast_f64", "mag1c_f64")),
        ("FastMAG1C - MAG1C, float32", *residual("fast_f32", "mag1c_f32")),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), constrained_layout=True)
    for i, (ax, (title, img, cmap, (lo, hi))) in enumerate(zip(axes, panels)):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("0.85")
        im = ax.imshow(np.ma.masked_where(invalid, img), cmap=cm, vmin=lo, vmax=hi,
                       interpolation="nearest")
        if i < 2 and m.any():
            ax.contour(m[sl], levels=[0.5], colors="cyan", linewidths=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        cb = fig.colorbar(im, ax=ax, shrink=0.8, label="ppm·m")
        if i >= 2:
            cb.formatter.set_powerlimits((-2, 2))
            cb.update_ticks()
    fig.suptitle(name)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(data):
    granules = json.loads((HERE / "granules.json").read_text())
    (HERE / "figures").mkdir(exist_ok=True)
    rows = []
    for g in granules:
        path = Path(data) / f"{g['granule']}.nc"
        if not path.exists():
            print("missing", path.name)
            continue
        mask = plume_mask(path, g["plumes"])
        for window, wl_range in WINDOWS.items():
            rad, wl = fastmag1c.read_emit(path, wl_range)
            t = fastmag1c.load_template(wl)
            maps = {}
            for dtype in (torch.float64, torch.float32):
                c = fastmag1c.compare(rad, template=t, dtype=dtype, repeats=1)
                key = str(dtype).removeprefix("torch.float")
                maps[f"mag1c_f{key}"], maps[f"fast_f{key}"] = c.mag1c, c.fastmag1c

            valid = maps["mag1c_f64"] != NODATA
            for region, where in [("all", valid), ("plume", valid & mask)]:
                for label, a, b in [
                    ("fast_vs_mag1c_f64", "fast_f64", "mag1c_f64"),
                    ("fast_vs_mag1c_f32", "fast_f32", "mag1c_f32"),
                    ("mag1c_f32_vs_f64", "mag1c_f32", "mag1c_f64"),
                    ("fast_f32_vs_mag1c_f64", "fast_f32", "mag1c_f64"),
                ]:
                    rows.append([g["name"], window, region, int(where.sum()), label,
                                 *errors(maps[a], maps[b], where)])
            print(g["name"], window, "bands done", flush=True)

            if window == "50":
                figure(path, maps, mask, f"{g['name']} ({g['surface']})",
                       HERE / "figures" / f"{g['name']}.png")

    with open(HERE / "errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["granule", "bands", "region", "n_pixels", "comparison", "max", "mean", "p99"])
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1])
