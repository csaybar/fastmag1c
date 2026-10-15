# fastmag1c

MAG1C methane retrieval with an exact rank-2 covariance update.

MAG1C recomputes the background covariance of each CCD column at every
iteration, which costs O(Ns²) per step. The update is a rank-2 change of a
fixed base covariance, so it can be done in O(Ns) with the same result.

## Installation

```bash
pip install "fastmag1c[emit] @ git+https://github.com/csaybar/fastmag1c"
pip install earthaccess  # only to download the example granule
```

## Quick start

```python
import earthaccess
import fastmag1c

# EMIT L1B radiance from NASA LP DAAC, needs a free Earthdata account
earthaccess.login()
granule = earthaccess.search_data(short_name="EMITL1BRAD", granule_name="*20240604T140353_2415609_030*")
url = [u for u in granule[0].data_links() if "EMIT_L1B_RAD" in u]
path = earthaccess.download(url, ".")[0]

rad, wl = fastmag1c.read_emit(path)
fastmag1c.compare(rad, wl, reference=True)
```

`compare` runs both filters, times them (mean of 3 runs after a warm-up) and
reports the difference over all pixels and over plume pixels. With `reference=True` it also runs MAG1C in
float64, to show the float32 error of each method.

To get only the map:

```python
ch4 = fastmag1c.retrieve(rad, wl)  # (downtrack, crosstrack) in ppm·m

# lat/lon grid, GDAL geotransform and CRS (WKT) of the granule
ch4_map, transform, crs = fastmag1c.orthorectify(ch4, path)
```

## Results on EMIT

Granule `20240604T140353_2415609_030` (2528 × 1242 pixels, 50 bands in
2122–2488 nm, 30 iterations). Times are the mean of 3 runs after a warm-up.
Differences are in ppm·m, measured on the RTX A5500 (the A100 gives nearly the
same values), and plume pixels are the 185,015 pixels above 500 ppm·m.

| GPU | dtype | MAG1C (s) | FastMAG1C (s) | Speedup |
|---|---|---:|---:|---:|
| RTX A5500 | float64 | 3.29 | 1.22 | 2.70× |
| RTX A5500 | float32 | 0.89 | 0.78 | 1.15× |
| A100 | float64 | 0.71 | 0.54 | 1.30× |
| A100 | float32 | 0.66 | 0.35 | 1.90× |

| Difference | dtype | max | P99 | P99, plume |
|---|---|---:|---:|---:|
| FastMAG1C − MAG1C | float64 | 2.7e-4 | 2.8e-8 | 2.4e-7 |
| FastMAG1C − MAG1C | float32 | 6.4e3 | 17.9 | 146 |
| MAG1C float32 − MAG1C float64 | | 5.9e3 | 11.0 | 79 |
| FastMAG1C float32 − MAG1C float64 | | 6.0e3 | 14.9 | 128 |

In float64 both methods give the same map. In float32 MAG1C itself already
departs from its float64 result by a similar amount, so use float64 when the
values matter.

## Citation

```bibtex
@article{aybar2026fastmag1c,
  title   = {Fast Iterative Methane Retrieval Through Algebraically Exact Rank-2 Acceleration},
  author  = {Aybar, Cesar and Contreras, Julio and G{\'o}mez-Chova, Luis},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2026}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
