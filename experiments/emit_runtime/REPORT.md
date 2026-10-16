# Runtime on a full EMIT granule

MAG1C and FastMAG1C on the full EMIT granule
EMIT_L1B_RAD_001_20240604T140353_2415609_030 (2528 x 1242 pixels), through
fastmag1c.retrieve with batch_size = 64, K = 30 and α = 10⁻⁴. Two spectral
windows, 50 bands (2122 to 2488 nm) and 128 bands (1550 to 2493 nm). Each
configuration runs once on a few columns to warm up and then 5 times on the
whole granule. Hardware and software are listed in
[environment.txt](../environment.txt).

Times are in seconds, mean ± standard deviation of 5 runs on the CPU and 10
on the GPU (runtime_cuda_1.csv and runtime_cuda_2.csv). The last two columns
are the max and P99 of |FastMAG1C - MAG1C| in ppm·m.

| Device | Bands | dtype | MAG1C (s) | FastMAG1C (s) | Speedup | max | P99 |
|---|---|---|---|---|---|---|---|
| CPU | 50 | float64 | 24.43 ± 0.97 | 9.97 ± 0.32 | 2.45 ± 0.13 | 3.6e-05 | 4.9e-09 |
| CPU | 50 | float32 | 9.76 ± 0.45 | 3.82 ± 0.11 | 2.56 ± 0.14 | 1.7e+03 | 2.7e+00 |
| CPU | 128 | float64 | 101.30 ± 3.65 | 34.94 ± 0.88 | 2.90 ± 0.13 | 9.4e-06 | 8.3e-09 |
| CPU | 128 | float32 | 41.87 ± 1.21 | 18.40 ± 0.37 | 2.28 ± 0.08 | 4.0e+03 | 4.8e+00 |
| GPU | 50 | float64 | 3.27 ± 0.09 | 1.10 ± 0.09 | 2.97 ± 0.25 | 2.6e-04 | 2.8e-08 |
| GPU | 50 | float32 | 0.81 ± 0.04 | 0.61 ± 0.02 | 1.34 ± 0.07 | 6.4e+03 | 1.8e+01 |
| GPU | 128 | float64 | 9.81 ± 0.20 | 2.79 ± 0.21 | 3.51 ± 0.27 | 9.5e-05 | 4.7e-08 |
| GPU | 128 | float32 | 1.54 ± 0.07 | 0.92 ± 0.01 | 1.68 ± 0.08 | 1.0e+04 | 2.9e+01 |

FastMAG1C is faster in every configuration. The gain is largest in float64
on the GPU, 3.0x with 50 bands and 3.5x with 128 bands, and on CPU it stays
between 2.3x and 2.9x. On the GPU in float32 each batch of 64 columns runs so fast
that kernel launches dominate, and the speedup drops to 1.3x and 1.7x. With
larger batches it recovers, to 2.05x (50 bands) and 1.85x (128 bands) with
256 columns. The batch size sweep is in batch_size_gpu.txt (bands, dtype,
batch size, MAG1C s, FastMAG1C s, speedup).

In float64 both filters give the same map, with a largest difference of
2.6e-4 ppm·m. In float32 the P99 difference is a few ppm·m on CPU and tens of
ppm·m on GPU, well below the usual 500 ppm·m detection threshold. The max
values come from single pixels that end at zero in one filter and above zero
in the other. The same comparison on five granules with catalogued plumes is
in [emit_plumes](../emit_plumes/REPORT.md).

## Reproduce

```bash
python runtime.py EMIT_L1B_RAD_001_20240604T140353_2415609_030.nc cpu
python runtime.py EMIT_L1B_RAD_001_20240604T140353_2415609_030.nc cuda
python summarize.py   # table_runtime.md
```
