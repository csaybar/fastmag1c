Full EMIT granule 20240604T140353 (2528 x 1242 pixels), K = 30, batch_size = 64.
Times in seconds, mean ± standard deviation of 5 runs after a warm-up (10 on the GPU).
The last two columns are the max and P99 of |FastMAG1C - MAG1C| in ppm·m.

| Device | Bands | dtype | MAG1C (s) | FastMAG1C (s) | Speedup | max | P99 |
|---|---|---|---|---|---|---|---|
| cpu | 50 | float64 | 24.43 ± 0.97 | 9.97 ± 0.32 | 2.45 ± 0.13 | 3.6e-05 | 4.9e-09 |
| cpu | 50 | float32 | 9.76 ± 0.45 | 3.82 ± 0.11 | 2.56 ± 0.14 | 1.7e+03 | 2.7e+00 |
| cpu | 128 | float64 | 101.30 ± 3.65 | 34.94 ± 0.88 | 2.90 ± 0.13 | 9.4e-06 | 8.3e-09 |
| cpu | 128 | float32 | 41.87 ± 1.21 | 18.40 ± 0.37 | 2.28 ± 0.08 | 4.0e+03 | 4.8e+00 |
| cuda | 50 | float64 | 3.27 ± 0.09 | 1.10 ± 0.09 | 2.97 ± 0.25 | 2.6e-04 | 2.8e-08 |
| cuda | 50 | float32 | 0.81 ± 0.04 | 0.61 ± 0.02 | 1.34 ± 0.07 | 6.4e+03 | 1.8e+01 |
| cuda | 128 | float64 | 9.81 ± 0.20 | 2.79 ± 0.21 | 3.51 ± 0.27 | 9.5e-05 | 4.7e-08 |
| cuda | 128 | float32 | 1.54 ± 0.07 | 0.92 ± 0.01 | 1.68 ± 0.08 | 1.0e+04 | 2.9e+01 |
