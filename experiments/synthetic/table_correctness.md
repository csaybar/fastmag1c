Error between MAG1C and FastMAG1C in ppm·m, one column per seed, 100 seeds.
P99 and MAE are the mean ± standard deviation over seeds, max is the largest
value over all seeds.

| Device | dtype | N | P99 | MAE | max |
|---|---|---|---|---|---|
| cpu | float64 | 512 | 1.44e-09 ± 2.02e-09 | 1.22e-10 ± 2.45e-10 | 7.87e-07 |
| cpu | float64 | 1024 | 6.85e-10 ± 8.86e-10 | 3.71e-11 ± 4.39e-11 | 9.70e-08 |
| cpu | float64 | 2048 | 3.86e-10 ± 6.11e-10 | 2.27e-11 ± 3.28e-11 | 2.13e-07 |
| cpu | float32 | 512 | 6.55e-01 ± 8.49e-01 | 4.24e-02 ± 6.03e-02 | 7.33e+01 |
| cpu | float32 | 1024 | 2.71e-01 ± 3.15e-01 | 1.67e-02 ± 1.84e-02 | 6.56e+01 |
| cpu | float32 | 2048 | 1.67e-01 ± 1.85e-01 | 9.56e-03 ± 1.01e-02 | 7.19e+01 |
| cuda | float64 | 512 | 1.05e-09 ± 1.43e-09 | 7.85e-11 ± 1.28e-10 | 2.88e-07 |
| cuda | float64 | 1024 | 4.43e-10 ± 5.12e-10 | 2.74e-11 ± 3.67e-11 | 1.88e-07 |
| cuda | float64 | 2048 | 3.53e-10 ± 5.44e-10 | 1.97e-11 ± 2.56e-11 | 1.43e-07 |
| cuda | float32 | 512 | 3.89e-01 ± 5.00e-01 | 2.70e-02 ± 4.42e-02 | 1.64e+02 |
| cuda | float32 | 1024 | 1.61e-01 ± 1.86e-01 | 1.15e-02 ± 1.86e-02 | 1.48e+02 |
| cuda | float32 | 2048 | 1.80e-01 ± 1.88e-01 | 1.04e-02 ± 9.91e-03 | 6.29e+01 |
