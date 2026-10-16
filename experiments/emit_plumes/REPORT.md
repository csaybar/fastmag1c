# MAG1C and FastMAG1C on EMIT plume scenes

We ran both filters on five EMIT L1B granules with plumes from the EMIT L2B
CH4 plume catalog, and on one granule over the Black Sea coast with no
catalogued plume. Every granule was retrieved with 50 bands (2122 to 2488 nm)
and with 128 bands (1550 to 2493 nm), in float64 and in float32, with 30
iterations.

A pixel counts as a plume pixel when its orthorectified position falls inside
one of the catalogued plume polygons. All differences are absolute and in
ppm·m.

## Granules

| Name | Surface | L1B granule | Plumes |
|---|---|---|---|
| turkmenistan | desert, oil and gas | EMIT_L1B_RAD_001_20230607T062333_2315804_033 | 13 |
| permian | semi-arid, oil and gas | EMIT_L1B_RAD_001_20240127T195904_2402713_008 | 3 |
| buenos_aires | urban, landfill | EMIT_L1B_RAD_001_20231124T142642_2332809_049 | 1 |
| campeche | offshore, oil platforms | EMIT_L1B_RAD_001_20240823T173814_2423611_036 | 1 |
| shanxi | mountainous, coal mines | EMIT_L1B_RAD_001_20240217T052731_2404804_010 | 3 |
| black_sea | coastal, no plume | EMIT_L1B_RAD_001_20240604T140353_2415609_030 | 0 |

The plume polygons are in [granules.json](granules.json).

## Float64

Largest |FastMAG1C - MAG1C| with both filters in float64.

| Granule | Plume pixels | 50 bands, all | 50 bands, plume | 128 bands, all | 128 bands, plume |
|---|---|---|---|---|---|
| turkmenistan | 176,925 | 4.6e-07 | 2.9e-07 | 1.6e-06 | 9.3e-07 |
| permian | 93,661 | 5.4e-06 | 1.7e-06 | 1.2e-05 | 5.2e-06 |
| buenos_aires | 105,594 | 1.5e-05 | 8.1e-06 | 9.8e-06 | 8.5e-06 |
| campeche | 57,883 | 6.7e-05 | 1.4e-06 | 2.5e-05 | 5.5e-06 |
| shanxi | 49,916 | 3.0e-04 | 1.1e-05 | 1.6e-05 | 5.2e-06 |
| black_sea | 0 | 2.1e-04 | n/a | 8.9e-05 | n/a |

The worst case over every granule and window is 3e-4 ppm·m, and 1.1e-5 ppm·m
inside the plumes. In float64 the two filters produce the same map. What is
left comes from summing in a different order.

## Float32

Error of each filter in float32 against MAG1C in float64, over plume pixels.

| Granule | 50 bands, MAG1C P99 | 50 bands, FastMAG1C P99 | 50 bands, MAG1C mean | 50 bands, FastMAG1C mean | 128 bands, MAG1C P99 | 128 bands, FastMAG1C P99 |
|---|---|---|---|---|---|---|
| turkmenistan | 0.39 | 0.77 | 0.02 | 0.05 | 1.11 | 2.15 |
| permian | 5.04 | 7.73 | 0.30 | 0.53 | 9.70 | 13.9 |
| buenos_aires | 11.5 | 13.3 | 0.59 | 0.81 | 16.5 | 18.2 |
| campeche | 8.33 | 9.81 | 0.51 | 0.65 | 7.94 | 8.58 |
| shanxi | 19.2 | 29.6 | 2.18 | 3.27 | 33.2 | 45.0 |

Neither filter is exact in float32. MAG1C drifts away from its own float64
result, and FastMAG1C drifts by a similar amount. The few very large
differences in errors.csv, up to a few thousand ppm·m, come from pixels that
flip between zero and a positive value. When MAG1C pushes a pixel to zero, its
weight 1/(R·α) becomes huge and holds it there, so a rounding error close to
that point decides where the pixel ends up. The original MAG1C code computes
in float64 by default for this reason, and we do the same.

## Figures

Each figure crops the scene around the catalogued plumes, drawn in cyan on
the two maps. The last two panels show the signed difference FastMAG1C minus
MAG1C in float64 and in float32, with the color range set to the 99th
percentile of the absolute difference. Pixels where both filters return zero
have no difference at all, so the residual only shows up on pixels with a
positive enhancement, with no spatial pattern. 50 band window.

![turkmenistan](figures/turkmenistan.png)
![permian](figures/permian.png)
![buenos_aires](figures/buenos_aires.png)
![campeche](figures/campeche.png)
![shanxi](figures/shanxi.png)
![black_sea](figures/black_sea.png)

## Runtimes

Mean and standard deviation of 5 runs after a warm-up, from timing.py, on an
NVIDIA A100 80GB.

Another user was training on the same GPU during these runs, so the times are
noisy and the speedups too low. In two cases FastMAG1C even looks slower. We
keep them here for completeness and do not use them as a benchmark. On an idle
RTX A5500 the same code gave 2.7x with 50 bands and 3.9x with 128 bands in
float64 on the black_sea granule.

| Granule | Bands | float64 MAG1C (s) | float64 FastMAG1C (s) | float64 speedup | float32 MAG1C (s) | float32 FastMAG1C (s) | float32 speedup |
|---|---|---|---|---|---|---|---|
| turkmenistan | 50 | 0.85 ± 0.09 | 0.65 ± 0.05 | 1.31x | 0.79 ± 0.06 | 0.59 ± 0.12 | 1.34x |
| turkmenistan | 128 | 1.66 ± 0.07 | 1.26 ± 0.19 | 1.31x | 1.37 ± 0.08 | 1.24 ± 0.05 | 1.11x |
| permian | 50 | 0.98 ± 0.14 | 1.21 ± 0.05 | 0.81x | 0.96 ± 0.13 | 0.78 ± 0.17 | 1.23x |
| permian | 128 | 1.72 ± 0.09 | 1.25 ± 0.06 | 1.38x | 1.25 ± 0.00 | 0.98 ± 0.21 | 1.28x |
| buenos_aires | 50 | 0.47 ± 0.01 | 0.62 ± 0.11 | 0.75x | 0.81 ± 0.12 | 0.61 ± 0.02 | 1.33x |
| buenos_aires | 128 | 1.67 ± 0.07 | 1.27 ± 0.06 | 1.31x | 1.25 ± 0.00 | 0.86 ± 0.12 | 1.45x |
| campeche | 50 | 0.85 ± 0.08 | 0.71 ± 0.07 | 1.19x | 0.78 ± 0.10 | 0.60 ± 0.00 | 1.30x |
| campeche | 128 | 1.64 ± 0.10 | 1.26 ± 0.05 | 1.30x | 1.25 ± 0.00 | 0.86 ± 0.14 | 1.45x |
| shanxi | 50 | 0.82 ± 0.07 | 0.65 ± 0.05 | 1.27x | 0.81 ± 0.07 | 0.59 ± 0.05 | 1.37x |
| shanxi | 128 | 1.72 ± 0.11 | 1.27 ± 0.03 | 1.36x | 1.25 ± 0.00 | 0.86 ± 0.11 | 1.46x |
| black_sea | 50 | 1.20 ± 0.03 | 0.96 ± 0.08 | 1.25x | 1.17 ± 0.09 | 0.69 ± 0.07 | 1.68x |
| black_sea | 128 | 2.45 ± 0.11 | 1.62 ± 0.20 | 1.51x | 1.83 ± 0.11 | 1.09 ± 0.10 | 1.67x |

## Reproduce

```bash
pip install "fastmag1c[emit]" earthaccess matplotlib
python download.py DATA_DIR   # needs an Earthdata login
python run.py DATA_DIR        # errors.csv and figures
python timing.py DATA_DIR     # times.csv, use an idle GPU
```
