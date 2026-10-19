### Samples

| restaurant | variant | train orders | holdout orders | cut | groups/order |
|---|---|---|---|---|---|
| 1 | all | 8829 | 4539 | 2018-10-07 | 4.85 |
| 1 | small | 1654 | 799 | 2018-11-12 | 3.01 |
| 1 | all_size | 8829 | 4539 | 2018-10-07 | 4.85 |
| 2 | all | 13915 | 5742 | 2018-10-27 | 5.27 |
| 2 | small | 1987 | 913 | 2018-11-25 | 2.95 |
| 2 | all_size | 13915 | 5742 | 2018-10-27 | 5.27 |

### Holdout joint basket log-likelihood (nats per order, 21 groups)

| r | variant | M0 | M1 | M1 reversed | M3 | M3 − M0 | M3 − M1 | M3 − M1 rev |
|---|---|---|---|---|---|---|---|---|
| 1 | all | -10.483 | -9.929 | -9.932 | -9.977 | +0.505 [+0.475, +0.536] | -0.049 [-0.058, -0.041] | -0.045 [-0.054, -0.037] |
| 1 | small | -8.525 | -7.485 | -7.482 | -7.907 | +0.617 [+0.531, +0.702] | -0.422 [-0.479, -0.370] | -0.425 [-0.481, -0.369] |
| 1 | all_size | -9.925 | -9.290 | -9.268 | -9.398 | +0.527 [+0.487, +0.560] | -0.109 [-0.128, -0.091] | -0.130 [-0.152, -0.110] |
| 2 | all | -10.653 | -10.033 | -10.033 | -10.078 | +0.574 [+0.544, +0.603] | -0.045 [-0.052, -0.037] | -0.046 [-0.053, -0.039] |
| 2 | small | -7.828 | -6.883 | -6.858 | -7.282 | +0.545 [+0.463, +0.626] | -0.399 [-0.454, -0.344] | -0.424 [-0.477, -0.367] |
| 2 | all_size | -10.066 | -9.357 | -9.336 | -9.451 | +0.614 [+0.586, +0.640] | -0.095 [-0.106, -0.082] | -0.115 [-0.127, -0.103] |

### Coded-subset sub-basket log-likelihood (5 groups, nats per order)

| r | variant | M2 groups | M0 | M1 (MC) | M2 | M3 | M3 − M2 | M3 − M0 |
|---|---|---|---|---|---|---|---|---|
| 1 | all | mild_curry, veg_side, naan_flav, rice_pilau, naan_plain | -3.259 | -3.231 | -3.191 | -3.191 | -0.001 [-0.007, +0.006] | +0.067 [+0.056, +0.079] |
| 1 | small | mild_curry, naan_flav, papadum, starter_fried, rice_pilau | -2.583 | -2.626 | -2.704 | -2.585 | +0.119 [+0.071, +0.172] | -0.001 [-0.014, +0.011] |
| 1 | all_size | mild_curry, veg_side, naan_flav, rice_pilau, naan_plain | -3.098 | -3.077 | -3.044 | -3.043 | +0.001 [-0.006, +0.008] | +0.056 [+0.045, +0.066] |
| 2 | all | mild_curry, veg_side, rice_pilau, naan_flav, naan_plain | -3.306 | -3.245 | -3.198 | -3.209 | -0.011 [-0.018, -0.005] | +0.097 [+0.086, +0.107] |
| 2 | small | hot_curry, naan_plain, starter_fried, rice_pilau, veg_side | -2.562 | -2.568 | -2.699 | -2.530 | +0.169 [+0.095, +0.246] | +0.032 [+0.012, +0.053] |
| 2 | all_size | mild_curry, veg_side, rice_pilau, naan_flav, naan_plain | -3.115 | -3.071 | -3.016 | -3.033 | -0.017 [-0.024, -0.009] | +0.082 [+0.072, +0.092] |

### All 210 pairs: mean absolute error of the co-occurrence rate, pts

| r | variant | M0 | M1 | M1 rev | M3 | M0 − M3 | M1 − M3 | M1 rev − M3 |
|---|---|---|---|---|---|---|---|---|
| 1 | all | 0.936 | 0.540 | 0.551 | 0.551 | +0.385 [+0.314, +0.430] | -0.010 [-0.019, -0.002] | +0.000 [-0.010, +0.011] |
| 1 | small | 0.977 | 0.557 | 0.567 | 0.663 | +0.314 [+0.247, +0.360] | -0.106 [-0.118, -0.070] | -0.096 [-0.110, -0.063] |
| 1 | all_size | 0.929 | 0.539 | 0.557 | 0.587 | +0.342 [+0.283, +0.386] | -0.048 [-0.066, -0.025] | -0.029 [-0.042, -0.007] |
| 2 | all | 1.014 | 0.644 | 0.651 | 0.656 | +0.357 [+0.289, +0.425] | -0.012 [-0.016, +0.001] | -0.006 [-0.010, +0.007] |
| 2 | small | 1.086 | 0.660 | 0.649 | 0.769 | +0.318 [+0.247, +0.349] | -0.109 [-0.134, -0.095] | -0.120 [-0.142, -0.104] |
| 2 | all_size | 1.066 | 0.621 | 0.625 | 0.673 | +0.393 [+0.344, +0.430] | -0.052 [-0.066, -0.029] | -0.048 [-0.060, -0.026] |

### Top 50 triples (train frequency): mean absolute error of the co-occurrence rate, pts

| r | variant | M0 | M1 | M1 rev | M3 | M0 − M3 | M1 − M3 | M1 rev − M3 |
|---|---|---|---|---|---|---|---|---|
| 1 | all | 1.787 | 0.641 | 0.654 | 0.582 | +1.205 [+0.804, +1.469] | +0.059 [-0.018, +0.102] | +0.071 [-0.011, +0.113] |
| 1 | small | 1.315 | 0.602 | 0.602 | 0.618 | +0.698 [+0.413, +0.822] | -0.015 [-0.046, +0.032] | -0.016 [-0.068, +0.049] |
| 1 | all_size | 1.035 | 0.537 | 0.625 | 0.649 | +0.386 [+0.257, +0.477] | -0.112 [-0.149, -0.034] | -0.024 [-0.104, +0.081] |
| 2 | all | 2.402 | 0.718 | 0.708 | 0.654 | +1.748 [+1.273, +2.097] | +0.064 [-0.005, +0.112] | +0.054 [-0.006, +0.104] |
| 2 | small | 1.151 | 0.790 | 0.795 | 0.809 | +0.342 [+0.139, +0.436] | -0.020 [-0.040, +0.016] | -0.014 [-0.043, +0.018] |
| 2 | all_size | 1.292 | 0.646 | 0.622 | 0.739 | +0.553 [+0.398, +0.689] | -0.093 [-0.127, -0.022] | -0.116 [-0.190, +0.021] |

### All 546 bundles (decision test): mean absolute error of the co-occurrence rate, pts

| r | variant | M0 | M1 | M1 rev | M3 | M0 − M3 | M1 − M3 | M1 rev − M3 |
|---|---|---|---|---|---|---|---|---|
| 1 | all | 0.439 | 0.295 | 0.303 | 0.288 | +0.150 [+0.102, +0.175] | +0.007 [+0.001, +0.012] | +0.014 [+0.007, +0.020] |
| 1 | small | 0.351 | 0.231 | 0.234 | 0.267 | +0.084 [+0.062, +0.099] | -0.036 [-0.045, -0.029] | -0.033 [-0.044, -0.026] |
| 1 | all_size | 0.463 | 0.285 | 0.293 | 0.311 | +0.153 [+0.120, +0.171] | -0.026 [-0.041, -0.006] | -0.018 [-0.030, -0.000] |
| 2 | all | 0.461 | 0.325 | 0.328 | 0.316 | +0.145 [+0.095, +0.186] | +0.008 [+0.003, +0.014] | +0.012 [+0.006, +0.018] |
| 2 | small | 0.397 | 0.219 | 0.211 | 0.275 | +0.122 [+0.093, +0.128] | -0.056 [-0.066, -0.050] | -0.064 [-0.072, -0.056] |
| 2 | all_size | 0.538 | 0.308 | 0.305 | 0.345 | +0.193 [+0.163, +0.209] | -0.037 [-0.050, -0.016] | -0.040 [-0.051, -0.019] |

### Events inside M2's coded subset (all models on the same events), MAE pts

| r | variant | block | n | M0 | M1 | M2 | M3 |
|---|---|---|---|---|---|---|---|
| 1 | all | pairs | 10 | 1.86 [1.60, 2.22] | 1.30 [0.91, 1.87] | 1.25 [0.87, 1.80] | 1.16 [0.79, 1.74] |
| 1 | all | triples | 7 | 1.24 [0.97, 1.68] | 1.20 [0.80, 1.67] | 0.97 [0.63, 1.43] | 0.99 [0.62, 1.45] |
| 1 | all | bundles | 10 | 1.80 [1.49, 2.14] | 1.14 [0.76, 1.68] | 0.98 [0.63, 1.50] | 0.89 [0.53, 1.43] |
| 1 | small | pairs | 10 | 1.05 [0.80, 1.59] | 0.93 [0.77, 1.50] | 0.91 [0.77, 1.50] | 0.95 [0.79, 1.55] |
| 1 | small | triples | 6 | 0.09 [0.14, 0.49] | 0.66 [0.43, 0.98] | 0.64 [0.39, 0.97] | 0.64 [0.38, 0.95] |
| 1 | small | bundles | 10 | 0.50 [0.42, 0.98] | 0.93 [0.64, 1.42] | 0.98 [0.68, 1.48] | 0.92 [0.68, 1.43] |
| 1 | all_size | pairs | 10 | 1.88 [1.63, 2.25] | 0.96 [0.64, 1.38] | 0.95 [0.64, 1.36] | 1.08 [0.76, 1.58] |
| 1 | all_size | triples | 7 | 1.14 [0.89, 1.46] | 0.85 [0.54, 1.28] | 0.67 [0.45, 1.10] | 1.13 [0.74, 1.58] |
| 1 | all_size | bundles | 10 | 1.46 [1.20, 1.76] | 0.83 [0.51, 1.26] | 0.72 [0.44, 1.12] | 0.98 [0.60, 1.49] |
| 2 | all | pairs | 10 | 2.28 [2.06, 2.67] | 1.25 [0.93, 1.71] | 1.32 [0.98, 1.78] | 1.16 [0.85, 1.63] |
| 2 | all | triples | 7 | 1.88 [1.47, 2.32] | 1.29 [0.84, 1.79] | 1.07 [0.68, 1.58] | 1.06 [0.62, 1.56] |
| 2 | all | bundles | 10 | 1.99 [1.71, 2.34] | 1.79 [1.27, 2.35] | 1.67 [1.15, 2.22] | 1.51 [0.98, 2.06] |
| 2 | small | pairs | 10 | 1.90 [1.52, 2.52] | 1.31 [0.97, 1.92] | 1.26 [0.94, 1.89] | 1.28 [0.95, 1.88] |
| 2 | small | triples | 5 | 0.89 [0.54, 1.53] | 0.58 [0.42, 1.01] | 0.32 [0.21, 0.86] | 0.56 [0.39, 0.96] |
| 2 | small | bundles | 10 | 1.35 [1.00, 1.81] | 0.53 [0.37, 0.96] | 0.47 [0.31, 0.91] | 0.69 [0.54, 1.09] |
| 2 | all_size | pairs | 10 | 2.11 [1.87, 2.42] | 1.06 [0.83, 1.47] | 1.06 [0.82, 1.48] | 1.24 [0.93, 1.65] |
| 2 | all_size | triples | 7 | 1.06 [0.85, 1.31] | 1.03 [0.67, 1.47] | 0.78 [0.54, 1.20] | 1.37 [0.94, 1.80] |
| 2 | all_size | bundles | 10 | 1.67 [1.42, 1.94] | 1.49 [1.00, 2.00] | 1.32 [0.85, 1.85] | 1.76 [1.28, 2.28] |

### Ranking bundles by full-bundle rate (holdout observed vs model)

| r | variant | model | top-10 overlap /10 | Spearman (546) | MAE on observed top 10, pts |
|---|---|---|---|---|---|
| 1 | all | M0 | 8 [8, 9] | 0.955 [0.938, 0.955] | 1.93 [1.59, 2.34] |
| 1 | all | M1 | 9 [9, 10] | 0.968 [0.954, 0.969] | 0.65 [0.54, 1.12] |
| 1 | all | M1_rev | 9 [9, 10] | 0.968 [0.953, 0.968] | 0.70 [0.57, 1.18] |
| 1 | all | M3 | 9 [9, 10] | 0.968 [0.954, 0.969] | 0.62 [0.49, 1.05] |
| 1 | all | M3_raw | 9 [9, 10] | 0.968 [0.953, 0.968] | 0.63 [0.52, 1.09] |
| 1 | small | M0 | 7 [5, 8] | 0.710 [0.652, 0.699] | 2.76 [2.14, 3.42] |
| 1 | small | M1 | 8 [6, 9] | 0.811 [0.751, 0.793] | 1.30 [0.84, 2.00] |
| 1 | small | M1_rev | 7 [6, 8] | 0.814 [0.752, 0.793] | 1.23 [0.81, 1.94] |
| 1 | small | M3 | 7 [6, 8] | 0.794 [0.732, 0.776] | 1.40 [0.94, 2.11] |
| 1 | small | M3_raw | 7 [6, 8] | 0.792 [0.730, 0.774] | 1.33 [0.90, 2.05] |
| 1 | all_size | M0 | 8 [8, 9] | 0.962 [0.946, 0.961] | 1.24 [1.00, 1.60] |
| 1 | all_size | M1 | 9 [9, 10] | 0.968 [0.953, 0.968] | 0.68 [0.52, 1.06] |
| 1 | all_size | M1_rev | 9 [9, 10] | 0.968 [0.953, 0.968] | 0.75 [0.60, 1.15] |
| 1 | all_size | M3 | 9 [9, 10] | 0.969 [0.954, 0.969] | 0.63 [0.49, 1.01] |
| 1 | all_size | M3_raw | 9 [9, 10] | 0.968 [0.953, 0.968] | 0.64 [0.52, 1.04] |
| 2 | all | M0 | 9 [8, 9] | 0.967 [0.957, 0.966] | 1.73 [1.46, 2.13] |
| 2 | all | M1 | 9 [8, 9] | 0.976 [0.966, 0.975] | 1.24 [0.99, 1.74] |
| 2 | all | M1_rev | 9 [8, 9] | 0.975 [0.965, 0.975] | 1.24 [1.00, 1.75] |
| 2 | all | M3 | 9 [8, 9] | 0.976 [0.966, 0.975] | 1.20 [0.93, 1.62] |
| 2 | all | M3_raw | 9 [8, 9] | 0.975 [0.965, 0.974] | 1.25 [0.98, 1.67] |
| 2 | small | M0 | 4 [3, 5] | 0.606 [0.553, 0.610] | 2.93 [2.37, 3.54] |
| 2 | small | M1 | 7 [5, 8] | 0.733 [0.674, 0.727] | 1.41 [0.93, 2.12] |
| 2 | small | M1_rev | 7 [5, 8] | 0.730 [0.675, 0.726] | 1.33 [0.85, 2.02] |
| 2 | small | M3 | 6 [4, 7] | 0.690 [0.642, 0.690] | 1.29 [0.95, 2.00] |
| 2 | small | M3_raw | 6 [4, 7] | 0.688 [0.640, 0.689] | 1.24 [0.90, 1.93] |
| 2 | all_size | M0 | 9 [8, 9] | 0.970 [0.960, 0.969] | 1.53 [1.34, 1.85] |
| 2 | all_size | M1 | 9 [8, 9] | 0.975 [0.965, 0.975] | 1.17 [0.91, 1.59] |
| 2 | all_size | M1_rev | 9 [8, 9] | 0.976 [0.966, 0.975] | 1.17 [0.96, 1.64] |
| 2 | all_size | M3 | 9 [8, 9] | 0.977 [0.967, 0.976] | 1.18 [0.93, 1.60] |
| 2 | all_size | M3_raw | 9 [8, 9] | 0.976 [0.966, 0.975] | 1.20 [0.95, 1.62] |

### Shrinkage check: M3 (s = 0.05) vs unshrunk pairwise-IFM correlation

| r | variant | raw min eig | LL M3 − M3 raw | pairs M3 raw − M3 | bundles M3 raw − M3 |
|---|---|---|---|---|---|
| 1 | all | 1.0e-08 | +0.012 [+0.004, +0.022] | -0.002 [-0.006, +0.003] | +0.006 [+0.003, +0.008] |
| 1 | small | 1.0e-08 | +0.054 [+0.010, +0.111] | -0.007 [-0.011, +0.005] | -0.001 [-0.002, +0.001] |
| 1 | all_size | 1.0e-08 | +24266.049 [-0.005, +72798.155] | -0.004 [-0.007, +0.000] | +0.000 [-0.001, +0.003] |
| 2 | all | 1.0e-08 | +0.012 [+0.007, +0.018] | +0.001 [-0.004, +0.008] | +0.008 [+0.006, +0.010] |
| 2 | small | 1.0e-08 | +10117.700 [+719.704, +24378.053] | +0.000 [-0.009, +0.003] | -0.001 [-0.003, +0.000] |
| 2 | all_size | 1.0e-08 | +0.291 [-0.003, +0.884] | -0.007 [-0.010, -0.004] | -0.001 [-0.002, +0.001] |

### Restaurant 1 (all orders): M3 top complements and substitutes

| complement pair | ρ | substitute pair | ρ |
|---|---|---|---|
| papadum + condiment | +0.74 | naan_plain + naan_flav | -0.34 |
| fries + sauce | +0.34 | rice_pilau + rice_special | -0.34 |
| mild_curry + rice_pilau | +0.29 | rice_pilau + rice_plain | -0.29 |
| rice_pilau + naan_plain | +0.26 | biryani + rice_pilau | -0.27 |
| rice_pilau + papadum | +0.25 | rice_plain + rice_special | -0.26 |
| mild_curry + naan_plain | +0.24 | mild_curry + biryani | -0.25 |
| mild_curry + naan_flav | +0.23 | biryani + rice_plain | -0.22 |
| tandoori_main + sauce | +0.22 | biryani + rice_special | -0.22 |
| naan_plain + fries | +0.21 | naan_flav + bread_other | -0.21 |
| rice_plain + naan_plain | +0.20 | mild_curry + tandoori_main | -0.21 |

### Restaurant 1 (all orders): top 15 bundles by observed holdout full-bundle rate (%)

| bundle | observed | M0 | M1 | M3 | M0 err | M1 err | M3 err |
|---|---|---|---|---|---|---|---|
| mild curry + naan flav | 19.1 | 16.7 | 20.6 | 20.2 | -2.4 | +1.5 | +1.1 |
| mild curry + rice pilau | 18.5 | 15.1 | 20.2 | 19.4 | -3.4 | +1.6 | +0.9 |
| mild curry + veg side | 16.7 | 18.2 | 16.7 | 16.9 | +1.4 | -0.1 | +0.1 |
| mild curry + papadum | 14.3 | 11.7 | 14.3 | 14.1 | -2.6 | -0.0 | -0.2 |
| mild curry + naan plain | 13.9 | 12.0 | 15.4 | 15.2 | -1.9 | +1.6 | +1.3 |
| medium curry + rice pilau | 13.7 | 10.8 | 13.0 | 12.6 | -2.9 | -0.7 | -1.2 |
| medium curry + naan flav | 13.6 | 11.9 | 13.4 | 13.2 | -1.7 | -0.2 | -0.4 |
| medium curry + veg side | 13.1 | 13.0 | 13.3 | 13.3 | -0.1 | +0.2 | +0.2 |
| mild curry + rice special | 12.9 | 11.6 | 13.1 | 12.7 | -1.3 | +0.2 | -0.2 |
| hot curry + naan flav | 11.6 | 10.0 | 11.2 | 11.1 | -1.6 | -0.4 | -0.5 |
| hot curry + rice pilau | 11.0 | 9.1 | 11.7 | 11.3 | -2.0 | +0.6 | +0.2 |
| mild curry + starter fried | 10.8 | 11.7 | 12.5 | 12.5 | +0.9 | +1.7 | +1.8 |
| hot curry + veg side | 10.8 | 10.9 | 11.1 | 11.1 | +0.2 | +0.3 | +0.3 |
| hot curry + papadum | 10.2 | 7.0 | 8.8 | 8.6 | -3.2 | -1.4 | -1.5 |
| medium curry + rice special | 10.1 | 8.3 | 9.5 | 9.3 | -1.8 | -0.6 | -0.9 |

Account lead, restaurant 1: MILD_CURRY + RICE_PILAU + NAAN_FLAV (list £14.85), entree rate 39.5%: observed 23.3% already buy all, £52.00 given away per 100 entree orders; M0 15.7% already buy all, £35.07 given away per 100 entree orders; M1 28.1% already buy all, £62.62 given away per 100 entree orders; M3 26.8% already buy all, £59.60 given away per 100 entree orders

M3 calibration slopes r1: {'MILD_CURRY': 0.911, 'MEDIUM_CURRY': 0.892, 'HOT_CURRY': 0.306, 'HOUSE_SPECIAL': 0.148, 'BIRYANI': 0.312, 'TANDOORI_MAIN': 0.559, 'VEG_SIDE': 0.95, 'RICE_PILAU': 0.966, 'RICE_PLAIN': 0.573, 'RICE_SPECIAL': 0.788, 'NAAN_PLAIN': 0.898, 'NAAN_FLAV': 0.936, 'BREAD_OTHER': 0.331, 'PAPADUM': 0.914, 'FRIES': 0.665, 'STARTER_FRIED': 0.845, 'STARTER_GRILL': 0.625, 'STARTER_OTHER': 0.736, 'DRINK': 0.748, 'CONDIMENT': 0.899, 'SAUCE': 0.773}

GHK summed LL by scramble r1: shrunk [-45290.45, -45284.83, -45293.29, -45291.11], raw [-636802.51, -45377.64, -52027.47, -410028.95]

### Restaurant 2 (all orders): M3 top complements and substitutes

| complement pair | ρ | substitute pair | ρ |
|---|---|---|---|
| papadum + condiment | +0.79 | naan_plain + naan_flav | -0.42 |
| mild_curry + rice_pilau | +0.34 | rice_pilau + rice_special | -0.32 |
| rice_special + naan_flav | +0.30 | biryani + rice_special | -0.31 |
| mild_curry + naan_flav | +0.26 | rice_pilau + rice_plain | -0.30 |
| rice_plain + naan_plain | +0.22 | biryani + rice_pilau | -0.25 |
| rice_pilau + naan_plain | +0.21 | naan_flav + bread_other | -0.23 |
| fries + starter_grill | +0.21 | biryani + veg_side | -0.23 |
| tandoori_main + sauce | +0.21 | hot_curry + biryani | -0.22 |
| papadum + fries | +0.20 | medium_curry + house_special | -0.21 |
| medium_curry + rice_special | +0.20 | rice_special + naan_plain | -0.20 |

### Restaurant 2 (all orders): top 15 bundles by observed holdout full-bundle rate (%)

| bundle | observed | M0 | M1 | M3 | M0 err | M1 err | M3 err |
|---|---|---|---|---|---|---|---|
| mild curry + rice pilau | 21.7 | 18.0 | 23.9 | 23.2 | -3.7 | +2.2 | +1.5 |
| mild curry + naan flav | 19.8 | 17.7 | 22.2 | 21.7 | -2.1 | +2.3 | +1.9 |
| mild curry + veg side | 19.6 | 21.4 | 20.8 | 21.0 | +1.8 | +1.2 | +1.4 |
| hot curry + veg side | 15.9 | 15.3 | 16.0 | 16.0 | -0.6 | +0.1 | +0.1 |
| medium curry + rice pilau | 15.9 | 12.5 | 13.8 | 13.4 | -3.4 | -2.1 | -2.5 |
| mild curry + rice special | 15.6 | 13.2 | 15.4 | 14.9 | -2.4 | -0.2 | -0.7 |
| medium curry + veg side | 15.2 | 14.9 | 14.7 | 14.8 | -0.3 | -0.5 | -0.4 |
| hot curry + rice pilau | 15.1 | 12.9 | 15.0 | 14.6 | -2.2 | -0.1 | -0.4 |
| mild curry + papadum | 14.9 | 14.4 | 15.7 | 15.6 | -0.5 | +0.9 | +0.7 |
| mild curry + naan plain | 14.9 | 14.7 | 17.6 | 17.3 | -0.2 | +2.7 | +2.5 |
| mild curry + starter fried | 14.3 | 14.4 | 16.0 | 15.9 | +0.0 | +1.7 | +1.6 |
| hot curry + naan flav | 13.4 | 12.7 | 14.4 | 14.3 | -0.7 | +1.1 | +0.9 |
| tandoori main + veg side | 13.3 | 11.5 | 13.5 | 13.4 | -1.8 | +0.2 | +0.1 |
| medium curry + naan flav | 12.7 | 12.3 | 13.6 | 13.4 | -0.4 | +0.9 | +0.7 |
| medium curry + papadum | 11.8 | 10.1 | 11.3 | 11.1 | -1.7 | -0.5 | -0.6 |

Account lead, restaurant 2: MILD_CURRY + RICE_PILAU + NAAN_FLAV (list £14.85), entree rate 39.0%: observed 29.5% already buy all, £65.62 given away per 100 entree orders; M0 19.5% already buy all, £43.51 given away per 100 entree orders; M1 33.3% already buy all, £74.08 given away per 100 entree orders; M3 31.9% already buy all, £71.01 given away per 100 entree orders

M3 calibration slopes r2: {'MILD_CURRY': 0.925, 'MEDIUM_CURRY': 0.902, 'HOT_CURRY': 0.957, 'HOUSE_SPECIAL': 0.793, 'BIRYANI': 0.347, 'TANDOORI_MAIN': 0.829, 'VEG_SIDE': 0.968, 'RICE_PILAU': 0.923, 'RICE_PLAIN': 0.875, 'RICE_SPECIAL': 0.94, 'NAAN_PLAIN': 0.839, 'NAAN_FLAV': 0.92, 'BREAD_OTHER': 0.933, 'PAPADUM': 0.953, 'FRIES': 0.671, 'STARTER_FRIED': 0.794, 'STARTER_GRILL': 0.781, 'STARTER_OTHER': 0.67, 'DRINK': 0.821, 'CONDIMENT': 0.918, 'SAUCE': 0.585}

GHK summed LL by scramble r2: shrunk [-57874.06, -57871.53, -57872.94, -57872.37], raw [-200015.44, -376061.57, -103241.85, -93165.54]
