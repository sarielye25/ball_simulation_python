# 26.9.30 controlled experiment

## Five-seed validation MSE comparison

Seeds: `42`, `1006`, `10086`, `104792`, `230786`. Each value below is the arithmetic mean of five **validation standardized MSE** values at the stated optimizer update, using the `model` predictor. Lower is better.

| Experiment | Group | Update | General (overall) | Always moving | Static | Moving then stop | Breakaway |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 | E1-A | 2,000 | 0.006330313 | 0.007585044 | 0.005799031 | 0.004850208 | 0.005131671 |
| E1 | E1-B | 2,000 | 0.006330313 | 0.007585044 | 0.005799031 | 0.004850208 | 0.005131671 |
| E2 | 4,000 updates | 4,000 | 0.005798509 | 0.006920886 | 0.005512819 | 0.004445440 | 0.004912563 |
| E2 | 6,000 updates | 6,000 | 0.005716659 | 0.006828689 | 0.005429444 | 0.004376702 | 0.004824230 |
| E2 | 8,000 updates | 8,000 | 0.005623416 | 0.006725250 | 0.005325140 | 0.004297846 | 0.004736047 |
| E2 | 10,000 updates | 10,000 | 0.005535955 | 0.006629777 | 0.005221411 | 0.004222854 | 0.004643389 |

The table labels map to the saved metric scopes as follows: general = `overall`; always moving = `moving_at_observation`; static = `always_resting`; moving then stop = `moved_then_stopped`; breakaway = `breakaway`. Breakaway overlaps the motion groups, so it is a separate diagnostic subset.

All 30 runs (five per row) completed and loaded through the Qt viewer's data reader. E1-A and E1-B have identical means at update 2,000. E2 used the original first-training learning-rate schedule and changed the fixed maximum-update budget; progress-patience and target-success stopping were disabled. Every E2 run stopped with `MAX_UPDATES`. Each budget and seed was trained from a fresh model, with the same seed set and frozen train/validation labels. These are validation results; the reserved test split was not evaluated.

The reported MSE is from the checkpoint **at the stated update**, not necessarily the selected best-validation checkpoint. Run-level values are in `model_training/data/<group>/runs/<run_id>/evaluations.csv`, with E1 groups `lr_e1_a`, `lr_e1_b` and E2 groups `update_4000`, `update_6000`, `update_8000`, `update_10000`. Filter to `split=validation`, `predictor=model`, the stated `update`, and each scope above.

# 26.10.1 controlled experiment — E3 learning-rate schedule

## Question and controls

Does E1-A's longer learning-rate reduction patience improve training when the model can continue beyond 2,000 updates? E3 changes the scheduler reduction patience from 200 to 500 optimizer updates (`ReduceLROnPlateau` patience from 1 to 4 validation checks). Its absolute improvement threshold remains `0.001`; the reduction factor remains `0.5`. All other model, optimizer, data, evaluation, and fixed-budget settings match E2's 10,000-update runs.

Five paired seeds were used: `42`, `1006`, `10086`, `104792`, and `230786`. E3 ran each seed continuously to 10,000 updates. E2 values below come from the matching seed's **10,000-update continuous run**, evaluated at the same checkpoints. Train and validation dataset hashes match between E2 and E3 for every seed. The reserved test split was not evaluated.

## Five-seed means at fixed updates

### Comparison between E2 and E3

| Update | E2 MSE | E3 MSE | E2 displacement MAE (m) | E3 displacement MAE (m) | E2 velocity MAE (m/s) | E3 velocity MAE (m/s) | E2 pass rate | E3 pass rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 | 0.006554601 | 0.006330313 | 0.470678 | 0.466784 | 0.296236 | 0.289362 | 0.04% | 0.06% |
| 4,000 | 0.005798509 | 0.003802858 | 0.443388 | 0.367978 | 0.277380 | 0.216786 | 0.06% | 0.12% |
| 6,000 | 0.005716659 | 0.003585901 | 0.440226 | 0.359455 | 0.275178 | 0.208580 | 0.04% | 0.24% |
| 8,000 | 0.005623416 | 0.003547406 | 0.436940 | 0.357797 | 0.272676 | 0.207574 | 0.04% | 0.14% |
| 10,000 | 0.005535955 | 0.003509172 | 0.433938 | 0.356117 | 0.270008 | 0.205985 | 0.06% | 0.16% |

### Group added: 16,000 and 20,000 updates in E3 group

Five-seed mean training standardized MSE at each fixed update:

| Update | Overall | Moving at observation | Always resting | Moved then stopped | Breakaway |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16,000 | 0.003459172 | 0.004327619 | 0.003177936 | 0.002429816 | 0.002824036 |
| 20,000 | 0.003396477 | 0.004254905 | 0.003119279 | 0.002378861 | 0.002765702 |


### Results:
The results favor the new learning-rate schedule, and longer training improves the model further. However, the percentage improvement decreases as the number of updates increases.

## E3 and E4 comparison:

Five-seed mean training standardized MSE at each fixed update:

| Update | Overall | Moving at observation | Always resting | Moved then stopped | Breakaway |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 (E3) | 0.006328840 | 0.007828243 | 0.005968580 | 0.004530456 | 0.005783830 |
| 2,000 (E4) | 0.004072417 | 0.005220684 | 0.003844855 | 0.002687019 | 0.003496568 |
| 4,000 (E3) | 0.003859414 | 0.004810276 | 0.003551841 | 0.002732313 | 0.003225394 |
| 4,000 (E4) | 0.002598792 | 0.003353920 | 0.002397706 | 0.001696412 | 0.002051277 |
| 6,000 (E3) | 0.003629392 | 0.004528441 | 0.003354290 | 0.002561055 | 0.003005246 |
| 6,000 (E4) | 0.002497223 | 0.003224191 | 0.002306822 | 0.001627956 | 0.001961896 |
| 8,000 (E3) | 0.003592539 | 0.004488024 | 0.003310143 | 0.002529851 | 0.002965399 |
| 8,000 (E4) | 0.002467008 | 0.003187870 | 0.002265454 | 0.001607195 | 0.001926218 |
| 10,000 (E3) | 0.003557779 | 0.004446915 | 0.003269848 | 0.002503899 | 0.002925159 |
| 10,000 (E4) | 0.002439747 | 0.003155605 | 0.002238108 | 0.001586154 | 0.001896633 |

Results: E4 generally performs better than E3, which may show the benefit of adding one more hidden layer.
Future direction: add one more hidden layer and compare E3, E4, and E5.

## E5's training
Observations:
1. Training results vary substantially across seeds. At 10,000 updates, seed 42 has an MSE of 0.0033, while seed 1006 has an MSE of 0.0017, nearly half as large.
2. Lowering the learning rate reduces the model's improvement. If the model's MSE is above 0.001, lowering it may not be necessary.

[I will compile a list of identical statistics and discuss it with Astra.]

## E4 and E5 comparison:

Five-seed mean training standardized MSE at each fixed update:

| Update | Overall | Moving at observation | Always resting | Moved then stopped | Breakaway |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 (E4) | 0.004072417 | 0.005220684 | 0.003844855 | 0.002687019 | 0.003496568 |
| 2,000 (E5) | 0.003695507 | 0.004804813 | 0.003784679 | 0.002304919 | 0.003347182 |
| 4,000 (E4) | 0.002598792 | 0.003353920 | 0.002397706 | 0.001696412 | 0.002051277 |
| 4,000 (E5) | 0.002426062 | 0.003152362 | 0.002402708 | 0.001529405 | 0.001999276 |
| 6,000 (E4) | 0.002497223 | 0.003224191 | 0.002306822 | 0.001627956 | 0.001961896 |
| 6,000 (E5) | 0.002338700 | 0.003031182 | 0.002341675 | 0.001479527 | 0.001946722 |
| 8,000 (E4) | 0.002467008 | 0.003187870 | 0.002265454 | 0.001607195 | 0.001926218 |
| 8,000 (E5) | 0.002310379 | 0.002999673 | 0.002287108 | 0.001459592 | 0.001899009 |
| 10,000 (E4) | 0.002439747 | 0.003155605 | 0.002238108 | 0.001586154 | 0.001896633 |
| 10,000 (E5) | 0.002280410 | 0.002960735 | 0.002257689 | 0.001440653 | 0.001870781 |

## E3, E4 and E5 overall MSE comparison:

Five-seed mean training overall standardized MSE at each fixed update:

| Update | E3 MSE | E4 MSE | E5 MSE |
| ---: | ---: | ---: | ---: |
| 2,000 | 0.006328840 | 0.004072417 | 0.003695507 |
| 4,000 | 0.003859414 | 0.002598792 | 0.002426062 |
| 6,000 | 0.003629392 | 0.002497223 | 0.002338700 |
| 8,000 | 0.003592539 | 0.002467008 | 0.002310379 |
| 10,000 | 0.003557779 | 0.002439747 | 0.002280410 |

## E5 seed and learning-rate details

### Training overall standardized MSE by seed

| Seed | 2,000 | 4,000 | 6,000 | 8,000 | 10,000 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.005599059 | 0.003509132 | 0.003414089 | 0.003374119 | 0.003327616 |
| 1006 | 0.002689264 | 0.001849374 | 0.001773540 | 0.001748031 | 0.001722113 |
| 10086 | 0.002892244 | 0.002233673 | 0.002155743 | 0.002128696 | 0.002100424 |
| 104792 | 0.002570141 | 0.001924508 | 0.001865997 | 0.001840469 | 0.001815667 |
| 230786 | 0.004726828 | 0.002613625 | 0.002484130 | 0.002460579 | 0.002436229 |

### Learning-rate changes and the next validation check

| Seed | LR change update | LR before → after | Next validation update | Validation MSE at change | Next validation MSE | Improvement | Improvement % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 2,200 | 0.001 → 0.0005 | 2,300 | 0.004556643 | 0.004136343 | +0.000420300 | +9.22% |
| 42 | 2,800 | 0.0005 → 0.00025 | 2,900 | 0.003838949 | 0.003689981 | +0.000148968 | +3.88% |
| 42 | 3,300 | 0.00025 → 0.000125 | 3,400 | 0.003718139 | 0.003554983 | +0.000163155 | +4.39% |
| 42 | 3,800 | 0.000125 → 6.25e-05 | 3,900 | 0.003456528 | 0.003433622 | +0.000022907 | +0.66% |
| 42 | 4,300 | 6.25e-05 → 3.125e-05 | 4,400 | 0.003396576 | 0.003412670 | -0.000016094 | -0.47% |
| 42 | 4,800 | 3.125e-05 → 1.5625e-05 | 4,900 | 0.003345991 | 0.003363772 | -0.000017781 | -0.53% |
| 42 | 5,300 | 1.5625e-05 → 1e-05 | 5,400 | 0.003351492 | 0.003343240 | +0.000008252 | +0.25% |
| 1006 | 2,100 | 0.001 → 0.0005 | 2,200 | 0.003070085 | 0.002665864 | +0.000404222 | +13.17% |
| 1006 | 2,800 | 0.0005 → 0.00025 | 2,900 | 0.002239502 | 0.002201047 | +0.000038455 | +1.72% |
| 1006 | 3,300 | 0.00025 → 0.000125 | 3,400 | 0.002178106 | 0.002109504 | +0.000068602 | +3.15% |
| 1006 | 3,800 | 0.000125 → 6.25e-05 | 3,900 | 0.002037036 | 0.002019404 | +0.000017632 | +0.87% |
| 1006 | 4,300 | 6.25e-05 → 3.125e-05 | 4,400 | 0.002008473 | 0.001998960 | +0.000009513 | +0.47% |
| 1006 | 4,800 | 3.125e-05 → 1.5625e-05 | 4,900 | 0.001986752 | 0.001980713 | +0.000006039 | +0.30% |
| 1006 | 5,300 | 1.5625e-05 → 1e-05 | 5,400 | 0.001967655 | 0.001971671 | -0.000004016 | -0.20% |
| 10086 | 2,000 | 0.001 → 0.0005 | 2,100 | 0.002928162 | 0.002763780 | +0.000164382 | +5.61% |
| 10086 | 2,600 | 0.0005 → 0.00025 | 2,700 | 0.002603161 | 0.002527846 | +0.000075315 | +2.89% |
| 10086 | 3,100 | 0.00025 → 0.000125 | 3,200 | 0.002494525 | 0.002469228 | +0.000025297 | +1.01% |
| 10086 | 3,600 | 0.000125 → 6.25e-05 | 3,700 | 0.002398070 | 0.002369643 | +0.000028427 | +1.19% |
| 10086 | 4,100 | 6.25e-05 → 3.125e-05 | 4,200 | 0.002349227 | 0.002315861 | +0.000033365 | +1.42% |
| 10086 | 4,600 | 3.125e-05 → 1.5625e-05 | 4,700 | 0.002296711 | 0.002299166 | -0.000002455 | -0.11% |
| 10086 | 5,100 | 1.5625e-05 → 1e-05 | 5,200 | 0.002285965 | 0.002295964 | -0.000009999 | -0.44% |
| 104792 | 2,000 | 0.001 → 0.0005 | 2,100 | 0.002752219 | 0.002721628 | +0.000030591 | +1.11% |
| 104792 | 2,700 | 0.0005 → 0.00025 | 2,800 | 0.002466895 | 0.002208490 | +0.000258405 | +10.47% |
| 104792 | 3,200 | 0.00025 → 0.000125 | 3,300 | 0.002318593 | 0.002135039 | +0.000183554 | +7.92% |
| 104792 | 3,700 | 0.000125 → 6.25e-05 | 3,800 | 0.002096342 | 0.002071864 | +0.000024478 | +1.17% |
| 104792 | 4,200 | 6.25e-05 → 3.125e-05 | 4,300 | 0.002032758 | 0.002033278 | -0.000000521 | -0.03% |
| 104792 | 4,700 | 3.125e-05 → 1.5625e-05 | 4,800 | 0.002034619 | 0.002031584 | +0.000003035 | +0.15% |
| 104792 | 5,200 | 1.5625e-05 → 1e-05 | 5,300 | 0.002014961 | 0.002020967 | -0.000006006 | -0.30% |
| 230786 | 2,600 | 0.001 → 0.0005 | 2,700 | 0.003640817 | 0.003228829 | +0.000411988 | +11.32% |
| 230786 | 3,100 | 0.0005 → 0.00025 | 3,200 | 0.003009367 | 0.002973993 | +0.000035373 | +1.18% |
| 230786 | 3,800 | 0.00025 → 0.000125 | 3,900 | 0.002835853 | 0.002823523 | +0.000012330 | +0.43% |
| 230786 | 4,300 | 0.000125 → 6.25e-05 | 4,400 | 0.002743813 | 0.002765612 | -0.000021799 | -0.79% |
| 230786 | 4,800 | 6.25e-05 → 3.125e-05 | 4,900 | 0.002772768 | 0.002726713 | +0.000046055 | +1.66% |
| 230786 | 5,300 | 3.125e-05 → 1.5625e-05 | 5,400 | 0.002714841 | 0.002698990 | +0.000015851 | +0.58% |
| 230786 | 5,800 | 1.5625e-05 → 1e-05 | 5,900 | 0.002702239 | 0.002687793 | +0.000014446 | +0.53% |

Improvement is validation standardized MSE at the learning-rate change minus MSE at the next validation check; a positive value means lower error.

