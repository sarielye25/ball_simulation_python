# First-stage training record

This record preserves the first-stage experiment design and main findings. It was compiled from `../../training log&result/training_log_first.md`, `../../training log&result/training_result_first.md`, experiment plans and runners, and the available run manifests and evaluations in `../data/` before cleanup. Historical source paths below identify where each experiment ran; the non-E4 experiment folders may no longer be present.

## Task and comparison rules

The model predicts one-dimensional cuboid displacement and final velocity from initial velocity, applied force, push duration, and observation time. The fixed dataset has 8,000 train, 1,000 validation, and 1,000 reserved test rows. Training fits input and target normalization on the train split. Adam uses initial learning rate `0.001`, batch size 128, weight decay 0, shuffled batches, mean standardized MSE loss, and CPU float32. Validation and train splits are evaluated every 100 updates, with extra early evaluations. The reserved test split was not used for the reported comparisons.

Controlled comparisons use paired seeds `42`, `1006`, `10086`, `104792`, and `230786`, the same frozen labels, and metrics at the same optimizer update. The main metric is standardized MSE; lower is better. `overall`, `moving_at_observation`, `always_resting`, and `moved_then_stopped` are reported separately. `breakaway` is an overlapping diagnostic subset. A fixed-update measurement can differ from the run's selected best-validation checkpoint. Train and validation MSE must not be interchanged.

## Experiment definitions and code changes

| Experiment | Architecture | Main change | Budget and comparison |
| --- | --- | --- | --- |
| Initial protocol | 4 → 32 → 32 → 2, ReLU | Original training and stopping policy. Scheduler reduction factor `0.5`, minimum LR `0.00001`, improvement threshold `0.001`, reduction patience 200 updates. | The surviving run reached 2,000 updates and ended with `NO_PROGRESS`; its selected checkpoint was update 1,900. The early failure discussed in the log should be treated as a diagnostic observation, not the final termination record of this run. |
| E1-A | Same | Scheduler reduction patience 500 updates, equivalent to four `ReduceLROnPlateau` validation checks. Threshold `0.001`. Fixed-budget stopping. | Five runs to 2,000 updates. Source: `first_training_experiments_lr/scripts_e1-a/`; data group: `lr_e1_a`. |
| E1-B | Same | Reduction patience 200 updates; scheduler absolute improvement threshold `0.00001`. Fixed-budget stopping. | Five runs to 2,000 updates. Source: `first_training_experiments_lr/scripts_e1-b/`; data group: `lr_e1_b`. |
| E2 | Same | Original 200-update scheduler patience and `0.001` threshold, with longer fixed budgets. | Separate fresh runs for the 4,000, 6,000, 8,000, and 10,000 update comparisons described in the result report. The current workspace does not contain those detailed E2 data groups. The `scripts - 10000` configuration is the surviving 10,000-update baseline code. |
| E3 | Same | E2's 10,000-update runner with scheduler patience overridden to 500 updates; threshold stays `0.001`, factor `0.5`, minimum LR `0.00001`. | Five continuous 10,000-update runs compared with matched E2 10,000-update runs. A separate five-seed extension ran continuously to 20,000 updates. Source: `first_training_experiments_update&lr/run_e3.py` and `run_e3_extended.py`; data group: `e3`. |
| E4 | 4 → 32 → 32 → 32 → 2, ReLU | Adds one 32-unit hidden layer to E3. Same 500-update scheduler patience and other training settings. | Five continuous 10,000-update runs. Source: `training_E4/run_e4.py` and `training_E4/scripts - e4/`; data group: `e4`. **Chosen architecture and retained baseline for stage two.** |
| E5 | 4 → 32 → 32 → 32 → 32 → 2, ReLU | Adds one further 32-unit hidden layer to E4; keeps the schedule and fixed budget. | Five continuous 10,000-update runs. Source: `first_training_experiments_additional_hidden_layer2/run_e5.py` and `scripts - e5/`; data group: `e5`. |

Across the copied script folders, `checkpoints.py`, `metrics.py`, `physics_formula.py`, `report.py`, and `training_records.py` are byte-identical. The experiment-specific behavior lies chiefly in the runners, `config.py`, `neural_network.py`, `training_loop.py`, and, for some earlier conditions, `stopping.py` and `checks.py`. This record describes the experimental changes; it is not a source-level replacement for every historical implementation detail.

## Results retained in text

The complete published tables, including the five motion scopes at each fixed update and E5's per-seed learning-rate changes, remain in `../../training log&result/training_result_first.md`. The figures below give the main comparisons without relying on the detailed run folders.

### Early scheduler and update-budget experiments

Five-seed mean **validation** overall standardized MSE, at each condition's stated update:

| Condition | Update | MSE |
| --- | ---: | ---: |
| E1-A, 500-update patience | 2,000 | 0.006330313 |
| E1-B, `0.00001` threshold | 2,000 | 0.006330313 |
| E2, original schedule | 4,000 | 0.005798509 |
| E2, original schedule | 6,000 | 0.005716659 |
| E2, original schedule | 8,000 | 0.005623416 |
| E2, original schedule | 10,000 | 0.005535955 |

E1-A and E1-B are tied at update 2,000 to the precision reported. In the paired continuous 10,000-update E2/E3 comparison, validation overall MSE at 10,000 was `0.005535955` for E2 and `0.003509172` for E3. E3's five-seed mean training overall MSE reached `0.003459172` at 16,000 and `0.003396477` at 20,000 updates. Those extension numbers are **training**, not validation, values.

### Architecture comparison at 10,000 updates

These **validation** overall standardized MSE values were calculated from the five available per-seed `evaluations.csv` files for E3, E4, and E5, filtering `update=10000`, `split=validation`, `predictor=model`, and `scope=overall`:

| Architecture | Five-seed mean | Smallest seed value | Largest seed value |
| --- | ---: | ---: | ---: |
| E3, two hidden layers | 0.003509172 | 0.002998294 | 0.004687236 |
| E4, three hidden layers | 0.002503191 | 0.002240015 | 0.002788856 |
| E5, four hidden layers | 0.002399999 | 0.001910755 | 0.003255451 |

E4 improves validation overall MSE over E3 at this fixed update. E5's mean is slightly lower than E4's, but its seed range is wider, and the choice for stage two is **E4 because stage two will use E4's three-hidden-layer architecture**. The existing report's E3/E4/E5 comparison tables at 10,000 (`0.003557779`, `0.002439747`, `0.002280410`) are explicitly **training** MSE and answer a different question. E5's report also records substantial seed variation and shows that later LR reductions often precede very small or negative improvements at the next validation check; that observation alone does not establish a causal effect.

## Stage-two handoff and remaining limits

Stage two will keep E4's architecture and investigate a more precise learning-rate schedule and specific motion regimes. Retaining all five E4 runs supports paired-seed comparisons, per-regime error analysis, learning-rate traces, row-level failure inspection, and reuse of selected model checkpoints. The copied `../second_training_protocol/run_training.py` still contains E4/E3 paths and comparisons; it is scaffolding rather than an implemented stage-two protocol. The retained `training_E4/run_e4.py` runs independently of deleted E3 data; it does not recreate the historical E3/E4 comparison.

If detailed non-E4 data is removed, historical per-seed trajectories, checkpoints, predictions, and unreported metrics cannot be recovered from this text. In particular, the current report does not preserve every per-regime validation figure or every seed's LR trajectory for E1 through E3. Any future comparison against those historical conditions must use the published aggregates or rerun the experiment. `data/` is ignored by Git in this workspace, so Git should not be assumed to provide recovery for deleted runs.
