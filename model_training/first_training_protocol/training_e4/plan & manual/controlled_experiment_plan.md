# Controlled training experiments

## Objective

Measure the effect of learning-rate control and update budget on prediction accuracy, motion-group performance, and generalization for the fixed 4–32–32–2 ReLU model.

## Controls and comparison standard

| Item | Rule |
| --- | --- |
| Data | Identical frozen train and validation rows and group labels; reserved test rows excluded from selection. |
| Model and optimization | Identical architecture, Adam settings, batch size, initialization seed, and shuffled batch sequence. |
| Normalization | Fit once on the training split; use its target scales for every group, split, and condition. Verify scales and data hashes match. |
| Evaluation | Full train and validation splits at update 0 and every 100 updates; compare the same update numbers. |
| Primary metric | Overall validation standardized MSE at the specified update. Lower is better. |
| Group metrics | Standardized MSE for each mutually exclusive motion group on train and validation, with sample counts. Lower is better. |
| Supporting metrics | Overall and group displacement/velocity MAE in physical units, intermediate pass rates, and output-specific standardized MSE. |
| Generalization | Compare train and validation trajectories and their gap. Persistent validation deterioration while training improves indicates overfitting. |
| Selection | Report the fixed-update result and the best validation-MSE checkpoint within the budget separately. Record the selected update. |
| Reproducibility | Record actual scheduler settings, learning-rate events, stop reason, model specification, seed, data hashes, and normalization. Repeat promising comparisons with additional paired seeds. |

Standardized MSE is the mean of the two squared, training-scale-normalized target errors. Group MSE uses this same formula restricted to group rows. The `breakaway` scope overlaps motion groups and is reported separately. A condition is favorable when overall validation MSE improves without an unacceptable group regression; physical errors and pass rates determine practical value. Report absolute values and percentage change against the matched baseline.

## Experiment 1 — Learning-rate control at 2000 updates

| Condition | Changed variable | Scheduler reduction patience | Scheduler absolute threshold | Update cap |
| --- | --- | ---: | ---: | ---: |
| E1-A | Baseline | 200 updates | 0.001 | 2000 |
| E1-B | Reduction frequency | 500 updates | 0.001 | 2000 |
| E1-C | Improvement sensitivity | 200 updates | 0.00001 | 2000 |

Maintain the initial learning rate at 0.001, reduction factor at 0.5, minimum rate at 0.00001, and evaluation interval at 100 updates. Disable progress-patience and target-success termination for all three conditions so every condition reaches update 2000; retain numerical-failure termination. The existing first run is historical context; use it as a matched baseline only after verifying every control, including termination policy and batch trajectory. Compare at 2000 and inspect the complete learning-rate and validation curves.

## Experiment 2 — Update budget

Choose the learning-rate condition from Experiment 1 using validation results. Run it continuously to 10000 updates; report updates 2000, 4000, 6000, 8000, and 10000. Keep schedule settings constant across the trajectory. Continue past intermediate checkpoints with full optimizer, scheduler, sampler, and random states. Compare the original scheduler at the same updates as a duration control if the changed schedule remains promising. Retain the same evaluation standard and assess overfitting from train and validation trends.

## Implementation locations

| Concern | Location |
| --- | --- |
| Initial rate, scheduler patience, threshold, cap, evaluation interval | `../config.py`: `learning_rate`, `lr_reduction_patience_updates`, `min_delta`, `max_updates`, `eval_interval`. |
| Scheduler construction | `../training_loop.py`: `ReduceLROnPlateau(...)` in `run_training`. |
| Progress and target stopping | `../training_loop.py`: `check_stop(...)` call and terminal decision; `../stopping.py`: `check_stop` policy. Add an explicit fixed-budget mode; do not remove numerical checks. |
| Independent thresholds | Separate scheduler threshold and stopping `min_delta` before varying either in mixed stopping experiments. Currently both use `config.min_delta`. |
| Full-split metrics and group definitions | `../metrics.py`: `summarize_metrics`, `group_metrics`; `../data.py`: `assign_groups`. |
| Records and publication | `../training_records.py`, `../../shared_publication.py`. |
| Experiment output identity | `../config.py`: `training_data_dir` currently points to `first_training_protocol/runs`; change it to `first_training_experiments/runs` before running experiment conditions. |

## Reporting table

Each condition and specified update: run ID, seed, data identity, LR settings, actual LR, stop reason, selected checkpoint, train/validation overall MSE, train/validation MSE and count per motion group, train/validation physical MAE, intermediate pass rates, and change from the matched baseline.
