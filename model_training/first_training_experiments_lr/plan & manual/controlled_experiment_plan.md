# Learning-rate experiments

## Comparison Standard & Requirements

- Compare checkpoints at the same optimizer update. 
- Each group should be ran 5 times, and calculate the average mse for all groups to compare.
- Initial LR `0.001`, reduction factor `0.5`, minimum LR `0.00001`, and evaluation every 100 updates.
- Progress-patience is disabled for these fixed-budget comparisons.

## Experiment 1 — Learning-rate schedule

All conditions run to 2000 updates. Change one scheduler setting per condition.

| Label | Condition |
| --- | --- |
| E1-A — slower reductions | Scheduler patience: 500 updates. |
| E1-B — smaller improvements count | Scheduler absolute improvement threshold: `0.00001`. |

## Experiment 2 — Update budget

Use the selected Experiment 1 schedule. Change only the update budget: inspect checkpoints at 2000, 4000, 6000, 8000, and 10000 updates in one continuous run. 

Focus on MSE and overfitting.

## Code locations

| Change | Location |
| --- | --- |
| LR, scheduler patience, threshold, update cap | `../scripts/config.py`: `learning_rate`, `lr_reduction_patience_updates`, `min_delta`, `max_updates`. |
| Scheduler setup | `../scripts/training_loop.py`: `ReduceLROnPlateau(...)`. |
| Fixed-budget stopping | `../scripts/training_loop.py`: `check_stop(...)` call and terminal decision; `../scripts/stopping.py`: stop policy. Add an explicit fixed-budget mode while retaining numerical checks. |
| Threshold separation | Scheduler and early stopping currently share `min_delta`; give them separate settings before changing one independently. |
| Output directory | `../scripts/config.py`: `training_data_dir`; use the LR experiment's own group directory. |
