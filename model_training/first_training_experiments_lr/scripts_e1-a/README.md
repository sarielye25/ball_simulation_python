# Experiment 1 comparison

E1-A and E1-B both run to 2000 optimizer updates with numerical checks enabled.
E1-A changes scheduler patience to 500 updates; E1-B changes only the scheduler
absolute improvement threshold to `0.00001`. They share the same baseline
learning rate, reduction factor, minimum learning rate, evaluation interval,
data, and five seeds.

From this directory, run:

```powershell
..\..\..\.venv\Scripts\python.exe compare_e1.py
```

The runner creates ten separate runs under `e1_comparison`, reads each run's
validation standardized MSE at update 2000, and prints each condition's mean.
It also saves individual results to `validation_mse_at_2000.csv`. Re-running
with the same output directory refuses to overwrite prior runs; use `--output`
for a new directory. A single diagnostic run can use `training_loop.py
--max-updates 500 --seed 42` from either scripts directory.
