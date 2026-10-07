# E4 learning-rate comparison

`run_training.py` compares the E4 network with the same five seeds as the original E4 run (42, 1006, 10086, 104792, 230786). It writes each run to `../data/e4_2`, `../data/e4_3`, or `../data/e4_4`. E4-1 remains in `../data/e4`.

| Group | Learning rate | Scheduler threshold |
| --- | --- | --- |
| E4-1 | Original E4 scheduler | Absolute 0.001 |
| E4-2 | Original E4 scheduler | Absolute 0.001 |
| E4-3 | Constant 0.001 | None |
| E4-4 | ReduceLROnPlateau | Relative 0.001 |

The plateau groups retain E4's 500-update LR patience, factor 0.5, minimum LR 0.00001, and zero cooldown. Early stopping and all other training settings remain shared. E4-2 is a reproducibility control against E4-1.

Run with the project virtual environment:

```powershell
& '..\..\.venv\Scripts\python.exe' run_training.py
& '..\..\.venv\Scripts\python.exe' evaluate_lr_comparison.py
```

Use `--group e4_4` and/or `--seed 42` to run a subset. `--max-updates 1` creates separate diagnostic runs. The evaluator writes `lr_comparison_validation.md` from completed runs and saved validation predictions; it can be rerun as more seeds finish.
