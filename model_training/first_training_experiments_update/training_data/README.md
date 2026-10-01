# Update-budget experiment

The four script folders use the original first-training model, labels, Adam
settings, batch size, and learning-rate scheduler. Only the maximum number of
optimizer updates changes: 4,000, 6,000, 8,000, or 10,000. Progress-patience
and target-success stopping are disabled so each run reaches its budget unless
training fails numerically. The scheduler's separate 200-update patience is
unchanged.

Each script reads this experiment's `labels/train.csv` and
`labels/validation.csv`; the reserved test split is unused. Results go to
`model_training/data/update_<budget>/runs/<timestamp>/`, with a separate Qt
viewer group for each budget. Each launch starts a new model. Use the same seed
in each budget for a matched comparison.

From the project root, for example:

```powershell
& .\.venv\Scripts\python.exe 'model_training/first_training_experiments_update/scripts-4000/training_loop.py' --seed 42
& .\.venv\Scripts\python.exe 'model_training/first_training_experiments_update/scripts - 6000/training_loop.py' --seed 42
& .\.venv\Scripts\python.exe 'model_training/first_training_experiments_update/scripts - 8000/training_loop.py' --seed 42
& .\.venv\Scripts\python.exe 'model_training/first_training_experiments_update/scripts - 10000/training_loop.py' --seed 42
```

For a diagnostic run, add `--max-updates 1`. For the experiment, compare the
`evaluations.csv` row with `split=validation`, `predictor=model`,
`scope=overall`, and the specified `update`. The selected checkpoint may be
earlier than the final update.

## Run artifact reference

Each training run creates a separate timestamped folder. Training writes the
results; Qt reads them. Output format version: **1**.

## Folder structure

```text
training_data/<timestamp>/
    run.json                    # Settings, normalization, units and dataset hashes
    datasets/
        train.csv               # Frozen training rows with IDs and motion groups
        validation.csv          # Frozen validation rows
    batches.csv                 # Loss and learning rate for each update
    evaluations.csv             # Train/validation metrics for graphs
    evaluation_events.csv       # Evaluation timing and stopping decisions
    baselines.csv               # Zero and constant-velocity reference metrics
    termination.json            # Final status, reason and selected checkpoint
    checkpoints/
        index.json              # Available, best, last and selected checkpoints
        update_XXXXXX.pt         # Model weights, normalization and training state
    figures/                    # Exported graphs
    exports/                    # Exported Qt tables
```

Folders are created automatically. Existing runs are not overwritten.
The reserved test dataset is not copied or evaluated.

## Start training

Run from one of the four script folders, using the project virtual environment:

```powershell
..\..\..\.venv\Scripts\python.exe training_loop.py --seed 42
```

Add `--max-updates 500` for a short diagnostic run. Each command starts fresh.

## Qt reading rules

- Read settings and paths from `run.json`, not the current `config.py`.
- Use `evaluations.csv` for graphs, with optimizer update on the X axis.
- Load the checkpoint marked `selected` in `checkpoints/index.json`.
  If absent, use `last` and indicate that no final model was selected.
- For failure tables, verify dataset snapshot hashes, then predict the frozen
  rows using the checkpoint’s saved normalization. Never refit normalization.
- Pass rates use 0–1. Both errors must be strictly below their tolerances.
  Empty metric cells mean undefined, not zero.
- Missing `termination.json` means incomplete or unknown status.
  For the first viewer, open runs after training stops.

## Scripts

| Script | Responsibility |
| --- | --- |
| `training_records.py` | Create folders and write datasets, JSON and CSV records |
| `training_loop.py` | Call writers during training and record final status |
| `checkpoints.py` | Save and load models; maintain the checkpoint index |
| `report.py` | Generate static PNG/SVG graphs |
