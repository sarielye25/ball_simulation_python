# Training data

Each training run creates a separate timestamped folder. Training writes the
results; Qt reads them. Output format version: **1**.

## Folder structure

```text
model_training/data/sample_training/runs/<timestamp>/
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

The default run folder is `model_training/data/sample_training/runs/`. Folders are created automatically. Existing runs are not overwritten.
The reserved test dataset is not copied or evaluated.

## Start training

Run from the protocol folder:

```powershell
..\..\.venv\Scripts\python.exe training_loop.py
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

The Qt interface is not implemented yet.

## Scripts

| Script | Responsibility |
| --- | --- |
| `training_records.py` | Create folders and write datasets, JSON and CSV records |
| `training_loop.py` | Call writers during training and record final status |
| `checkpoints.py` | Save and load models; maintain the checkpoint index |
| `report.py` | Generate static PNG/SVG graphs |
