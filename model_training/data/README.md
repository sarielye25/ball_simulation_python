# Training data files

## Directory layout

| Path | Contents |
| --- | --- |
| `<group_id>/group.json` | Protocol version, group ID, display name, description, training-source identifier. |
| `<group_id>/runs/<run_id>/run.json` | Published run manifest: identity, status, splits, columns, units, tolerances, checkpoints, file paths, byte counts, SHA-256 hashes. |
| `<group_id>/runs/<run_id>/datasets/<split_id>.csv` | Frozen samples: `row_id`, `motion_group`, input columns, target columns. Values use physical units. |
| `<group_id>/runs/<run_id>/evaluations.csv` | Metrics by `update`, `split`, `predictor`, and `scope`. Train and validation rows share this file. |
| `<group_id>/runs/<run_id>/baselines.csv` | Training-set reference metrics for `zero` and `constant_velocity` predictors. |
| `<group_id>/runs/<run_id>/evaluation_events.csv` | Evaluation timing, learning rate, stopping decision, and progress state. |
| `<group_id>/runs/<run_id>/batches.csv` | Per-update batch size, learning rate, and loss before the update. |
| `<group_id>/runs/<run_id>/checkpoints/index.json` | Saved checkpoint list and `last`, `best_mse`, `best_pass_rate`, `selected` pointers. |
| `<group_id>/runs/<run_id>/checkpoints/update_*.pt` | PyTorch model weights, optimizer/scheduler state, normalization, and resume state. |
| `<group_id>/runs/<run_id>/predictions/<checkpoint_id>/<split_id>.csv` | `row_id` and predicted target values; join to dataset rows by `(split_id, row_id)`. |
| `<group_id>/runs/<run_id>/termination.json` | Final status, stop reason, update, selected checkpoint, and elapsed time. |
| `<group_id>/runs/<run_id>/training_run.json` | Original training-format manifest retained by the sample-run importer. |

## Reading metrics

| Column | Contents |
| --- | --- |
| `update` | Optimizer step; graph X axis. |
| `split` | Dataset ID, such as `train` or `validation`. |
| `predictor` | `model`, `zero`, or `constant_velocity`. |
| `scope` | `overall`, `breakaway`, or a motion group. |
| `count` | Number of samples in this metric row. |
| `standardized_mse` | Mean squared error in normalized target space. |
| `displacement_*` | Displacement error metrics; physical errors use metres. |
| `velocity_*` | Velocity error metrics; physical errors use m/s. |
| `pass_rate_*` | Fraction of rows passing the named tolerance, from 0 to 1. |

## Current sample runs

| Run | Format | Contents |
| --- | --- | --- |
| `sample_training/runs/20260928_212044_053781` | Original training v1 | 10-update diagnostic; raw training records and checkpoints. |
| `sample_training/runs/20260928_214346_696295` | Original training v1 | 1-update diagnostic; raw training records and checkpoints. |
| `sample_training/runs/imported_20260928_214346_696295` | Shared contract v1 | Copy of the latest diagnostic with per-checkpoint predictions; readable by Qt. |

New runs from `sample_training` and `first_training_protocol` write shared-contract `run.json` files directly. Older runs with `format_version: 1` remain in the legacy format.

In the shared format, `committed_bytes` and `sha256` identify the published prefix of a growing CSV. Bytes after that prefix are not yet published. Immutable files use `bytes` and `sha256` for the complete file.
