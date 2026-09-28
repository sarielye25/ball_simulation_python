"""Save evaluation summaries separately from metric calculation and plotting."""

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile

from config import checkpoint_subdirectory, training_data_dir


TRAINING_DATA_DIR = training_data_dir
METRIC_FIELDS = (
    "count", "standardized_mse",
    "displacement_standardized_mse", "velocity_standardized_mse",
    "displacement_loss_contribution", "velocity_loss_contribution",
    "displacement_mae_m", "displacement_p95_m", "displacement_max_m",
    "velocity_mae_m_s", "velocity_p95_m_s", "velocity_max_m_s",
)
PASS_FIELDS = tuple(f"pass_rate_{name}" for name in ("coarse", "intermediate", "fine"))
FIELDS = ("update", "split", "predictor", "scope") + METRIC_FIELDS + PASS_FIELDS


def create_run_directory(run_directory=None):
    """Create a separate directory for each run without overwriting older runs."""

    run_directory = (Path(run_directory) if run_directory is not None else
                     TRAINING_DATA_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / checkpoint_subdirectory).mkdir()
    (run_directory / "figures").mkdir()
    (run_directory / "datasets").mkdir()
    (run_directory / "exports").mkdir()
    return run_directory


def write_json(path, values):
    """Publish complete JSON documents atomically for independent readers."""
    path = Path(path)
    encoded = json.dumps(values, indent=2, allow_nan=False).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def append_csv(path, row):
    """Append a complete record, enforcing the existing column order."""
    path = Path(path)
    has_header = path.exists() and path.stat().st_size > 0
    if has_header:
        with path.open(newline="", encoding="utf-8") as stream:
            if next(csv.reader(stream)) != list(row):
                raise ValueError(f"Unexpected CSV columns in {path}.")
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not has_header:
            writer.writeheader()
        writer.writerow(row)


def save_datasets(run_directory, splits, input_columns, target_columns):
    """Freeze the physical rows used by this run for portable failure inspection."""
    identity = {}
    fields = ("row_id", "motion_group", *input_columns, *target_columns)
    for name in ("train", "validation"):
        split = splits[name]
        path = Path(run_directory) / "datasets" / f"{name}.csv"
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            for row_id, group, inputs, targets in zip(
                split["row_ids"], split["groups"], split["physical_inputs"],
                split["physical_targets"], strict=True,
            ):
                writer.writerow((row_id, group, *inputs, *targets))
        identity[name] = {
            "sha256": split["sha256"], "row_ids": split["row_ids"],
            "snapshot": path.relative_to(run_directory).as_posix(),
            "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(split["row_ids"]),
        }
    return identity


def _summary_rows(update, split, predictor, result):
    summaries = {"overall": result["overall"], **result.get("groups", {})}
    if "boundary" in result:
        summaries["breakaway"] = result["boundary"]
    for scope, summary in summaries.items():
        row = dict(update=update, split=split, predictor=predictor, scope=scope)
        row.update({field: summary[field] for field in METRIC_FIELDS})
        row.update({f"pass_rate_{name}": summary["pass_rates"][name]
                    for name in ("coarse", "intermediate", "fine")})
        yield row


def _append_rows(path, rows):
    rows = list(rows)
    has_header = path.exists() and path.stat().st_size > 0
    if has_header:
        with path.open(newline="", encoding="utf-8") as stream:
            if next(csv.reader(stream)) != list(FIELDS):
                raise ValueError(f"Unexpected CSV columns in {path}.")
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not has_header:
            writer.writeheader()
        writer.writerows(rows)


def save_evaluation(run_directory, update, split, result):
    """Append one full-split evaluation; undefined metrics become blank cells."""

    if split not in ("train", "validation"):
        raise ValueError("split must be train or validation.")
    if not isinstance(update, int) or update < 0:
        raise ValueError("update must be a nonnegative integer.")
    _append_rows(Path(run_directory) / "evaluations.csv",
                 _summary_rows(update, split, "model", result))


def save_training_baselines(run_directory, results):
    """Save the two fixed training baselines once for this run."""

    rows = [row for name in ("zero", "constant_velocity")
            for row in _summary_rows(0, "train", name, results[name])]
    with (Path(run_directory) / "baselines.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
