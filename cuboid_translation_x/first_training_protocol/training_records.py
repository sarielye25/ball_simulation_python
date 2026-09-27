"""Save evaluation summaries separately from metric calculation and plotting."""

import csv
from datetime import datetime
from pathlib import Path

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


def create_run_directory():
    """Create a separate directory for each run without overwriting older runs."""

    run_directory = TRAINING_DATA_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / checkpoint_subdirectory).mkdir()
    (run_directory / "figures").mkdir()
    return run_directory


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
