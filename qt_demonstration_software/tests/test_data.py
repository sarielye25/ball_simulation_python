import csv
import hashlib
import json
from pathlib import Path

import pytest

from viewer.contracts import ViewerError
from viewer.report_data import load_run
from viewer.training_adapter import training_modules


@pytest.fixture
def demo_run(tmp_path):
    config = training_modules()["config"]
    columns = ("row_id", "motion_group", *config.input_columns, *config.target_columns)
    identity = {}
    for name in ("train", "validation"):
        path = tmp_path / "datasets" / f"{name}.csv"
        path.parent.mkdir(exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerow((f"{name}:2", "always_resting", 0, 0, 0, 1, 0, 0))
        identity[name] = {"sha256": "external-unused", "snapshot": f"datasets/{name}.csv", "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": 1, "row_ids": [f"{name}:2"]}
    manifest = {"format_version": 1, "config": {"input_columns": list(config.input_columns), "target_columns": list(config.target_columns), "tolerances": {name: list(value) for name, value in config.tolerances.items()}, "stopping_ruler": "intermediate"}, "normalization": {}, "model_config": {}, "fixed_parameters": {"mass_kg": 1, "mu_s": .4, "gravity_m_s2": 9.81}, "data_identity": identity, "software": {}, "artifacts": {"evaluations": "evaluations.csv", "baselines": "baselines.csv", "events": "evaluation_events.csv", "batches": "batches.csv", "termination": "termination.json", "checkpoint_index": "checkpoints/index.json"}}
    (tmp_path / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_snapshot_identity_and_missing_optional_files(demo_run):
    bundle = load_run(demo_run)
    assert bundle.splits["train"].row_ids == ("train:2",)
    assert bundle.termination is None
    assert bundle.index is None


def test_bad_snapshot_does_not_fall_back_to_labels(demo_run):
    path = demo_run / "datasets" / "train.csv"
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    bundle = load_run(demo_run)
    assert "train" not in bundle.splits
    assert "SHA-256" in bundle.split_errors["train"]


def test_unknown_schema_rejected(demo_run):
    path = demo_run / "run.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["format_version"] = 2
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ViewerError, match="版本"):
        load_run(demo_run)
