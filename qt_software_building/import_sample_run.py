"""Import one completed sample-training run into the shared viewer contract."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


METRIC_FILES = ("evaluations.csv", "baselines.csv", "evaluation_events.csv", "batches.csv")


def descriptor(path, root, append_only=False):
    content = path.read_bytes()
    return {
        "file": path.relative_to(root).as_posix(),
        "committed_bytes" if append_only else "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def publish(path, manifest):
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    descriptor_id, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor_id, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def import_run(source, data_root, training_source):
    source = Path(source).resolve()
    data_root = Path(data_root).resolve()
    training_source = Path(training_source).resolve()
    legacy = json.loads((source / "run.json").read_text(encoding="utf-8"))
    if legacy.get("format_version") != 1:
        raise ValueError("Expected a sample-training v1 run")
    group_id = "sample_training"
    run_id = f"imported_{source.name}"
    target = data_root / group_id / "runs" / run_id
    if target.exists():
        raise FileExistsError(target)
    sys.path.insert(0, str(training_source))
    from neural_network import transmodel
    import torch
    import numpy as np

    config = legacy["config"]
    inputs = config["input_columns"]
    targets = config["target_columns"]
    normalization = legacy["normalization"]
    index = json.loads((source / "checkpoints" / "index.json").read_text(encoding="utf-8"))
    termination = json.loads((source / "termination.json").read_text(encoding="utf-8"))
    target.mkdir(parents=True)
    try:
        for filename in METRIC_FILES:
            source_file = source / filename
            if source_file.is_file():
                shutil.copy2(source_file, target / filename)
        shutil.copytree(source / "datasets", target / "datasets")
        shutil.copytree(source / "checkpoints", target / "checkpoints")
        shutil.copy2(source / "run.json", target / "training_run.json")
        shutil.copy2(source / "termination.json", target / "termination.json")
        splits = []
        dataset_rows = {}
        for split_id, role in (("train", "train"), ("validation", "validation")):
            path = target / "datasets" / f"{split_id}.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                rows = tuple(csv.DictReader(stream))
            dataset_rows[split_id] = rows
            splits.append({"id": split_id, "name": split_id.title(), "role": role, "rows": len(rows), "dataset": descriptor(path, target)})
        checkpoints = []
        for entry in index["snapshots"]:
            checkpoint_file = entry["file"]
            checkpoint_id = Path(checkpoint_file).stem
            payload = torch.load(source / "checkpoints" / checkpoint_file, map_location="cpu", weights_only=True)
            model = transmodel().to(device="cpu", dtype=torch.float32)
            model.load_state_dict(payload["model_state"], strict=True)
            model.eval()
            prediction_files = {}
            for split_id, rows in dataset_rows.items():
                physical = np.asarray([[float(row[name]) for name in inputs] for row in rows], dtype=np.float64)
                standardized = (physical - np.asarray(normalization["input_mean"])) / np.asarray(normalization["input_scale"])
                with torch.inference_mode():
                    predicted = model(torch.as_tensor(standardized, dtype=torch.float32)).numpy()
                predicted = predicted * np.asarray(normalization["target_scale"]) + np.asarray(normalization["target_mean"])
                path = target / "predictions" / checkpoint_id / f"{split_id}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("row_id", *(f"pred_{name}" for name in targets)))
                    for row, values in zip(rows, predicted, strict=True):
                        writer.writerow((row["row_id"], *values))
                prediction_files[split_id] = descriptor(path, target)
            checkpoints.append({"id": checkpoint_id, "update": entry["update"], "predictions": prediction_files, "model_file": f"checkpoints/{checkpoint_file}"})
        selected = Path(index["selected"]).stem if index.get("selected") else None
        manifest = {
            "protocol_version": 1, "group_id": group_id, "run_id": run_id,
            "generation": 1, "status": termination["status"], "purpose": "diagnostic",
            "splits": splits,
            "columns": {"inputs": inputs, "targets": targets, "units": legacy["units"]},
            "tolerances": config["tolerances"], "stopping_ruler": config["stopping_ruler"],
            "artifacts": {
                "evaluations": descriptor(target / "evaluations.csv", target, True),
                "baselines": descriptor(target / "baselines.csv", target),
                "evaluation_events": descriptor(target / "evaluation_events.csv", target, True),
                "batches": descriptor(target / "batches.csv", target, True),
            },
            "checkpoints": checkpoints, "selected_checkpoint": selected,
            "termination": {"reason": termination["reason"], "final_update": termination["update"], "selected_checkpoint": selected},
        }
        publish(target / "run.json", manifest)
    except BaseException:
        shutil.rmtree(target)
        raise
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent / "model_training" / "data")
    parser.add_argument("--training-source", type=Path, default=Path(__file__).resolve().parent.parent / "model_training" / "sample_training")
    args = parser.parse_args()
    print(import_run(args.source, args.data_root, args.training_source))
