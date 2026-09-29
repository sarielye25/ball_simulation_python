"""Publish training runs in the Qt viewer's shared data contract."""

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch


def descriptor(path, root, append_only=False):
    content = path.read_bytes()
    return {
        "file": path.relative_to(root).as_posix(),
        "committed_bytes" if append_only else "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def artifacts(root):
    files = {
        "evaluations": ("evaluations.csv", True),
        "baselines": ("baselines.csv", False),
        "evaluation_events": ("evaluation_events.csv", True),
        "batches": ("batches.csv", True),
    }
    return {
        name: descriptor(root / filename, root, append_only) if (root / filename).is_file() else None
        for name, (filename, append_only) in files.items()
    }


def initial_manifest(root, splits, config, diagnostic, *, run_config=None,
                     normalization=None, model_config=None):
    manifest = {
        "protocol_version": 1,
        "group_id": config.training_data_dir.parent.name,
        "run_id": root.name,
        "generation": 0,
        "status": "running",
        "purpose": "diagnostic" if diagnostic else "training",
        "splits": [
            {
                "id": name,
                "name": name.title(),
                "role": name,
                "rows": len(split["row_ids"]),
                "dataset": descriptor(root / "datasets" / f"{name}.csv", root),
            }
            for name, split in splits.items()
        ],
        "columns": {
            "inputs": config.input_columns,
            "targets": config.target_columns,
            "units": dict(zip(config.input_columns + config.target_columns,
                              ("m/s", "N", "s", "s", "m", "m/s"))),
        },
        "tolerances": config.tolerances,
        "stopping_ruler": config.stopping_ruler,
        "artifacts": artifacts(root),
        "checkpoints": [],
        "selected_checkpoint": None,
        "termination": None,
    }
    if run_config is not None:
        manifest["run_config"] = run_config
    if normalization is not None:
        manifest["normalization"] = {
            name: np.asarray(values, dtype=float).tolist()
            for name, values in normalization.items()
        }
    if model_config is not None:
        manifest["model_config"] = model_config
    return manifest


def save_predictions(root, checkpoint_path, model, splits, normalization, target_columns):
    checkpoint_id = checkpoint_path.stem
    result = {}
    model.eval()
    with torch.inference_mode():
        for name, split in splits.items():
            batches = []
            for start in range(0, len(split["inputs"]), 1024):
                inputs = torch.as_tensor(split["inputs"][start:start + 1024], dtype=torch.float32)
                batches.append(model(inputs).cpu().numpy())
            predicted = np.concatenate(batches, axis=0)
            predicted = predicted * np.asarray(normalization["target_scale"]) + np.asarray(normalization["target_mean"])
            if not np.isfinite(predicted).all():
                raise ValueError("Non-finite prediction")
            path = root / "predictions" / checkpoint_id / f"{name}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("row_id", *(f"pred_{target}" for target in target_columns)))
                for row_id, values in zip(split["row_ids"], predicted, strict=True):
                    writer.writerow((row_id, *values))
            result[name] = descriptor(path, root)
    return {"id": checkpoint_id, "update": int(checkpoint_id.removeprefix("update_")),
            "predictions": result, "model_file": checkpoint_path.relative_to(root).as_posix()}


def publish(root, manifest, write_json):
    published = {**manifest, "artifacts": artifacts(root)}
    write_json(root / "run.json", published)
