"""Preserve evaluated model states and independently rank validation results."""

import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from config import checkpoint_subdirectory, stopping_ruler


def _portable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_portable(item) for item in value)
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def _atomic_write(path, write):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_index(directory):
    path = directory / "index.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"format_version": 1, "snapshots": [], "best_mse": None,
            "best_pass_rate": None, "last": None, "selected": None}


def _write_index(directory, index):
    encoded = json.dumps(index, indent=2, allow_nan=False).encode("utf-8")
    _atomic_write(directory / "index.json", lambda stream: stream.write(encoded))


def _normalization(values):
    result = {}
    for name, size in (("input_mean", 4), ("input_scale", 4),
                       ("target_mean", 2), ("target_scale", 2)):
        array = np.asarray(values[name], dtype=float)
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain {size} finite values.")
        if name.endswith("scale") and np.any(array <= 0):
            raise ValueError(f"{name} must be positive.")
        result[name] = array.tolist()
    return result


def save_checkpoint(run_directory, update, model, validation_result, *,
                    normalization, model_config, run_config, train_result=None,
                    optimizer=None, scheduler=None, continuation=None,
                    data_identity=None, ruler=stopping_ruler):
    """Save one immutable evaluated state and update both best pointers.

    Call after evaluation and scheduler decisions, before the next optimizer
    update. continuation stores caller-managed epoch, batch permutation/cursor,
    data-generator state, stopping state and exposure count for resumption.
    This function does not make stopping or final-selection decisions.
    """
    if type(update) is not int or update < 0:
        raise ValueError("update must be a nonnegative integer.")
    summary = validation_result["overall"]
    mse = float(summary["standardized_mse"])
    pass_rate = float(summary["pass_rates"][ruler])
    if not math.isfinite(mse) or mse < 0:
        raise ValueError("Validation MSE must be finite and nonnegative.")
    if not math.isfinite(pass_rate) or not 0 <= pass_rate <= 1:
        raise ValueError("Validation pass rate must be finite and in [0, 1].")
    if scheduler is not None and (optimizer is None or scheduler.optimizer is not optimizer):
        raise ValueError("Scheduler must belong to the supplied optimizer.")
    state = model.state_dict()
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Model state contains non-finite values.")
    directory = Path(run_directory) / checkpoint_subdirectory
    directory.mkdir(parents=True, exist_ok=True)
    index = _read_index(directory)
    if index["snapshots"]:
        if index["ruler"] != ruler:
            raise ValueError("Cannot change the selection ruler within a run.")
        if update <= index["snapshots"][-1]["update"]:
            raise ValueError("Updates must increase; use a new run for a rollback branch.")
    path = directory / f"update_{update:06d}.pt"
    if path.exists():
        raise FileExistsError(path)
    payload = _portable({
        "format_version": 1, "update": update, "ruler": ruler,
        "model_state": state, "model_config": model_config,
        "normalization": _normalization(normalization), "run_config": run_config,
        "validation_result": validation_result, "train_result": train_result,
        "data_identity": data_identity,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "learning_rates": [group["lr"] for group in optimizer.param_groups]
                          if optimizer is not None else None,
        "continuation": continuation,
        "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                      "torch": torch.get_rng_state()},
    })
    _atomic_write(path, lambda stream: torch.save(payload, stream))
    entry = {"file": path.name, "update": update,
             "validation_mse": mse, "validation_pass_rate": pass_rate}
    index["snapshots"].append(entry)
    index["ruler"] = ruler
    index["last"] = path.name
    previous_mse = next((item for item in index["snapshots"]
                         if item["file"] == index["best_mse"]), None)
    previous_pass = next((item for item in index["snapshots"]
                          if item["file"] == index["best_pass_rate"]), None)
    if previous_mse is None or mse < previous_mse["validation_mse"]:
        index["best_mse"] = path.name
    if previous_pass is None or (-pass_rate, mse) < (
            -previous_pass["validation_pass_rate"], previous_pass["validation_mse"]):
        index["best_pass_rate"] = path.name
    _write_index(directory, index)
    return path


def select_checkpoint(run_directory, choice="best_mse", *, reason):
    """Record an explicit final choice by best pointer or saved filename."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A selection reason is required.")
    directory = Path(run_directory) / checkpoint_subdirectory
    index = _read_index(directory)
    filename = index[choice] if choice in ("best_mse", "best_pass_rate", "last") else choice
    if filename not in {item["file"] for item in index["snapshots"]}:
        raise ValueError("Selection must refer to a recorded checkpoint.")
    if not (directory / filename).is_file():
        raise FileNotFoundError(directory / filename)
    index["selected"] = filename
    index["selection_reason"] = reason
    _write_index(directory, index)
    return directory / filename


def load_checkpoint(path, model, *, optimizer=None, scheduler=None, restore_rng=False):
    """Restore CPU model states; return normalization and continuation metadata.

    Construct the matching architecture and, for training, optimizer/scheduler
    before loading. The caller restores the sampler and stopping state and calls
    model.train() when resuming. Inference loads default to evaluation mode.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format.")
    _normalization(payload["normalization"])
    if optimizer is not None and payload["optimizer_state"] is None:
        raise ValueError("Checkpoint contains no optimizer state.")
    if scheduler is not None:
        if optimizer is None or scheduler.optimizer is not optimizer:
            raise ValueError("Scheduler must belong to the supplied optimizer.")
        if payload["scheduler_state"] is None:
            raise ValueError("Checkpoint contains no scheduler state.")
    model.load_state_dict(payload["model_state"], strict=True)
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng:
        rng = payload["rng_state"]
        random.setstate(rng["python"])
        numpy_state = rng["numpy"]
        np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                             *numpy_state[2:]))
        torch.set_rng_state(rng["torch"])
    model.eval()
    return payload
