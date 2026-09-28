"""CPU checkpoint inference with saved-summary verification."""
import math
import threading
import json

import numpy as np
import torch

from .contracts import GROUPS, RULERS, PredictionResult, ViewerError
from .training_adapter import training_modules


def _compare(saved, actual, prefix, mismatches):
    if not isinstance(saved, dict):
        mismatches.append(f"{prefix}: 保存摘要缺失")
        return
    for field, value in actual.items():
        if field == "pass_rates":
            _compare(saved.get(field), value, f"{prefix}.{field}", mismatches)
            continue
        original = saved.get(field)
        if original is None or value is None:
            if original != value:
                mismatches.append(f"{prefix}.{field}: 保存={original} 重算={value}")
        elif field == "count":
            if int(original) != value:
                mismatches.append(f"{prefix}.{field}: 保存={original} 重算={value}")
        elif not math.isclose(float(original), value, rel_tol=1e-6, abs_tol=1e-8):
            mismatches.append(f"{prefix}.{field}: 保存={original} 重算={value} 差值={value - float(original):.9g}")


def infer_checkpoint(bundle, checkpoint_file, split, cancel_event=None):
    cancel_event = cancel_event or threading.Event()
    if split not in bundle.splits:
        raise ViewerError(bundle.split_errors.get(split, f"无数据集: {split}"))
    index = bundle.index or {}
    entries = {item["file"]: item for item in index.get("snapshots", [])}
    if checkpoint_file not in entries:
        raise ViewerError(f"checkpoint 未列于 index: {checkpoint_file}")
    checkpoint_dir = (bundle.run_dir / bundle.manifest["artifacts"]["checkpoint_index"]).resolve().parent
    path = (checkpoint_dir / checkpoint_file).resolve()
    if path.parent != checkpoint_dir:
        raise ViewerError(f"checkpoint 路径越界: {checkpoint_file}")
    before = path.stat()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("update") != entries[checkpoint_file]["update"]:
        raise ViewerError(f"{path}: 版本或 update 不匹配")
    for checkpoint_key, manifest_key in (("run_config", "config"), ("model_config", "model_config"), ("normalization", "normalization"), ("data_identity", "data_identity")):
        if json.loads(json.dumps(payload.get(checkpoint_key))) != bundle.manifest[manifest_key]:
            raise ViewerError(f"{path}: {checkpoint_key} 与 run.json 不一致")
    if payload["model_config"] != {"widths": [4, 32, 32, 2], "activation": "ReLU"}:
        raise ViewerError("不支持的模型结构")
    norm = payload["normalization"]
    for name, size in (("input_mean", 4), ("input_scale", 4), ("target_mean", 2), ("target_scale", 2)):
        values = np.asarray(norm[name], dtype=np.float64)
        if values.shape != (size,) or not np.isfinite(values).all() or (name.endswith("scale") and np.any(values <= 0)):
            raise ViewerError(f"{path}: {name} 无效")
    modules = training_modules()
    data, metrics = modules["data"], modules["metrics"]
    with torch.random.fork_rng(devices=[]):
        model = modules["neural_network"].transmodel().to(device="cpu", dtype=torch.float32)
        modules["checkpoints"].load_checkpoint(path, model, restore_rng=False)
    rows = bundle.splits[split]
    standardized = data.standardize_inputs(rows.inputs, norm["input_mean"], norm["input_scale"])
    predictions = []
    for start in range(0, len(rows.row_ids), 128):
        if cancel_event.is_set():
            raise InterruptedError("已取消")
        predictions.append(metrics.predict_full_split(model, standardized[start:start + 128], 128))
    standardized_predictions = np.concatenate(predictions)
    standardized_targets = data.standardize_targets(rows.targets, norm["target_mean"], norm["target_scale"])
    errors = metrics.calculate_errors(standardized_predictions, standardized_targets, norm["target_mean"], norm["target_scale"])
    tolerances = payload["run_config"]["tolerances"]
    pass_masks = metrics.calculate_pass_masks(errors["physical_absolute"], tolerances)
    fixed = bundle.manifest["fixed_parameters"]
    boundary = metrics.make_breakaway_selection_mask(rows.inputs, fixed["mu_s"] * fixed["mass_kg"] * fixed["gravity_m_s2"], bundle.manifest["config"]["breakaway_band_N"])
    squared, absolute = errors["standardized_squared"], errors["physical_absolute"]
    summaries = {"overall": metrics.summarize_metrics(squared, absolute, pass_masks)}
    summaries.update(metrics.group_metrics(squared, absolute, pass_masks, rows.groups))
    summaries["breakaway"] = metrics.summarize_metrics(squared, absolute, pass_masks, boundary)
    mismatches = []
    saved = payload.get(f"{split}_result")
    if saved is not None:
        for scope in summaries:
            record = saved.get("overall") if scope == "overall" else saved.get("boundary") if scope == "breakaway" else saved.get("groups", {}).get(scope)
            _compare(record, summaries[scope], f"checkpoint.{split}.{scope}", mismatches)
    for scope, actual in summaries.items():
        record = next((row for row in bundle.evaluations if row["update"] == payload["update"] and row["split"] == split and row["scope"] == scope), None)
        if record is None:
            continue
        saved_row = {field: record[field] for field in actual if field != "pass_rates"}
        saved_row["pass_rates"] = {ruler: record[f"pass_rate_{ruler}"] for ruler in RULERS}
        _compare(saved_row, actual, f"evaluations.csv.{split}.{scope}", mismatches)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ViewerError(f"{path}: 读取期间文件变化")
    arrays = (rows.inputs, rows.targets, errors["physical_predictions"], errors["physical_signed"], absolute, boundary, *pass_masks.values())
    for array in arrays:
        array.setflags(write=False)
    return PredictionResult(bundle.generation, checkpoint_file, payload["update"], split, rows.row_ids, rows.inputs, rows.targets, errors["physical_predictions"], errors["physical_signed"], absolute, rows.groups, boundary, pass_masks, summaries, tuple(mismatches))
