"""Validate saved run artifacts and produce factual summaries."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .contracts import GROUPS, INPUT_COLUMNS, RULERS, TARGET_COLUMNS, RunBundle, SplitData, ViewerError
from .training_adapter import training_modules, TRAINING_ROOT

EVENT_FIELDS = ("update", "epoch", "exposures", "elapsed_seconds", "lr_before", "lr_next", "scheduler_checked", "reason", "should_stop", "patience_reached", "max_updates_reached")
BATCH_FIELDS = ("update", "epoch", "batch_rows", "exposures", "elapsed_seconds", "learning_rate", "batch_mse_before_update")


def _path(root, reference):
    if not isinstance(reference, str) or not reference:
        raise ViewerError(f"无效路径引用: {reference!r}")
    path = (root / reference).resolve()
    if not path.is_relative_to(root):
        raise ViewerError(f"路径越界: {reference}")
    return path


def _read_json(path):
    before = path.stat()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ViewerError(f"{path}: {error}") from error
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ViewerError(f"{path}: 读取期间文件变化；请停止训练后刷新")
    return value


def _read_csv(path, fields, required=None):
    if not path.is_file():
        return ()
    before = path.stat()
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or (list(reader.fieldnames) != list(fields) if required is None else not set(required).issubset(reader.fieldnames)):
                raise ViewerError(f"{path}: 未知 CSV 表头")
            rows = []
            for line, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ViewerError(f"{path}:{line}: 截断或多余列")
                rows.append(row)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ViewerError(f"{path}: {error}") from error
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ViewerError(f"{path}: 读取期间文件变化；请停止训练后刷新")
    return tuple(rows)


def _metric_rows(rows, path, predictors, fields):
    seen = set()
    output = []
    for row in rows:
        try:
            update = int(row["update"])
            count = int(row["count"])
            if update < 0 or count < 0 or row["split"] not in ("train", "validation") or row["predictor"] not in predictors:
                raise ValueError("invalid key or count")
            key = (update, row["split"], row["predictor"], row["scope"])
            if key in seen:
                raise ValueError("duplicate key")
            seen.add(key)
            converted = dict(row, update=update, count=count)
            for field in fields[5:]:
                raw = row[field]
                if raw == "" and count == 0:
                    converted[field] = None
                else:
                    number = float(raw)
                    if not math.isfinite(number) or number < 0 or (field.startswith("pass_rate_") and number > 1):
                        raise ValueError(f"invalid {field}")
                    converted[field] = number
            output.append(converted)
        except (KeyError, TypeError, ValueError) as error:
            raise ViewerError(f"{path}: {error}") from error
    return tuple(sorted(output, key=lambda row: (row["update"], row["split"], row["scope"])))


def _split(root, labels, name, identity, fixed):
    modules = training_modules()
    data = modules["data"]
    source = _path(root, identity["snapshot"])
    snapshot = source.is_file()
    if not snapshot:
        source = labels / f"{name}.csv"
    before = source.stat()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = identity["snapshot_sha256"] if snapshot else identity["sha256"]
    if digest != expected:
        raise ViewerError(f"{source}: SHA-256 不匹配")
    if snapshot:
        rows = _read_csv(source, ("row_id", "motion_group", *INPUT_COLUMNS, *TARGET_COLUMNS))
        row_ids = tuple(row["row_id"] for row in rows)
        inputs = np.asarray([[float(row[field]) for field in INPUT_COLUMNS] for row in rows], dtype=np.float64)
        targets = np.asarray([[float(row[field]) for field in TARGET_COLUMNS] for row in rows], dtype=np.float64)
        groups = tuple(row["motion_group"] for row in rows)
    else:
        raw_inputs, raw_targets = data.load_split(source, expected_rows=identity["rows"])
        inputs = np.asarray(raw_inputs, dtype=np.float64)
        targets = np.asarray(raw_targets, dtype=np.float64)
        row_ids = tuple(f"{name}:{number}" for number in range(2, len(inputs) + 2))
        groups = tuple(data.assign_groups(inputs, targets, fixed["mass_kg"], fixed["mu_s"], fixed["gravity_m_s2"]))
    if inputs.shape != (identity["rows"], 4) or targets.shape != (identity["rows"], 2) or not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ViewerError(f"{source}: 数据维度或数值无效")
    if np.any(inputs[:, 2] < 0) or np.any(inputs[:, 2] > inputs[:, 3]):
        raise ViewerError(f"{source}: 时间范围无效")
    if row_ids != tuple(identity["row_ids"]) or len(set(row_ids)) != len(row_ids):
        raise ViewerError(f"{source}: row ID 不匹配")
    expected_groups = tuple(data.assign_groups(inputs, targets, fixed["mass_kg"], fixed["mu_s"], fixed["gravity_m_s2"]))
    if groups != expected_groups or set(groups) - set(GROUPS):
        raise ViewerError(f"{source}: 运动组不匹配")
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ViewerError(f"{source}: 读取期间文件变化")
    inputs.setflags(write=False)
    targets.setflags(write=False)
    return SplitData(name, row_ids, groups, inputs, targets, digest, source)


def load_run(run_dir, labels_dir=None):
    root = Path(run_dir).resolve()
    labels = Path(labels_dir).resolve() if labels_dir else TRAINING_ROOT / "labels"
    manifest = _read_json(root / "run.json")
    if manifest.get("format_version") != 1:
        raise ViewerError(f"{root / 'run.json'}: 不支持的版本")
    for field in ("config", "normalization", "model_config", "fixed_parameters", "data_identity", "software", "artifacts"):
        if not isinstance(manifest.get(field), dict):
            raise ViewerError(f"run.json: 缺少 {field}")
    config = manifest["config"]
    if tuple(config.get("input_columns", ())) != INPUT_COLUMNS or tuple(config.get("target_columns", ())) != TARGET_COLUMNS:
        raise ViewerError("run.json: 物理列与 4→2 契约不符")
    for ruler in RULERS:
        tolerance = config.get("tolerances", {}).get(ruler)
        if not isinstance(tolerance, (list, tuple)) or len(tolerance) != 2 or any(not math.isfinite(float(value)) or float(value) <= 0 for value in tolerance):
            raise ViewerError(f"run.json: {ruler} 容差无效")
    artifacts = manifest["artifacts"]
    paths = {name: _path(root, value) for name, value in artifacts.items()}
    fields = training_modules()["training_records"].FIELDS
    evaluations = _metric_rows(_read_csv(paths["evaluations"], fields), paths["evaluations"], {"model"}, fields)
    baselines = _metric_rows(_read_csv(paths["baselines"], fields), paths["baselines"], {"zero", "constant_velocity"}, fields)
    events = _read_csv(paths["events"], (), EVENT_FIELDS)
    batches = _read_csv(paths["batches"], (), BATCH_FIELDS)
    termination = _read_json(paths["termination"]) if paths["termination"].is_file() else None
    index = _read_json(paths["checkpoint_index"]) if paths["checkpoint_index"].is_file() else None
    if index is not None:
        if index.get("format_version") != 1:
            raise ViewerError("checkpoints/index.json: 不支持的版本")
        names = set()
        checkpoint_dir = paths["checkpoint_index"].parent
        for item in index.get("snapshots", []):
            path = _path(root, str(checkpoint_dir.relative_to(root) / item["file"]))
            if path.parent != checkpoint_dir or item["file"] in names or int(item["update"]) < 0:
                raise ViewerError("checkpoints/index.json: 无效快照")
            names.add(item["file"])
        for pointer in ("selected", "last", "best_mse", "best_pass_rate"):
            if index.get(pointer) is not None and index[pointer] not in names:
                raise ViewerError(f"checkpoints/index.json: 无效 {pointer}")
    splits, errors = {}, {}
    for name in ("train", "validation"):
        try:
            splits[name] = _split(root, labels, name, manifest["data_identity"][name], manifest["fixed_parameters"])
        except (OSError, KeyError, ValueError, ViewerError) as error:
            errors[name] = str(error)
    warnings = tuple(f"{name}: {message}" for name, message in errors.items())
    return RunBundle(root, manifest, evaluations, baselines, events, batches, termination, index, splits, errors, warnings)


def build_summary(bundle):
    term = bundle.termination or {}
    status = term.get("status", "未知/未完成")
    completed = term.get("update") if status == "completed" else term.get("last_completed_update", "未记录")
    latest = max((row["update"] for row in bundle.evaluations), default="未记录")
    index = bundle.index or {}
    selected = index.get("selected")
    lines = [f"状态：{status}    完成 update：{completed}    最后评估 update：{latest}",
             f"已进入 epoch：{term.get('epoch', '未记录')}    曝光次数：{term.get('exposures', '未记录')}    耗时：{term.get('elapsed_seconds', '未记录')} 秒",
             f"停止原因：{term.get('reason', '未记录')}    {term.get('message', '')}",
             f"选中模型：{selected or '无正式选中模型'}    最后有效 checkpoint：{term.get('last_valid_checkpoint', index.get('last', '未记录'))}",
             f"目标通过率：{bundle.manifest['config'].get('target_pass_rate', '未记录')}    停止容差：{bundle.manifest['config'].get('stopping_ruler', '未记录')}"]
    if bundle.warnings:
        lines.extend(bundle.warnings)
    return "\n".join(lines)
