"""Read one published run generation from the shared data directory."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re

from .model import Checkpoint, Group, Metric, Prediction, RunSnapshot, Sample, Split


IDENTIFIER = re.compile(r"[A-Za-z0-9_-]+\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
METRIC_NAMES = (
    "standardized_mse", "displacement_standardized_mse", "velocity_standardized_mse",
    "displacement_loss_contribution", "velocity_loss_contribution",
    "displacement_mae_m", "displacement_p95_m", "displacement_max_m",
    "velocity_mae_m_s", "velocity_p95_m_s", "velocity_max_m_s",
    "pass_rate_coarse", "pass_rate_intermediate", "pass_rate_fine",
)
METRIC_HEADER = ("update", "split", "predictor", "scope", "count", *METRIC_NAMES)


class DataError(ValueError):
    """A published run violates the shared data contract."""


def _require(condition, context, message):
    if not condition:
        raise DataError(f"{context}: {message}")


def _identifier(value, context):
    _require(isinstance(value, str) and IDENTIFIER.fullmatch(value), context, "invalid ID")
    return value


def _integer(value, context, minimum=0):
    _require(type(value) is int and value >= minimum, context, "invalid integer")
    return value


def _number(value, context, positive=False):
    _require(type(value) in (int, float) and math.isfinite(value), context, "invalid number")
    _require(value > 0 if positive else value >= 0, context, "number out of range")
    return float(value)


def _text(value, context, allow_empty=False):
    _require(isinstance(value, str) and (allow_empty or bool(value)), context, "invalid text")
    return value


def _object(value, context):
    _require(isinstance(value, dict), context, "expected object")
    return value


def _array(value, context):
    _require(isinstance(value, list), context, "expected list")
    return value


def _json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {value}")))
    except (OSError, UnicodeError, ValueError) as error:
        raise DataError(f"{path}: {error}") from error


def _path(root, reference):
    _require(isinstance(reference, str) and bool(reference), root, "invalid artifact path")
    pure = PurePosixPath(reference)
    _require(not pure.is_absolute() and ":" not in reference and "\\" not in reference and all(part not in (".", "..", "") for part in reference.split("/")), root, f"invalid artifact path {reference}")
    path = (root / reference).resolve()
    _require(path.is_relative_to(root) and path.is_file(), root, f"missing or escaping artifact {reference}")
    return path


def _file_bytes(root, descriptor, append_only=False):
    descriptor = _object(descriptor, root)
    path = _path(root, descriptor.get("file"))
    size_key = "committed_bytes" if append_only else "bytes"
    size = _integer(descriptor.get(size_key), path, 1)
    digest = descriptor.get("sha256")
    _require(isinstance(digest, str) and SHA256.fullmatch(digest), path, "invalid SHA-256")
    try:
        with path.open("rb") as stream:
            content = stream.read(size)
            if not append_only:
                _require(not stream.read(1), path, "immutable file has extra bytes")
    except OSError as error:
        raise DataError(f"{path}: {error}") from error
    _require(len(content) == size, path, "file shorter than published size")
    _require(hashlib.sha256(content).hexdigest() == digest, path, "SHA-256 mismatch")
    if append_only:
        _require(content.endswith(b"\n"), path, "committed prefix ends mid-record")
    return path, content


def _csv_rows(root, descriptor, header, append_only=False):
    path, content = _file_bytes(root, descriptor, append_only)
    _require(content.endswith(b"\n"), path, "CSV ends mid-record")
    try:
        reader = csv.reader(io.StringIO(content.decode("utf-8"), newline=""), strict=True)
        rows = list(reader)
    except (UnicodeError, csv.Error) as error:
        raise DataError(f"{path}: {error}") from error
    _require(bool(rows) and tuple(rows[0]) == tuple(header), path, "CSV header mismatch")
    for index, row in enumerate(rows[1:], 2):
        _require(len(row) == len(header), path, f"wrong field count on row {index}")
    return path, tuple(dict(zip(header, row)) for row in rows[1:])


def _float_cell(row, name, context):
    try:
        value = float(row[name])
    except (ValueError, KeyError) as error:
        raise DataError(f"{context}: invalid {name}") from error
    _require(math.isfinite(value), context, f"nonfinite {name}")
    return value


def _int_cell(row, name, context):
    try:
        value = int(row[name])
    except (ValueError, KeyError) as error:
        raise DataError(f"{context}: invalid {name}") from error
    _require(str(value) == row[name] and value >= 0, context, f"invalid {name}")
    return value


def list_groups(data_root):
    root = Path(data_root).resolve()
    _require(root.is_dir(), root, "data directory missing")
    groups = []
    for directory in sorted(root.iterdir()):
        if directory.is_dir() and (directory / "group.json").is_file():
            groups.append(_load_group(directory))
    return tuple(groups)


def list_runs(data_root, group_id):
    _identifier(group_id, "group_id")
    directory = Path(data_root).resolve() / group_id
    _load_group(directory)
    runs = directory / "runs"
    if not runs.is_dir():
        return ()
    available = []
    for path in runs.iterdir():
        if path.is_dir() and (path / "run.json").is_file():
            manifest = _json(path / "run.json")
            if isinstance(manifest, dict) and manifest.get("protocol_version") == 1:
                available.append(path.name)
    return tuple(sorted(available))


def _load_group(directory):
    group = _object(_json(directory / "group.json"), directory)
    _require(type(group.get("protocol_version")) is int and group["protocol_version"] == 1, directory, "unsupported group version")
    group_id = _identifier(group.get("group_id"), directory)
    _require(group_id == directory.name, directory, "group ID mismatch")
    return Group(group_id, _text(group.get("name"), directory), _text(group.get("description"), directory, True), _text(group.get("source"), directory))


def _metric_unit(name):
    if name.startswith("pass_rate_") or "standardized" in name or "contribution" in name:
        return ""
    if name.startswith("displacement_"):
        return "m"
    if name.startswith("velocity_"):
        return "m/s"
    return ""


def _load_metrics(root, descriptor, split_ids, baseline=False):
    if descriptor is None:
        return ()
    path, rows = _csv_rows(root, descriptor, METRIC_HEADER, append_only=not baseline)
    seen = set()
    result = []
    for row in rows:
        update = _int_cell(row, "update", path)
        count = _int_cell(row, "count", path)
        split_id = row["split"]
        predictor = row["predictor"]
        scope = row["scope"]
        _require(split_id in split_ids and bool(scope), path, "unknown split or empty scope")
        _require(predictor in (("zero", "constant_velocity") if baseline else ("model",)), path, "invalid predictor")
        if baseline:
            _require(update == 0 and split_ids[split_id] == "train", path, "invalid baseline")
        key = (update, split_id, predictor, scope)
        _require(key not in seen, path, "duplicate metric key")
        seen.add(key)
        for name in METRIC_NAMES:
            raw = row[name]
            value = None if raw == "" and count == 0 else _float_cell(row, name, path)
            if value is not None:
                _require(value >= 0 and (not name.startswith("pass_rate_") or value <= 1), path, f"invalid {name}")
            result.append(Metric(update, split_id, predictor, scope, name, value, count, _metric_unit(name)))
    return tuple(result)


def load_run(data_root, group_id, run_id):
    """Validate and return one complete published generation, with no writes."""
    _identifier(group_id, "group_id")
    _identifier(run_id, "run_id")
    group_dir = Path(data_root).resolve() / group_id
    group = _load_group(group_dir)
    root = (group_dir / "runs" / run_id).resolve()
    _require(root.is_relative_to(group_dir.resolve() / "runs") and root.name == run_id, root, "run path escapes group")
    manifest = _object(_json(root / "run.json"), root)
    _require(type(manifest.get("protocol_version")) is int and manifest["protocol_version"] == 1, root, "unsupported run version")
    _require(manifest.get("group_id") == group_id and manifest.get("run_id") == run_id, root, "run identity mismatch")
    generation = _integer(manifest.get("generation"), root)
    status = manifest.get("status")
    _require(status in ("running", "completed", "aborted"), root, "invalid status")
    purpose = manifest.get("purpose")
    _require(purpose in ("diagnostic", "training"), root, "invalid purpose")

    columns = _object(manifest.get("columns"), root)
    inputs = tuple(_array(columns.get("inputs"), root))
    targets = tuple(_array(columns.get("targets"), root))
    _require(inputs and targets and all(isinstance(name, str) and name and name not in ("row_id", "motion_group") for name in (*inputs, *targets)) and len(set((*inputs, *targets))) == len(inputs) + len(targets), root, "invalid columns")
    units = _object(columns.get("units"), root)
    _require(all(isinstance(units.get(name), str) and units[name] for name in (*inputs, *targets)), root, "missing units")
    tolerances_raw = _object(manifest.get("tolerances"), root)
    _require(bool(tolerances_raw), root, "missing tolerances")
    tolerances = {}
    for ruler, values in tolerances_raw.items():
        _identifier(ruler, root)
        values = _array(values, root)
        _require(len(values) == len(targets), root, "tolerance length mismatch")
        tolerances[ruler] = tuple(_number(value, root, True) for value in values)
    stopping_ruler = manifest.get("stopping_ruler")
    _require(stopping_ruler in tolerances, root, "invalid stopping ruler")

    split_items = _array(manifest.get("splits"), root)
    _require(bool(split_items), root, "no splits")
    splits = []
    samples = {}
    for item in split_items:
        item = _object(item, root)
        split_id = _identifier(item.get("id"), root)
        _require(split_id not in samples, root, "duplicate split")
        role = item.get("role")
        _require(role in ("train", "validation"), root, "invalid split role")
        count = _integer(item.get("rows"), root, 1)
        path, rows = _csv_rows(root, item.get("dataset"), ("row_id", "motion_group", *inputs, *targets))
        _require(len(rows) == count, path, "row count mismatch")
        seen_ids = set()
        sample_rows = []
        for row in rows:
            row_id = _text(row["row_id"], path)
            _require(row_id not in seen_ids, path, "duplicate row ID")
            seen_ids.add(row_id)
            sample_rows.append(Sample(row_id, _text(row["motion_group"], path), tuple(_float_cell(row, name, path) for name in inputs), tuple(_float_cell(row, name, path) for name in targets)))
        splits.append(Split(split_id, _text(item.get("name"), root), role, count))
        samples[split_id] = tuple(sample_rows)

    split_roles = {split.id: split.role for split in splits}
    artifacts = _object(manifest.get("artifacts"), root)
    _require("evaluations" in artifacts, root, "missing evaluations descriptor")
    metrics = list(_load_metrics(root, artifacts["evaluations"], split_roles))
    metrics.extend(_load_metrics(root, artifacts.get("baselines"), split_roles, True))
    for optional in ("evaluation_events", "batches"):
        descriptor = artifacts.get(optional)
        if descriptor is not None:
            _file_bytes(root, descriptor, append_only=True)

    checkpoints = []
    predictions = {}
    prior_update = -1
    for item in _array(manifest.get("checkpoints"), root):
        item = _object(item, root)
        checkpoint_id = _identifier(item.get("id"), root)
        _require(all(existing.id != checkpoint_id for existing in checkpoints), root, "duplicate checkpoint")
        update = _integer(item.get("update"), root)
        _require(update > prior_update, root, "checkpoint updates not increasing")
        prior_update = update
        listed = _object(item.get("predictions"), root)
        _require(bool(listed) and set(listed) <= set(samples), root, "invalid prediction splits")
        model_file = item.get("model_file")
        _require(model_file is None or isinstance(model_file, str), root, "invalid model reference")
        for split_id, descriptor in listed.items():
            path, rows = _csv_rows(root, descriptor, ("row_id", *(f"pred_{name}" for name in targets)))
            row_ids = [row["row_id"] for row in rows]
            _require(len(row_ids) == len(set(row_ids)) and set(row_ids) == {sample.row_id for sample in samples[split_id]}, path, "prediction row IDs do not match dataset")
            predictions[(checkpoint_id, split_id)] = tuple(Prediction(row["row_id"], tuple(_float_cell(row, f"pred_{name}", path) for name in targets)) for row in rows)
        checkpoints.append(Checkpoint(checkpoint_id, update, tuple(listed)))

    selected = manifest.get("selected_checkpoint")
    _require(selected is None or any(item.id == selected for item in checkpoints), root, "invalid selected checkpoint")
    termination = manifest.get("termination")
    if status == "running":
        _require(termination is None, root, "running run has termination")
    else:
        termination = _object(termination, root)
        _text(termination.get("reason"), root)
        if status == "completed":
            _integer(termination.get("final_update"), root)
            _require(selected is not None and termination.get("selected_checkpoint") == selected, root, "invalid completion selection")
        else:
            _integer(termination.get("last_completed_update"), root)
            _text(termination.get("message"), root, True)
    if generation == 0:
        _require(status == "running" and not checkpoints and not metrics and selected is None, root, "invalid initial generation")
    return RunSnapshot(group, run_id, generation, status, purpose, inputs, targets, dict(units), tolerances, stopping_ruler, tuple(splits), samples, tuple(metrics), tuple(checkpoints), predictions, selected, termination)
