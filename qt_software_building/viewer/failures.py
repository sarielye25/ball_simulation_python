"""Analyze per-sample prediction errors in memory."""

from dataclasses import dataclass

from .model import RunSnapshot


@dataclass(frozen=True)
class FailureRow:
    row_id: str
    motion_group: str
    inputs: tuple[float, ...]
    targets: tuple[float, ...]
    predictions: tuple[float, ...]
    signed_errors: tuple[float, ...]
    absolute_errors: tuple[float, ...]
    passed: bool
    severity: float


@dataclass(frozen=True)
class FailureReport:
    split_id: str
    checkpoint_id: str
    ruler: str
    total: int
    passed: int
    failed: int
    rows: tuple[FailureRow, ...]


def analyze_failures(snapshot: RunSnapshot, split_id: str, checkpoint_id: str, ruler: str) -> FailureReport:
    snapshot.split(split_id)
    checkpoint = snapshot.checkpoint(checkpoint_id)
    if ruler not in snapshot.tolerances:
        raise KeyError(f"unknown tolerance ruler: {ruler}")
    if split_id not in checkpoint.available_splits:
        raise ValueError(f"predictions unavailable: {checkpoint_id}/{split_id}")
    predictions = {row.row_id: row.values for row in snapshot.predictions[(checkpoint_id, split_id)]}
    tolerance = snapshot.tolerances[ruler]
    rows = []
    for sample in snapshot.samples[split_id]:
        values = predictions[sample.row_id]
        signed = tuple(predicted - target for predicted, target in zip(values, sample.targets, strict=True))
        absolute = tuple(abs(value) for value in signed)
        passed = all(error <= limit for error, limit in zip(absolute, tolerance, strict=True))
        severity = max(error / limit for error, limit in zip(absolute, tolerance, strict=True))
        rows.append(FailureRow(sample.row_id, sample.motion_group, sample.inputs, sample.targets, values, signed, absolute, passed, severity))
    passed_count = sum(row.passed for row in rows)
    return FailureReport(split_id, checkpoint_id, ruler, len(rows), passed_count, len(rows) - passed_count, tuple(rows))
