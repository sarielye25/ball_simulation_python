"""Prepare metric series without reading files or using Qt."""

from dataclasses import dataclass

from .model import RunSnapshot


@dataclass(frozen=True)
class Curve:
    split_id: str
    split_name: str
    metric: str
    predictor: str
    scope: str
    unit: str
    points: tuple[tuple[int, float], ...]


def available_metrics(snapshot: RunSnapshot, split_id: str) -> tuple[str, ...]:
    snapshot.split(split_id)
    return tuple(dict.fromkeys(metric.name for metric in snapshot.metrics if metric.split_id == split_id))


def prepare_curve(snapshot: RunSnapshot, split_id: str, metric_name: str, *, predictor="model", scope="overall") -> Curve:
    split = snapshot.split(split_id)
    records = [record for record in snapshot.metrics if record.split_id == split_id and record.name == metric_name and record.predictor == predictor and record.scope == scope and record.value is not None]
    records.sort(key=lambda record: record.update)
    unit = records[0].unit if records else ""
    return Curve(split_id, split.name, metric_name, predictor, scope, unit, tuple((record.update, record.value) for record in records))
