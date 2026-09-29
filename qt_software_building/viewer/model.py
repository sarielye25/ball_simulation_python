"""Stable in-memory data passed from the reader to analysis code."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Group:
    id: str
    name: str
    description: str
    source: str


@dataclass(frozen=True)
class Split:
    id: str
    name: str
    role: str
    rows: int


@dataclass(frozen=True)
class Sample:
    row_id: str
    motion_group: str
    inputs: tuple[float, ...]
    targets: tuple[float, ...]


@dataclass(frozen=True)
class Prediction:
    row_id: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class Metric:
    update: int
    split_id: str
    predictor: str
    scope: str
    name: str
    value: float | None
    count: int
    unit: str


@dataclass(frozen=True)
class Checkpoint:
    id: str
    update: int
    available_splits: tuple[str, ...]


@dataclass(frozen=True)
class RunSnapshot:
    group: Group
    run_id: str
    generation: int
    status: str
    purpose: str
    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    units: dict[str, str]
    tolerances: dict[str, tuple[float, ...]]
    stopping_ruler: str
    splits: tuple[Split, ...]
    samples: dict[str, tuple[Sample, ...]]
    metrics: tuple[Metric, ...]
    checkpoints: tuple[Checkpoint, ...]
    predictions: dict[tuple[str, str], tuple[Prediction, ...]]
    selected_checkpoint: str | None
    termination: dict | None

    def split(self, split_id: str) -> Split:
        return next(split for split in self.splits if split.id == split_id)

    def checkpoint(self, checkpoint_id: str) -> Checkpoint:
        return next(checkpoint for checkpoint in self.checkpoints if checkpoint.id == checkpoint_id)
