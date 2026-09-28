"""Data shared between the loader, worker, and interface."""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

INPUT_COLUMNS = ("v0_m_s", "force_N", "force_duration_s", "observation_time_s")
TARGET_COLUMNS = ("displacement_m", "v_final_m_s")
GROUPS = ("always_resting", "moved_then_stopped", "moving_at_observation")
SCOPES = ("overall", *GROUPS, "breakaway")
RULERS = ("coarse", "intermediate", "fine")


class ViewerError(Exception):
    """An artifact cannot be trusted or displayed."""


class State(Enum):
    EMPTY = "EMPTY"
    LOADING = "LOADING"
    READY = "READY"
    INFERENCING = "INFERENCING"
    ERROR = "ERROR"
    CLOSING = "CLOSING"


@dataclass(frozen=True)
class SplitData:
    name: str
    row_ids: tuple[str, ...]
    groups: tuple[str, ...]
    inputs: np.ndarray
    targets: np.ndarray
    sha256: str
    source: Path


@dataclass(frozen=True)
class RunBundle:
    run_dir: Path
    manifest: dict
    evaluations: tuple[dict, ...]
    baselines: tuple[dict, ...]
    events: tuple[dict, ...]
    batches: tuple[dict, ...]
    termination: dict | None
    index: dict | None
    splits: dict[str, SplitData]
    split_errors: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    generation: int = 0
    demo: bool = False


@dataclass(frozen=True)
class PredictionResult:
    generation: int
    checkpoint_file: str
    update: int
    split: str
    row_ids: tuple[str, ...]
    inputs: np.ndarray
    targets: np.ndarray
    predictions: np.ndarray
    signed_errors: np.ndarray
    absolute_errors: np.ndarray
    groups: tuple[str, ...]
    breakaway_mask: np.ndarray
    pass_masks: dict[str, np.ndarray]
    summaries: dict[str, dict]
    verification: tuple[str, ...]

    @property
    def verified(self):
        return not self.verification
