"""Load label data, assign motion groups, and normalize numerical values."""

import csv
import math
from pathlib import Path
import numpy as np
import torch

from config import input_columns, target_columns

ALWAYS_RESTING = "always_resting"
MOVED_THEN_STOPPED = "moved_then_stopped"
MOVING_AT_OBSERVATION = "moving_at_observation"

def load_split(csv_path, expected_rows=None):
    """Load and validate one CSV split, then return its inputs and targets."""

    inputs = []
    targets = []
    required_columns = input_columns + target_columns
    path = Path(csv_path)

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header.")

        missing_columns = [
            column for column in required_columns if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(f"{path} is missing columns: {missing_columns}")

        for csv_row_number, row in enumerate(reader, start=2):
            try:
                input_row = [float(row[column]) for column in input_columns]
                target_row = [float(row[column]) for column in target_columns]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}, row {csv_row_number}, contains a non-numeric value."
                ) from error

            all_values = input_row + target_row
            if not all(math.isfinite(value) for value in all_values):
                raise ValueError(
                    f"{path}, row {csv_row_number}, contains a non-finite value."
                )

            force_duration = input_row[2]
            observation_time = input_row[3]
            if not 0 <= force_duration <= observation_time:
                raise ValueError(
                    f"{path}, row {csv_row_number}, must satisfy 0 <= t1 <= t."
                )

            if len(input_row) != 4:
                raise ValueError(
                    f"{path}, row {csv_row_number}, does not contain 4 inputs."
                )
            if len(target_row) != 2:
                raise ValueError(
                    f"{path}, row {csv_row_number}, does not contain 2 targets."
                )

            inputs.append(input_row)
            targets.append(target_row)

    if not inputs:
        raise ValueError(f"{path} does not contain any data rows.")

    if len(inputs) != len(targets):
        raise ValueError(f"{path} has different numbers of input and target rows.")

    if expected_rows is not None and len(inputs) != expected_rows:
        raise ValueError(
            f"{path} contains {len(inputs)} rows; expected {expected_rows}."
        )

    return inputs, targets


def assign_groups(inputs, targets, mass_kg, static_friction, gravity):
    """Assign one reference motion group to every input-target pair."""

    if len(inputs) != len(targets):
        raise ValueError("Inputs and targets must contain the same number of rows.")

    groups = []
    static_friction_limit = static_friction * mass_kg * gravity

    for input_row, target_row in zip(inputs, targets):
        initial_velocity = input_row[0]
        force = input_row[1]
        final_velocity = target_row[1]

        always_resting = (
            initial_velocity == 0
            and abs(force) <= static_friction_limit
        )

        if always_resting:
            group = ALWAYS_RESTING
        elif final_velocity == 0:
            group = MOVED_THEN_STOPPED
        else:
            group = MOVING_AT_OBSERVATION

        groups.append(group)

    return groups

def _as_float_array(name, values, expected_columns=None):
    """Convert tensor-like values to a finite NumPy float array."""

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if expected_columns is not None and array.shape[1] != expected_columns:
        raise ValueError(
            f"{name} must have {expected_columns} columns; got {array.shape[1]}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value.")
    return array


def _as_scaler_array(name, values):
    """Validate one two-value target-scaler parameter."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"{name} must contain exactly two values.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value.")
    return array

def inverse_standardize_targets(standardized_values, target_mean, target_scale):
    """Convert standardized [displacement, velocity] values to physical units."""

    standardized = _as_float_array(
        "standardized_values", standardized_values, expected_columns=2
    )
    mean = _as_scaler_array("target_mean", target_mean)
    scale = _as_scaler_array("target_scale", target_scale)
    if np.any(scale <= 0):
        raise ValueError("target_scale values must be greater than zero.")
    return standardized * scale + mean

def standardize_targets(physical_values, target_mean, target_scale):
    """Convert physical [displacement, velocity] values to standardized units."""

    physical = _as_float_array("physical_values", physical_values, expected_columns=2)
    mean = _as_scaler_array("target_mean", target_mean)
    scale = _as_scaler_array("target_scale", target_scale)
    if np.any(scale <= 0):
        raise ValueError("target_scale values must be greater than zero.")
    return (physical - mean) / scale

if __name__ == "__main__":
    train_inputs, train_targets = load_split(
        "labels/train.csv",
        expected_rows=8000,
    )
    train_groups = assign_groups(
        train_inputs,
        train_targets,
        mass_kg=1.0,
        static_friction=0.4,
        gravity=9.81,
    )

    print(f"Loaded {len(train_inputs)} training rows.")
    print(f"First input row: {train_inputs[0]}")
    print(f"First target row: {train_targets[0]}")
    print(f"First reference group: {train_groups[0]}")
