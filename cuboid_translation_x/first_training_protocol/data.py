"""Load label data, assign motion groups, and normalize numerical values."""

import csv
import math
from pathlib import Path

from config import (
    input_columns,
    near_zero_scale_policy,
    scale_epsilon,
    target_columns,
)


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


def fit_scaler(rows):
    """Calculate one mean and standard deviation per training-data column."""

    if not rows:
        raise ValueError("Cannot fit a scaler to empty data.")

    column_count = len(rows[0])
    if column_count == 0:
        raise ValueError("Cannot fit a scaler to rows without columns.")
    if any(len(row) != column_count for row in rows):
        raise ValueError("All rows must contain the same number of columns.")

    row_count = len(rows)
    means = [
        math.fsum(row[column] for row in rows) / row_count
        for column in range(column_count)
    ]
    standard_deviations = []

    for column, mean in enumerate(means):
        variance = (
            math.fsum((row[column] - mean) ** 2 for row in rows)
            / row_count
        )
        standard_deviation = math.sqrt(variance)

        if standard_deviation < scale_epsilon:
            if near_zero_scale_policy == "replace_with_one":
                standard_deviation = 1.0
            else:
                raise ValueError(
                    "A column has a near-zero standard deviation, but the "
                    f"policy {near_zero_scale_policy!r} is not supported."
                )

        standard_deviations.append(standard_deviation)

    return means, standard_deviations


def transform(rows, means, standard_deviations):
    """Standardize rows using statistics fitted on the training split."""

    _check_scaler_dimensions(rows, means, standard_deviations)
    return [
        [
            (value - means[column]) / standard_deviations[column]
            for column, value in enumerate(row)
        ]
        for row in rows
    ]


def inverse_transform(rows, means, standard_deviations):
    """Convert standardized rows back to their original physical values."""

    _check_scaler_dimensions(rows, means, standard_deviations)
    return [
        [
            value * standard_deviations[column] + means[column]
            for column, value in enumerate(row)
        ]
        for row in rows
    ]


def _check_scaler_dimensions(rows, means, standard_deviations):
    """Check that data rows and scaler statistics have matching widths."""

    if len(means) != len(standard_deviations):
        raise ValueError("Means and standard deviations must have equal lengths.")
    if any(len(row) != len(means) for row in rows):
        raise ValueError("Each row must match the number of scaler columns.")
    if any(scale <= 0 or not math.isfinite(scale) for scale in standard_deviations):
        raise ValueError("Every standard deviation must be positive and finite.")


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
    input_means, input_standard_deviations = fit_scaler(train_inputs)
    target_means, target_standard_deviations = fit_scaler(train_targets)
    standardized_inputs = transform(
        train_inputs,
        input_means,
        input_standard_deviations,
    )
    standardized_targets = transform(
        train_targets,
        target_means,
        target_standard_deviations,
    )

    print(f"Loaded {len(train_inputs)} training rows.")
    print(f"First input row: {train_inputs[0]}")
    print(f"First target row: {train_targets[0]}")
    print(f"First reference group: {train_groups[0]}")
    print(f"First standardized input row: {standardized_inputs[0]}")
    print(f"First standardized target row: {standardized_targets[0]}")
