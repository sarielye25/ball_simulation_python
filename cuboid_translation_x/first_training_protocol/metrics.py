"""Evaluate model predictions without updating the model."""

import numpy as np
import torch

from config import batch_size as default_batch_size
from config import breakaway_band_N, tolerances
from data import (
    ALWAYS_RESTING,
    MOVED_THEN_STOPPED,
    MOVING_AT_OBSERVATION,
    _as_float_array,
    inverse_standardize_targets,
    standardize_targets,
)


GROUP_NAMES = (
    ALWAYS_RESTING,
    MOVED_THEN_STOPPED,
    MOVING_AT_OBSERVATION,
)

def calculate_errors(
    standardized_predictions,
    standardized_targets,
    target_mean,
    target_scale,
):
    """Calculate standardized squared errors and physical-unit errors per row."""

    predictions = _as_float_array(
        "standardized_predictions", standardized_predictions, expected_columns=2
    )
    targets = _as_float_array(
        "standardized_targets", standardized_targets, expected_columns=2
    )
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")

    physical_predictions = inverse_standardize_targets(
        predictions, target_mean, target_scale
    )
    physical_targets = inverse_standardize_targets(targets, target_mean, target_scale)
    physical_signed_errors = physical_predictions - physical_targets

    return {
        "standardized_squared": np.square(predictions - targets),
        "physical_signed": physical_signed_errors,
        "physical_absolute": np.abs(physical_signed_errors),
        "physical_predictions": physical_predictions,
        "physical_targets": physical_targets,
    }

def calculate_pass_masks(physical_absolute_errors, ruler_tolerances=tolerances):
    """Return the rows passing both displacement and velocity tolerances."""

    errors = _as_float_array(
        "physical_absolute_errors", physical_absolute_errors, expected_columns=2
    )
    pass_masks = {}
    for ruler_name, (displacement_tolerance, velocity_tolerance) in (
        ruler_tolerances.items()
    ):
        if displacement_tolerance <= 0 or velocity_tolerance <= 0:
            raise ValueError(f"Tolerances for {ruler_name!r} must be positive.")
        pass_masks[ruler_name] = (
            (errors[:, 0] < displacement_tolerance)
            & (errors[:, 1] < velocity_tolerance)
        )
    return pass_masks


def summarize_metrics(
    standardized_squared_errors,
    physical_absolute_errors,
    pass_masks,
    selection_mask=None,
):
    """Summarize errors for all rows or for rows selected by a Boolean mask."""

    squared_errors = _as_float_array(
        "standardized_squared_errors",
        standardized_squared_errors,
        expected_columns=2,
    )
    absolute_errors = _as_float_array(
        "physical_absolute_errors", physical_absolute_errors, expected_columns=2
    )
    if squared_errors.shape != absolute_errors.shape:
        raise ValueError("Standardized and physical errors must have the same shape.")

    row_count = squared_errors.shape[0]
    if selection_mask is None:
        selected = np.ones(row_count, dtype=bool)
    else:
        selected = np.asarray(selection_mask, dtype=bool)
        if selected.shape != (row_count,):
            raise ValueError("selection_mask must contain one value per row.")

    count = int(selected.sum())
    if count == 0:
        return {
            "count": 0,
            "standardized_mse": None,
            "displacement_standardized_mse": None,
            "velocity_standardized_mse": None,
            "displacement_loss_contribution": None,
            "velocity_loss_contribution": None,
            "displacement_mae_m": None,
            "displacement_p95_m": None,
            "displacement_max_m": None,
            "velocity_mae_m_s": None,
            "velocity_p95_m_s": None,
            "velocity_max_m_s": None,
            "pass_rates": {name: None for name in pass_masks},
        }

    selected_absolute = absolute_errors[selected]
    selected_pass_rates = {}
    for ruler_name, pass_mask in pass_masks.items():
        pass_array = np.asarray(pass_mask, dtype=bool)
        if pass_array.shape != (row_count,):
            raise ValueError(f"Pass mask {ruler_name!r} must contain one value per row.")
        selected_pass_rates[ruler_name] = float(pass_array[selected].mean())

    return {
        "count": count,
        "standardized_mse": float(squared_errors[selected].sum() / (2 * count)),
        "displacement_standardized_mse": float(squared_errors[selected, 0].mean()),
        "velocity_standardized_mse": float(squared_errors[selected, 1].mean()),
        "displacement_loss_contribution": float(squared_errors[selected, 0].mean() / 2),
        "velocity_loss_contribution": float(squared_errors[selected, 1].mean() / 2),
        "displacement_mae_m": float(selected_absolute[:, 0].mean()),
        "displacement_p95_m": float(np.percentile(selected_absolute[:, 0], 95)),
        "displacement_max_m": float(selected_absolute[:, 0].max()),
        "velocity_mae_m_s": float(selected_absolute[:, 1].mean()),
        "velocity_p95_m_s": float(np.percentile(selected_absolute[:, 1], 95)),
        "velocity_max_m_s": float(selected_absolute[:, 1].max()),
        "pass_rates": selected_pass_rates,
    }


def group_metrics(standardized_squared_errors, physical_absolute_errors, pass_masks, groups):
    """Calculate the same metrics separately for each reference motion group."""

    group_array = np.asarray(groups, dtype=object)
    row_count = np.asarray(standardized_squared_errors).shape[0]
    if group_array.shape != (row_count,):
        raise ValueError("groups must contain one group name per row.")

    unknown_groups = set(group_array) - set(GROUP_NAMES)
    if unknown_groups:
        raise ValueError(f"Unknown reference groups: {sorted(unknown_groups)}")

    return {
        group_name: summarize_metrics(
            standardized_squared_errors,
            physical_absolute_errors,
            pass_masks,
            group_array == group_name,
        )
        for group_name in GROUP_NAMES
    }


def make_breakaway_selection_mask(
    physical_inputs,
    static_friction_limit=3.924,
    band_width=breakaway_band_N,
):
    """Select rows near the static-friction breakaway-force boundary."""

    inputs = _as_float_array("physical_inputs", physical_inputs, expected_columns=4)
    initial_velocity = inputs[:, 0]
    force = inputs[:, 1]
    return (initial_velocity == 0) & (
        np.abs(np.abs(force) - static_friction_limit) <= band_width
    )


def summarize_predictions(
    standardized_predictions,
    standardized_targets,
    target_mean,
    target_scale,
    groups=None,
    physical_inputs=None,
    include_per_row=False,
    ruler_tolerances=tolerances,
):
    """Build overall, group, and optional boundary summaries from predictions."""

    errors = calculate_errors(
        standardized_predictions,
        standardized_targets,
        target_mean,
        target_scale,
    )
    pass_masks = calculate_pass_masks(
        errors["physical_absolute"], ruler_tolerances
    )
    result = {
        "overall": summarize_metrics(
            errors["standardized_squared"],
            errors["physical_absolute"],
            pass_masks,
        )
    }

    if groups is not None:
        result["groups"] = group_metrics(
            errors["standardized_squared"],
            errors["physical_absolute"],
            pass_masks,
            groups,
        )

    if physical_inputs is not None:
        boundary = make_breakaway_selection_mask(physical_inputs)
        result["boundary"] = summarize_metrics(
            errors["standardized_squared"],
            errors["physical_absolute"],
            pass_masks,
            boundary,
        )

    if include_per_row:
        result["per_row"] = {
            **errors,
            "pass_masks": pass_masks,
        }

    return result


def predict_full_split(model, standardized_inputs, evaluation_batch_size=default_batch_size):
    """Predict every row of one already-defined dataset split in batches."""

    inputs = _as_float_array(
        "standardized_inputs", standardized_inputs, expected_columns=4
    )
    if evaluation_batch_size <= 0:
        raise ValueError("evaluation_batch_size must be positive.")

    try:
        first_parameter = next(model.parameters())
        model_device = first_parameter.device
        model_dtype = first_parameter.dtype
    except StopIteration:
        model_device = torch.device("cpu")
        model_dtype = torch.float32

    was_training = model.training
    model.eval()
    prediction_batches = []
    try:
        with torch.inference_mode():
            for start in range(0, len(inputs), evaluation_batch_size):
                input_batch = torch.as_tensor(
                    inputs[start : start + evaluation_batch_size],
                    dtype=model_dtype,
                    device=model_device,
                )
                prediction_batch = model(input_batch)
                if prediction_batch.ndim != 2 or prediction_batch.shape != (
                    len(input_batch),
                    2,
                ):
                    raise ValueError("The model must return predictions with shape [N, 2].")
                if not torch.isfinite(prediction_batch).all():
                    raise ValueError("The model returned a non-finite prediction.")
                prediction_batches.append(prediction_batch.cpu().numpy())
    finally:
        model.train(was_training)

    return np.concatenate(prediction_batches, axis=0).astype(np.float64, copy=False)


def evaluate(
    model,
    standardized_inputs,
    standardized_targets,
    target_mean,
    target_scale,
    groups=None,
    physical_inputs=None,
    include_per_row=False,
    evaluation_batch_size=default_batch_size,
    ruler_tolerances=tolerances,
):
    """Predict and measure every row in one training or validation split."""

    predictions = predict_full_split(
        model, standardized_inputs, evaluation_batch_size=evaluation_batch_size
    )
    return summarize_predictions(
        predictions,
        standardized_targets,
        target_mean,
        target_scale,
        groups=groups,
        physical_inputs=physical_inputs,
        include_per_row=include_per_row,
        ruler_tolerances=ruler_tolerances,
    )


def baseline_metrics(
    physical_inputs,
    physical_targets,
    target_mean,
    target_scale,
    groups=None,
    ruler_tolerances=tolerances,
):
    """Measure the zero and constant-velocity baseline predictions."""

    inputs = _as_float_array("physical_inputs", physical_inputs, expected_columns=4)
    targets = _as_float_array("physical_targets", physical_targets, expected_columns=2)
    if len(inputs) != len(targets):
        raise ValueError("Physical inputs and targets must have the same number of rows.")

    zero_predictions = np.zeros_like(targets)
    constant_velocity_predictions = np.column_stack(
        (inputs[:, 0] * inputs[:, 3], inputs[:, 0])
    )
    standardized_targets = standardize_targets(targets, target_mean, target_scale)

    baseline_predictions = {
        "zero": zero_predictions,
        "constant_velocity": constant_velocity_predictions,
    }
    results = {}
    for baseline_name, physical_predictions in baseline_predictions.items():
        results[baseline_name] = summarize_predictions(
            standardize_targets(physical_predictions, target_mean, target_scale),
            standardized_targets,
            target_mean,
            target_scale,
            groups=groups,
            physical_inputs=inputs,
            ruler_tolerances=ruler_tolerances,
        )
    return results
