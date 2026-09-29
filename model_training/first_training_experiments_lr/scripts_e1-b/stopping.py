"""Decide when to stop using full-training and full-validation metrics."""

from dataclasses import dataclass
from enum import Enum
import math

from config import (
    max_updates as default_max_updates,
    min_delta as default_min_delta,
    patience_updates as default_patience_updates,
    target_pass_rate as default_target_pass_rate,
)


class StopReason(str, Enum):
    """Possible outcomes of one stopping check."""

    CONTINUE = "CONTINUE"
    TARGET_REACHED = "TARGET_REACHED"
    NO_PROGRESS = "NO_PROGRESS"
    MAX_UPDATES = "MAX_UPDATES"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


@dataclass
class StopState:
    """Progress memory, serializable with dataclasses.asdict for checkpoints."""

    progress_reference_mse: float | None = None
    progress_reference_update: int | None = None
    last_checked_update: int | None = None


@dataclass(frozen=True)
class StopDecision:
    """Result for the training loop; selection is current, best_mse, or None."""

    should_stop: bool
    reason: StopReason
    selection: str | None = None
    meaningful_improvement: bool = False
    patience_reached: bool = False
    max_updates_reached: bool = False


def check_stop(
    state,
    update,
    train_pass_rate,
    validation_pass_rate,
    validation_mse,
    *,
    target_pass_rate=default_target_pass_rate,
    min_delta=default_min_delta,
    patience_updates=default_patience_updates,
    max_updates=default_max_updates,
    fixed_budget=False,
):
    """Update progress memory in place and return an ordered stopping decision.

    Supply full-set rates under the same selected ruler and validation MSE
    from the same model state. Initialize at update zero, then call once per
    evaluation with increasing update counts. Never use reserved test metrics.
    The caller saves checkpoints and runs the scheduler only when continuing.
    Non-finite metrics abort without changing progress memory; malformed inputs
    raise ValueError. A numerical failure does not select a final model.
    """
    if type(update) is not int or update < 0:
        raise ValueError("update must be a nonnegative integer.")
    if type(patience_updates) is not int or patience_updates <= 0:
        raise ValueError("patience_updates must be a positive integer.")
    if type(max_updates) is not int or max_updates <= 0:
        raise ValueError("max_updates must be a positive integer.")
    if type(fixed_budget) is not bool:
        raise ValueError("fixed_budget must be a boolean.")
    if not math.isfinite(target_pass_rate) or not 0 <= target_pass_rate <= 1:
        raise ValueError("target_pass_rate must be finite and in [0, 1].")
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and nonnegative.")
    if update > max_updates:
        raise ValueError("Training has exceeded the maximum update budget.")

    uninitialized = state.progress_reference_mse is None
    if uninitialized:
        if state.progress_reference_update is not None or state.last_checked_update is not None:
            raise ValueError("An uninitialized state must have all fields set to None.")
        if update != 0:
            raise ValueError("The first evaluation must be at update zero.")
    else:
        if (not math.isfinite(state.progress_reference_mse)
                or state.progress_reference_mse < 0
                or type(state.progress_reference_update) is not int
                or type(state.last_checked_update) is not int
                or not 0 <= state.progress_reference_update <= state.last_checked_update):
            raise ValueError("Invalid saved stopping state.")
        if update <= state.last_checked_update:
            raise ValueError("Evaluation update counts must increase.")

    cap_reached = update == max_updates
    if not all(math.isfinite(value) for value in (
            train_pass_rate, validation_pass_rate, validation_mse)):
        return StopDecision(True, StopReason.NUMERICAL_FAILURE,
                            max_updates_reached=cap_reached)
    if not 0 <= train_pass_rate <= 1 or not 0 <= validation_pass_rate <= 1:
        raise ValueError("Pass rates must be in [0, 1], not percentages.")
    if validation_mse < 0:
        raise ValueError("Validation MSE must be nonnegative.")

    if uninitialized:
        state.progress_reference_mse = float(validation_mse)
        state.progress_reference_update = update
    state.last_checked_update = update

    if (not fixed_budget and train_pass_rate >= target_pass_rate
            and validation_pass_rate >= target_pass_rate):
        return StopDecision(True, StopReason.TARGET_REACHED, selection="current",
                            max_updates_reached=cap_reached)

    improved = (not uninitialized
                and validation_mse < state.progress_reference_mse - min_delta)
    if improved:
        state.progress_reference_mse = float(validation_mse)
        state.progress_reference_update = update

    patience_reached = (not fixed_budget
                        and update - state.progress_reference_update >= patience_updates)
    if patience_reached:
        reason = StopReason.NO_PROGRESS
    elif cap_reached:
        reason = StopReason.MAX_UPDATES
    else:
        reason = StopReason.CONTINUE
    should_stop = reason != StopReason.CONTINUE
    return StopDecision(
        should_stop, reason, selection="best_mse" if should_stop else None,
        meaningful_improvement=improved, patience_reached=patience_reached,
        max_updates_reached=cap_reached,
    )
