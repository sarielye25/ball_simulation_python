"""Run preflight checks with `python checks.py` before starting fresh training.

Only train.csv and validation.csv are opened. The temporary learning model and
Adam state are discarded; a successful small fit is not generalization evidence.
"""

import copy
import csv
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

import config
import data
from metrics import calculate_pass_masks
from neural_network import transmodel
from stopping import StopReason, StopState, check_stop


LABELS_DIR = Path(__file__).resolve().parent.parent / "labels"
SMALL_FIT_ROWS = 64
SMALL_FIT_UPDATES = 500
MAX_FINAL_LOSS_RATIO = 0.2


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_data(labels_dir=LABELS_DIR):
    """Validate train/validation data and independent analytical motion cases."""
    random_state = random.getstate()
    try:
        import physics_formula as physics
    finally:
        random.setstate(random_state)

    splits = {}
    conditions = set()
    for split, expected in (
        ("train", config.expected_train_rows),
        ("validation", config.expected_validation_rows),
    ):
        inputs, targets = data.load_split(Path(labels_dir) / f"{split}.csv", expected)
        inputs, targets = np.asarray(inputs), np.asarray(targets)
        _require(inputs.shape == (expected, 4), f"{split}: invalid input shape")
        _require(targets.shape == (expected, 2), f"{split}: invalid target shape")
        groups = data.assign_groups(inputs, targets, physics.m, physics.mu_s, physics.g)
        allowed = {data.ALWAYS_RESTING, data.MOVED_THEN_STOPPED,
                   data.MOVING_AT_OBSERVATION}
        _require(len(groups) == expected and set(groups) <= allowed,
                 f"{split}: invalid motion groups")
        for row_number, (input_row, target_row) in enumerate(zip(inputs, targets), 2):
            initial_velocity, force, duration, observation = input_row
            condition = tuple(input_row[:3])
            _require(condition not in conditions,
                     f"{split} row {row_number}: repeated push condition")
            conditions.add(condition)
            velocity, displacement = physics.motion_cal(
                force, physics.m, duration, observation, initial_velocity
            )
            np.testing.assert_allclose(
                target_row, [displacement, velocity], rtol=1e-10, atol=1e-10,
                err_msg=f"{split} row {row_number}: reference target mismatch",
            )
        splits[split] = (inputs, targets)

    friction = physics.mu_k * physics.g
    threshold = physics.mu_s * physics.m * physics.g
    reverse_force = 10.0 * physics.m
    stop_time = 1.0 / (10.0 + friction)
    remaining = 1.0 - stop_time
    cases = [
        ([0, 0, 1, 1], [0, 0], data.ALWAYS_RESTING),
        ([0, threshold, 1, 1], [0, 0], data.ALWAYS_RESTING),
        ([0, -threshold, 1, 1], [0, 0], data.ALWAYS_RESTING),
        ([10, 0, 1, 1], [10 - friction / 2, 10 - friction],
         data.MOVING_AT_OBSERVATION),
        ([1, 0, 1, 1], [1 / (2 * friction), 0], data.MOVED_THEN_STOPPED),
        ([1, -reverse_force, 1, 1],
         [stop_time / 2 - (10 - friction) * remaining**2 / 2,
          -(10 - friction) * remaining], data.MOVING_AT_OBSERVATION),
        ([0, threshold + 0.1, 1, 1],
         [((threshold + 0.1) / physics.m - friction) / 2,
          (threshold + 0.1) / physics.m - friction], data.MOVING_AT_OBSERVATION),
    ]
    for inputs, expected, group in cases:
        velocity, displacement = physics.motion_cal(
            inputs[1], physics.m, inputs[2], inputs[3], inputs[0]
        )
        np.testing.assert_allclose([displacement, velocity], expected, atol=1e-12)
        _require(data.assign_groups([inputs], [expected], physics.m,
                                   physics.mu_s, physics.g) == [group],
                 f"Incorrect group for reference case {inputs}")
    return splits


def check_scaling(splits):
    """Fit only training rows and verify fixed-statistic transforms and recovery."""
    normalization = data.fit_scaler(*splits["train"])
    original = copy.deepcopy(normalization)
    for column, name, transform, inverse in (
        (0, "input", data.standardize_inputs, data.inverse_standardize_inputs),
        (1, "target", data.standardize_targets, data.inverse_standardize_targets),
    ):
        training = splits["train"][column]
        expected_scale = training.std(axis=0)
        expected_scale = np.where(expected_scale < config.scale_epsilon, 1, expected_scale)
        mean, scale = normalization[f"{name}_mean"], normalization[f"{name}_scale"]
        np.testing.assert_allclose(mean, training.mean(axis=0))
        np.testing.assert_allclose(scale, expected_scale)
        for split in ("train", "validation"):
            values = splits[split][column]
            standardized = transform(values, mean, scale)
            _require(np.isfinite(standardized).all(), f"{split}: non-finite scaling")
            np.testing.assert_allclose(inverse(standardized, mean, scale), values,
                                       rtol=1e-12, atol=1e-12)
    _require(normalization == original, "Transform changed training statistics")
    tiny = config.scale_epsilon / 10
    constant = data.fit_scaler([[1, 2, 3, 0], [1, 2, 3, tiny]], [[0, 1], [0, 1]])
    np.testing.assert_array_equal(constant["input_scale"], np.ones(4))
    np.testing.assert_array_equal(constant["target_scale"], np.ones(2))
    return normalization


def check_learning(splits, normalization):
    """Check one Adam update and a fixed 64-row fit using an isolated CPU model.

    The diagnostic requires at least 80% MSE reduction after 500 updates.
    These are preflight choices, not the real run's stopping or accuracy rules.
    """
    _require(config.optimizer_name.lower() == "adam", "Preflight expects Adam")
    _require(config.device == "cpu" and config.dtype == "float32",
             "Preflight expects the CPU float32 protocol")
    _require(config.loss_name == "mse" and config.loss_reduction == "mean",
             "Preflight expects mean MSE")
    inputs, targets = splits["train"]
    _require(len(inputs) >= SMALL_FIT_ROWS, "Not enough training rows for small fit")
    indices = np.random.default_rng(config.seed).choice(len(inputs), SMALL_FIT_ROWS, replace=False)
    inputs = torch.tensor(data.standardize_inputs(
        inputs[indices], normalization["input_mean"], normalization["input_scale"]
    ), dtype=torch.float32, device="cpu")
    targets = torch.tensor(data.standardize_targets(
        targets[indices], normalization["target_mean"], normalization["target_scale"]
    ), dtype=torch.float32, device="cpu")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        model = transmodel().to(device="cpu", dtype=torch.float32)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
        loss_function = torch.nn.MSELoss(reduction=config.loss_reduction)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        initial_loss = None
        for update in range(SMALL_FIT_UPDATES):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            _require(prediction.shape == targets.shape, "Invalid prediction shape")
            _require(torch.isfinite(prediction).all().item(), "Non-finite prediction")
            loss = loss_function(prediction, targets)
            _require(torch.isfinite(loss).item(), "Non-finite loss")
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            for name, parameter in model.named_parameters():
                _require(parameter.grad is not None, f"Missing gradient: {name}")
                _require(torch.isfinite(parameter.grad).all().item(),
                         f"Non-finite gradient: {name}")
            optimizer.step()
            _require(all(torch.isfinite(parameter).all().item()
                         for parameter in model.parameters()), "Non-finite parameter")
            if update == 0:
                _require(any(not torch.equal(old, new)
                             for old, new in zip(before, model.parameters())),
                         "One optimizer step did not change any parameters")
        with torch.inference_mode():
            final_loss = loss_function(model(inputs), targets).item()
        _require(np.isfinite(final_loss) and initial_loss > 0
                 and final_loss <= initial_loss * MAX_FINAL_LOSS_RATIO,
                 f"Small fit failed: MSE {initial_loss:.6g} -> {final_loss:.6g}")
    return {"initial_mse": initial_loss, "final_mse": final_loss,
            "rows": SMALL_FIT_ROWS, "updates": SMALL_FIT_UPDATES}


class PreflightChecks(unittest.TestCase):
    """Repeatable checks; failures produce a nonzero command exit status."""

    @classmethod
    def setUpClass(cls):
        cls.splits = check_data()

    def test_scaling(self):
        check_scaling(self.splits)

    def test_learning(self):
        result = check_learning(self.splits, check_scaling(self.splits))
        print(f"Small fit: {result['initial_mse']:.6g} -> {result['final_mse']:.6g} MSE")

    def test_invalid_csv(self):
        header = config.input_columns + config.target_columns
        valid = [0, 0, 0.5, 1, 0, 0]
        cases = [([], []), (header[:-1], [valid[:-1]]), (header, []),
                 (header, [["bad", *valid[1:]]]),
                 (header, [[float("nan"), *valid[1:]]]),
                 (header, [[float("inf"), *valid[1:]]]),
                 (header, [[0, 0, -1, 1, 0, 0]]),
                 (header, [[0, 0, 2, 1, 0, 0]]),
                 (header, [valid[:-1]])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            for columns, rows in cases:
                with self.subTest(columns=columns, rows=rows):
                    with path.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(columns)
                        writer.writerows(rows)
                    with self.assertRaises(ValueError):
                        data.load_split(path)
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerow(valid)
            with self.assertRaises(ValueError):
                data.load_split(path, expected_rows=2)

    def test_strict_tolerance(self):
        for ruler, (displacement, velocity) in config.tolerances.items():
            errors = [[0, 0], [displacement, 0], [0, velocity],
                      [displacement / 2, velocity / 2]]
            np.testing.assert_array_equal(
                calculate_pass_masks(errors)[ruler], [True, False, False, True]
            )

    def test_stopping(self):
        self.assertEqual(check_stop(StopState(), 0, .95, .95, 1).reason,
                         StopReason.TARGET_REACHED)
        fixed = StopState()
        self.assertEqual(check_stop(fixed, 0, 1, 1, 1, fixed_budget=True,
                                    max_updates=700).reason, StopReason.CONTINUE)
        self.assertEqual(check_stop(fixed, 700, 1, 1, 1, fixed_budget=True,
                                    max_updates=700).reason, StopReason.MAX_UPDATES)
        fixed = StopState()
        check_stop(fixed, 0, 0, 0, 1, fixed_budget=True, max_updates=700)
        self.assertEqual(check_stop(fixed, 500, 0, 0, 1, fixed_budget=True,
                                    max_updates=700).reason, StopReason.CONTINUE)
        self.assertEqual(check_stop(fixed, 700, 0, 0, 1, fixed_budget=True,
                                    max_updates=700).reason, StopReason.MAX_UPDATES)
        state = StopState()
        check_stop(state, 0, 0, 0, 1, min_delta=.1)
        self.assertFalse(check_stop(state, 100, 0, 0, .95, min_delta=.1).meaningful_improvement)
        self.assertTrue(check_stop(state, 200, 0, 0, .85, min_delta=.1).meaningful_improvement)
        self.assertFalse(check_stop(state, 600, 0, 0, .85, min_delta=.1).should_stop)
        self.assertEqual(check_stop(state, 700, 0, 0, .85, min_delta=.1).reason,
                         StopReason.NO_PROGRESS)
        state = StopState()
        check_stop(state, 0, 0, 0, 1, max_updates=500)
        decision = check_stop(state, 500, 0, 0, 1, max_updates=500)
        self.assertTrue(decision.patience_reached and decision.max_updates_reached)
        self.assertEqual(decision.selection, "best_mse")
        self.assertEqual(check_stop(StopState(), 0, 0, 0, float("nan")).reason,
                         StopReason.NUMERICAL_FAILURE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
