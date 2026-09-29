"""Train the first model; run with the project environment's Python interpreter."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

import config
import data
from checkpoints import load_checkpoint, save_checkpoint, select_checkpoint
from metrics import baseline_metrics, evaluate
from neural_network import transmodel
from stopping import StopState, check_stop
from training_records import (
    append_csv, create_run_directory, save_datasets, save_evaluation,
    save_training_baselines, write_json,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared_publication import initial_manifest, publish, save_predictions

def prepare_data():
    labels_directory = Path(__file__).resolve().parent.parent / "labels"

    train_inputs, train_targets = data.load_split(
        labels_directory / "train.csv",
        expected_rows=config.expected_train_rows,
    )
    validation_inputs, validation_targets = data.load_split(
        labels_directory / "validation.csv",
        expected_rows=config.expected_validation_rows,
    )

    normalization = data.fit_scaler(train_inputs, train_targets)

    splits = {}
    for name, inputs, targets in (
        ("train", train_inputs, train_targets),
        ("validation", validation_inputs, validation_targets),
        ):
        splits[name] = {
            "row_ids": [f"{name}:{row}" for row in range(2, len(inputs) + 2)],
            "physical_inputs": inputs,
            "physical_targets": targets,
            "sha256": hashlib.sha256(
                (labels_directory / f"{name}.csv").read_bytes()
            ).hexdigest(),
            "inputs": data.standardize_inputs(
                inputs,
                normalization["input_mean"],
                normalization["input_scale"],
            ),
            "targets": data.standardize_targets(
                targets,
                normalization["target_mean"],
                normalization["target_scale"],
            ),
            "groups": data.assign_groups(
                inputs,
                targets,
                mass_kg=1.0,
                static_friction=0.4,
                gravity=9.81,
            ),
        }

    return splits, normalization


def train_one_batch(model, optimizer, loss_function, inputs, targets):
    """Compute differentiable batch MSE and perform exactly one Adam update."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    predictions = model(inputs)
    if predictions.shape != targets.shape or not torch.isfinite(predictions).all():
        raise FloatingPointError("Invalid training predictions.")
    loss = loss_function(predictions, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss.")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Missing or non-finite gradient: {name}")
    optimizer.step()
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Non-finite parameter after Adam step: {name}")
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise FloatingPointError("Non-finite Adam state.")
    return loss.item()


def evaluate_splits(model, splits, normalization, update):
    """Evaluate both full splits without updating weights or selecting on test data."""
    results = {}
    for name, split in splits.items():
        results[name] = evaluate(
            model, split["inputs"], split["targets"],
            normalization["target_mean"], normalization["target_scale"],
            groups=split["groups"], physical_inputs=split["physical_inputs"],
            include_per_row=update in config.per_row_error_updates,
            ruler_tolerances=config.tolerances,
        )
        summary = results[name]["overall"]
        for value in summary.values():
            numbers = value.values() if isinstance(value, dict) else [value]
            if any(not math.isfinite(number) for number in numbers):
                raise FloatingPointError(f"Non-finite {name} evaluation.")
    return results


def verify_reload(path, model, inputs):
    """Confirm saved weights reproduce predictions without consuming run RNG."""
    with torch.random.fork_rng(devices=[]):
        restored = transmodel().to(device="cpu", dtype=torch.float32)
        load_checkpoint(path, restored)
        with torch.inference_mode():
            torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)


def run_training(*, max_updates=None, run_directory=None, seed=None):
    """Run fresh training. A smaller explicit update cap is for smoke checks.

    Checkpoints contain the permutation, next row position, sampler RNG and stop
    state needed for continuation. This entry point starts fresh, not from a
    checkpoint. A failed run retains last-valid evidence and raises its error.
    """
    update_limit = config.max_updates if max_updates is None else max_updates
    if type(update_limit) is not int or not 0 < update_limit <= config.max_updates:
        raise ValueError("max_updates must be positive and within the configured budget.")
    if (config.optimizer_name.lower() != "adam" or config.loss_name != "mse"
            or config.loss_reduction != "mean" or config.device != "cpu"
            or config.dtype != "float32" or config.drop_last or not config.shuffle):
        raise ValueError("Expected Adam, mean MSE, CPU float32, shuffle, and no dropped rows.")
    if config.stopping_ruler not in config.tolerances:
        raise ValueError("Unknown stopping ruler.")
    if config.eval_interval <= 0 or config.batch_size <= 0:
        raise ValueError("Evaluation interval and batch size must be positive.")
    run_seed = config.seed if seed is None else seed
    if type(run_seed) is not int or run_seed < 0 or run_seed >= 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")

    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    splits, normalization = prepare_data()
    train_inputs = torch.tensor(splits["train"]["inputs"], dtype=torch.float32)
    train_targets = torch.tensor(splits["train"]["targets"], dtype=torch.float32)
    model = transmodel().to(device="cpu", dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    loss_function = torch.nn.MSELoss(reduction="mean")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_reduction_factor,
        patience=config.lr_scheduler_patience, threshold=config.lr_scheduler_threshold,
        threshold_mode="abs", cooldown=config.lr_cooldown_checks,
        min_lr=config.lr_min,
    )
    sampler = torch.Generator(device="cpu").manual_seed(run_seed)
    stop_state = StopState()
    run_directory = create_run_directory(run_directory)
    run_config = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(config).items()
        if not name.startswith("_")
        and isinstance(value, (str, int, float, bool, tuple, dict, Path))
    }
    run_config["max_updates"] = update_limit
    run_config["seed"] = run_seed
    data_identity = save_datasets(
        run_directory, splits, config.input_columns, config.target_columns,
    )
    manifest = initial_manifest(
        run_directory, splits, config, max_updates is not None,
        run_config=run_config,
        normalization={name: normalization[name] for name in
                       ("input_mean", "input_scale", "target_mean", "target_scale")},
        model_config={"widths": [4, 32, 32, 2], "activation": "ReLU"},
    )
    publish(run_directory, manifest, write_json)
    update = 0
    epoch = 0
    exposures = 0
    next_position = 0
    permutation = torch.empty(0, dtype=torch.int64)
    last_checkpoint = None
    started = time.perf_counter()
    try:
        train = splits["train"]
        save_training_baselines(run_directory, baseline_metrics(
            train["physical_inputs"], train["physical_targets"],
            normalization["target_mean"], normalization["target_scale"],
            groups=train["groups"], ruler_tolerances=config.tolerances,
        ))
        while True:
            evaluation_due = (
                update == 0 or update in config.early_eval_updates
                or update % config.eval_interval == 0 or update == update_limit
            )
            if evaluation_due:
                results = evaluate_splits(model, splits, normalization, update)
                train_summary = results["train"]["overall"]
                validation_summary = results["validation"]["overall"]
                decision = check_stop(
                    stop_state, update,
                    train_summary["pass_rates"][config.stopping_ruler],
                    validation_summary["pass_rates"][config.stopping_ruler],
                    validation_summary["standardized_mse"],
                    target_pass_rate=config.target_pass_rate,
                    min_delta=config.min_delta, patience_updates=config.patience_updates,
                    max_updates=update_limit,
                    fixed_budget=config.fixed_budget,
                )
                rate_before = optimizer.param_groups[0]["lr"]
                scheduler_checked = not decision.should_stop and update % config.eval_interval == 0
                if scheduler_checked:
                    scheduler.step(validation_summary["standardized_mse"])
                path = save_checkpoint(
                    run_directory, update, model, results["validation"],
                    normalization=normalization,
                    model_config={"widths": [4, 32, 32, 2], "activation": "ReLU"},
                    run_config=run_config, train_result=results["train"],
                    optimizer=optimizer, scheduler=scheduler, ruler=config.stopping_ruler,
                    data_identity=data_identity,
                    continuation={
                        "epoch": epoch, "permutation": permutation,
                        "next_position": next_position, "sampler_rng": sampler.get_state(),
                        "exposures": exposures, "stopping_state": asdict(stop_state),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
                verify_reload(path, model, train_inputs[:config.batch_size])
                last_checkpoint = path.name
                for name, result in results.items():
                    save_evaluation(run_directory, update, name, result)
                append_csv(run_directory / "evaluation_events.csv", {
                    "update": update, "epoch": epoch, "exposures": exposures,
                    "elapsed_seconds": time.perf_counter() - started,
                    "lr_before": rate_before, "lr_next": optimizer.param_groups[0]["lr"],
                    "scheduler_checked": scheduler_checked,
                    "progress_reference_mse": stop_state.progress_reference_mse,
                    "progress_reference_update": stop_state.progress_reference_update,
                    **asdict(decision),
                })
                manifest["checkpoints"].append(save_predictions(
                    run_directory, path, model, splits, normalization, config.target_columns,
                ))
                manifest["generation"] += 1
                publish(run_directory, manifest, write_json)
                print(f"update={update} train_mse={train_summary['standardized_mse']:.6g} "
                      f"validation_mse={validation_summary['standardized_mse']:.6g} "
                      f"next_lr={optimizer.param_groups[0]['lr']:.6g} "
                      f"{decision.reason.value}")
                if decision.should_stop:
                    choice = path.name if decision.selection == "current" else decision.selection
                    selected = select_checkpoint(run_directory, choice, reason=decision.reason.value)
                    summary = {
                        "status": "completed", "update": update, "epoch": epoch,
                        "exposures": exposures, "elapsed_seconds": time.perf_counter() - started,
                        "last_valid_checkpoint": last_checkpoint,
                        "selected_checkpoint": selected.name, **asdict(decision),
                    }
                    write_json(run_directory / "termination.json", summary)
                    manifest["status"] = "completed"
                    manifest["selected_checkpoint"] = selected.stem
                    manifest["termination"] = {
                        "reason": decision.reason.value,
                        "final_update": update,
                        "selected_checkpoint": selected.stem,
                    }
                    manifest["generation"] += 1
                    publish(run_directory, manifest, write_json)
                    return run_directory

            if next_position == len(permutation):
                epoch += 1
                permutation = torch.randperm(len(train_inputs), generator=sampler)
                next_position = 0
            end = min(next_position + config.batch_size, len(permutation))
            indices = permutation[next_position:end]
            learning_rate_used = optimizer.param_groups[0]["lr"]
            batch_loss = train_one_batch(model, optimizer, loss_function,
                                         train_inputs[indices], train_targets[indices])
            update += 1
            exposures += len(indices)
            next_position = end
            append_csv(run_directory / "batches.csv", {
                "update": update, "epoch": epoch, "batch_rows": len(indices),
                "exposures": exposures, "elapsed_seconds": time.perf_counter() - started,
                "learning_rate": learning_rate_used, "batch_mse_before_update": batch_loss,
            })
    except (Exception, KeyboardInterrupt) as error:
        numerical = isinstance(error, FloatingPointError) or "non-finite" in str(error).lower()
        write_json(run_directory / "termination.json", {
            "status": "aborted", "reason": "NUMERICAL_FAILURE" if numerical else type(error).__name__,
            "message": str(error), "last_completed_update": update,
            "epoch": epoch, "exposures": exposures,
            "elapsed_seconds": time.perf_counter() - started,
            "last_valid_checkpoint": last_checkpoint, "selected_checkpoint": None,
        })
        aborted_manifest = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
        aborted_manifest["status"] = "aborted"
        aborted_manifest["termination"] = {
            "reason": "NUMERICAL_FAILURE" if numerical else type(error).__name__,
            "last_completed_update": update,
            "message": str(error),
        }
        aborted_manifest["generation"] += 1
        write_json(run_directory / "run.json", aborted_manifest)
        raise


def main():
    """Use --max-updates for a short diagnostic run in a new output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-directory", type=Path)
    arguments = parser.parse_args()
    directory = run_training(max_updates=arguments.max_updates,
                             run_directory=arguments.run_directory, seed=arguments.seed)
    print(f"Run saved to {directory}")


if __name__ == "__main__":
    main()
