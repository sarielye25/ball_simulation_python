# Cuboid Motion — First Training Plan

Predict one-dimensional motion: `[initial velocity, force, push duration, observation time] → [displacement, final velocity]`. Mass and friction stay fixed.

**Status — 27 September 2026:** The training pipeline has passed diagnostic checks. The full first training run and final test evaluation have not started.

## Done: how each step works

1. **Build the physics reference.** Calculate motion during the push and after release, including friction, stopping, and reversal. Analytical cases check rest, sliding, static thresholds, and breakaway.
2. **Prepare the data.** Generate 10,000 examples: 8,000 training, 1,000 validation, and 1,000 reserved test rows. `data.py` reads four inputs and two targets, validates values and shapes, and assigns reference motion groups. The training loop preserves row IDs and source-file hashes.
3. **Normalize the values.** Fit each input and output column's mean and standard deviation on training rows only. Apply those same statistics to validation and later test data. Replace scales below `1e-8` with one; verify that inverse scaling recovers physical values.
4. **Define the model.** Use a PyTorch `4 → 32 → 32 → 2` network with ReLU hidden layers, running on CPU with float32 tensors. Predict standardized displacement and velocity.
5. **Check learning.** `checks.py` verifies data, physics, scaling, finite gradients, and weight changes. A disposable 64-example fit reduced MSE from `1.74388` to `0.00584843` over 500 updates. Start the real run with a fresh model and optimizer.
6. **Implement training.** `training_loop.py` shuffles rows each epoch and retains the final partial batch. Each batch runs prediction, standardized MSE, gradient clearing, backpropagation, and an Adam update. Explicit shuffled indices and a batch cursor preserve the sampling position.
7. **Measure performance.** `metrics.py` evaluates the unchanged model on complete training and validation sets without gradients. It reports loss, physical errors, joint pass rates, motion groups, and breakaway cases. Zero-motion and constant-velocity predictions provide training baselines.
8. **Control progress and save models.** `stopping.py` checks target attainment, stalled progress, and the update limit. The loop reduces learning rate after stalls. `checkpoints.py` saves model, optimizer, scheduler, normalization, random states, and continuation metadata; reload checks require identical predictions. A resume command is still pending.
9. **Record and plot.** Each run gets its own `training_data/<timestamp>/` folder. CSV/JSON files store batch losses, evaluations, learning rates, settings, and termination details. `report.py` exports static PNG/SVG curves. The HTML report was retired; the Qt replacement remains a design draft.

The training record reports nine passing tests across preflight and loop checks, including a disposable 64-update run, learning-rate reduction, stopping, and checkpoint continuation. These checks establish implementation behavior; they are not full-run results.

## First-run settings

Current values in `config.py`:

| Setting | Value |
| --- | --- |
| Optimizer | Adam; initial learning rate `0.001`; weight decay `0` |
| Batch size / seed | `128` / `42` |
| Loss | Mean squared error across both standardized outputs and all batch rows |
| Evaluation updates | `0, 10, 20, 50, 100`, then every `100`, and at normal termination |
| Success target | At least `95%` joint passes on both full training and validation sets |
| Selected stopping tolerance | Intermediate: displacement error `< 0.01 m` **and** velocity error `< 0.01 m/s` |
| Meaningful progress | Validation MSE decreases by strictly more than `0.001` from the progress reference |
| Early stopping / maximum | `500` updates without meaningful progress / `10,000` updates |

Report all three tolerances on the same predictions: coarse (`0.1 m`, `0.1 m/s`), intermediate (`0.01 m`, `0.01 m/s`), and fine (`0.001 m`, `0.001 m/s`). Both errors must be strictly below their limits. Only the selected tolerance controls success.

### Learning rate and stopping

Use `ReduceLROnPlateau` on full-validation MSE: factor `0.5`, minimum learning rate `0.00001`, absolute threshold `0.001`, and cooldown `0`. On continuing runs, call it at update zero and every 100 updates. PyTorch `patience=1` reduces the rate after two unsuccessful regular checks, or 200 updates; early evaluations at 10, 20, and 50 do not advance it.

At each evaluation, reject non-finite metrics, then apply these decisions in order:

1. **Target reached:** select the first checkpoint meeting both 95% targets, including update zero.
2. **No progress:** update the progress reference after a meaningful improvement; otherwise stop once 500 updates have elapsed. Select the lowest-validation-MSE checkpoint.
3. **Budget exhausted:** stop at 10,000 updates and select the lowest-validation-MSE checkpoint.

If patience and the cap coincide, record both and use no progress as the primary reason. Reduce the learning rate only when continuing; a reduction does not reset stopping patience. Save every evaluation checkpoint and rank every absolute MSE improvement, even below the progress threshold. Numerical failure aborts separately.

## Next: run, review, and extend

1. **Run the first experiment.** Recheck settings, run preflight checks, and start fresh training. Keep reserved test data closed during training and model selection.
2. **Inspect curves and failures.** Compare matched training/validation curves against actual update counts. Show overall results, three motion groups (always resting, moved then stopped, moving at observation), and the overlapping breakaway slice within `0.2 N` of the static-force threshold at zero initial velocity. Groups are diagnostic; they add no stopping gate.
3. **Complete the report.** Retain counts, total and per-output MSE, each output's half-MSE contribution to loss, physical MAE/P95/maximum errors, and all three pass rates. Use matched plot scales, raw observations, and a 0–100% pass-rate axis; leave empty-group metrics undefined. Produce failure tables for selected and last valid checkpoints, ranked by `max(abs(d_error)/tau_d, abs(v_error)/tau_v)`. Qt implementation awaits design approval.
4. **Evaluate the frozen choice.** Implement `final_test.py`, freeze the protocol and selected checkpoint, then evaluate the 1,000 reserved test rows once. Report test results separately; do not tune against them.
5. **Demonstrate.** After reviewing the first result, animate learned versus reference motion and saved training progress.
6. **Compare.** Change one factor at a time: network size, dataset size, or loss. Use fresh models, training-subset normalization, and repeated seeds for promising comparisons. Study repeated predictions with and without reference-state feedback separately.
7. **Explore later.** Feedback control, optional reinforcement learning, rotation, language instructions, and physical validation.

Synthetic one-dimensional agreement does not establish real-world accuracy. Use observed failures to choose one explanation and one controlled follow-up.

**Learning rule:** Sariel writes code; the tutor explains, reviews, and gives hints. One experiment at a time.

Sources: [Project README](../../../README.md) · [Original HTML plan](transmodel_training_plan_updated.html) · [Training record](first_training_record.md) · [Current configuration](../config.py) · [Qt design](qt_building_plan.md). Latest implementation updates and current settings take precedence over older pending notes.
