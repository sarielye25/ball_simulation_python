# Transmodel: initial execution plan

Updated 2026-09-24. Sariel implements; Sary reviews.
See [supporting notes](transmodel_training_notes.md) for reasoning, hypotheses, evidence, and limitations.

## 1. Configure the run

- Inputs: [v0, F, t1, t]; outputs: [displacement, final velocity].
- Use labels/train.csv (8,000 rows) and labels/validation.csv (1,000 rows); reserve test data.
- Before training, choose ONE stopping milestone below and confirm required coverage.
- Proposed coverage: 95% overall AND 95% within each main motion group.
- Record a random seed; use the existing 4 → 32 → 32 → 2 network, CPU, and float32.
- Use Adam with learning rate 0.001 and zero weight decay; batch size 128; maximum 10,000 updates.

| Milestone | Displacement tolerance | Velocity tolerance |
|---|---:|---:|
| Coarse | 0.1 m | 0.1 m/s |
| Intermediate | 0.01 m | 0.01 m/s |
| Fine | 0.001 m | 0.001 m/s |

## 2. Complete implementation checks

1. Check column order, finite values, input shape [batch, 4], target shape [batch, 2], and t1 ≤ t.
2. Fit input/output means and standard deviations on training data only; handle zero or near-zero scales.
3. Verify that normalization followed by its inverse reconstructs the original values.
4. Independently check reference examples: rest, sliding, stopping, reversal, and breakaway.
5. Run one update; verify finite loss/gradients and changed parameters.
6. Fit 32–64 fixed training examples; verify substantial error reduction.
7. Start the baseline with a fresh model and optimizer after these checks.

## 3. Train

- Standardize inputs and outputs using the saved training statistics.
- Use MSE averaged over both standardized outputs and all batch examples.
- Each update: training mode → predict → loss → clear gradients → backpropagate → optimizer step.
- Reshuffle each epoch; retain the final partial batch and all examples, including those already passing.
- Log training loss and update count; weight epoch-average loss by example count.
- Validate at updates 0, 10, 20, 50, 100, then every 100 updates and at normal termination.

## 4. Validate

- Use the same full validation set, evaluation mode, and disabled gradient tracking.
- Record standardized MSE; inverse-transform predictions before computing physical errors.
- For each output, record MAE, 95th-percentile absolute error, and maximum absolute error.
- Evaluate all three milestones; a sample passes only when BOTH absolute output errors meet tolerance.
- Report overall joint pass rates and rates/counts for each reference-defined motion group:
  always resting; moved then stopped; still moving at observation.
- Report boundary errors, joint pass rates, and count: v0 = 0 and abs(abs(F) − 3.924 N) ≤ 0.2 N.
- Give the boundary group no separate acceptance threshold; retain its samples in overall/main-group scores.
- At update 0, also evaluate zero-output [0, 0] and constant-velocity [v0 × t, v0] baselines.
- Retain per-example errors at updates 0 and 10 if reached; compare improvement/regression fractions and sizes.
- Compare training and validation loss; flag sustained validation worsening while training loss falls.

## 5. Stop and select

1. At the first validation check meeting the chosen tolerance AND all required coverage thresholds, save and stop.
2. Apply this rule at update 0 too; perform no further optimizer updates after passing.
3. Otherwise stop at 10,000 updates; stop and diagnose non-finite values or implementation failures.
4. Keep the lowest-validation-MSE checkpoint throughout training.
5. Select the first-passing checkpoint if successful; otherwise select the lowest-MSE checkpoint.
6. Record termination reason, first observed passing update/time, and unmet requirements.
7. Enable an additional plateau/deterioration rule only after specifying metric, minimum improvement,
   patience in optimizer updates, and earliest eligible update; apply the same rule across comparisons.

## 6. Save and review

- Save one run directory containing configuration, metrics CSV, curves, and checkpoints.
- Record data hashes/subset row IDs, seeds, normalization, model/optimizer settings, software, and device.
- Record target/coverage, group definitions/counts, updates, epochs, unique examples, exposures, and time.
- Save weights, model configuration, and preprocessing together; verify reloaded predictions match.
- Save optimizer/random-generator states if exact continuation is needed.
- If errors concentrate near discontinuities, inspect those cases and consider targeted data or a rest/sliding model.
- For widespread errors, inspect implementation, normalization, optimization, and capacity first.
- Review boundary diagnostics even after target achievement.
- Freeze architecture, training protocol, and checkpoint selection before final test evaluation; do not tune on test.

## 7. Follow-up comparisons, after baseline review

1. Screen 4 → 16 → 2, 4 → 32 → 2, 4 → 16 → 16 → 2, and 4 → 32 → 32 → 2 on 8,000 rows.
2. Match data, maximum budget, target, optimizer protocol, and validation schedule; start with one seed.
3. Repeat promising contrasts with three seeds; select two informative architectures for data-size curves.
4. Use nested training subsets of 500, 1,000, 2,000, 4,000, and 8,000; pair subsets across architectures.
5. Refit normalization per subset; use a fresh model/optimizer for every size and repetition.
6. Plot errors/pass rates against distinct labels and updates/time; report the smallest tested size passing all three runs.
7. Compare losses after architecture results are interpretable; evaluate rollouts separately afterward.
8. For rollouts, match force schedules, track absolute position externally, and flag out-of-range states.
