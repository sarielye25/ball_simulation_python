# Transmodel: supporting notes

Companion to [the execution plan](transmodel_training_plan.md), updated 2026-09-25.
This file contains reasoning, hypotheses, evidence, and interpretation. The execution plan defines the workflow.
Related material: `transmodel_exploration_plan.md` and the tolerance/validation PDFs in `learning_documents/`.
Status: label-based baseline checks have run; neural-network training has not.

## Purpose and scope

The project learns [displacement, final velocity] from [v0, force, force duration, observation time].
Its immediate value is an understandable, reproducible learning experiment and portfolio output.
The analytical simulator already solves the reference task; any speed or deployment advantage needs measurement.
Mass and friction are fixed. Labels measure agreement with this synthetic reference, not real hardware.
The network has 1,282 trainable parameters. The physics function returns velocity first; stored labels use displacement first.

## Assumptions still being tested

| Assumption | Interpretation |
|---|---|
| 0.001 is the required tolerance | A provisional milestone; intended use must determine acceptable error and coverage. |
| Equal numerical tolerances give equal importance | Metres and metres per second are different units; equal numbers are a convenience. |
| Two hidden layers outperform one | Width, parameter count, optimization, and data also affect results. |
| More labels solve poor accuracy | Only when data quantity or coverage is a relevant limitation. |
| A passing example no longer needs training | Shared parameter updates can make it fail again. |
| The discontinuity dominates practical error | A known concern that must be assessed from observed failures. |

## Why validation uses several measurements

Training loss and optimizer gradients update parameters. Validation measures held-out behavior and informs model selection and stopping.
Average loss indicates progress under its scoring rule; tolerance pass rate indicates achievement of a requirement.
Neither alone describes where failures occur.

A pass rate can remain zero while every prediction improves toward the threshold.
It can also increase while average error worsens, if some examples barely pass and others deteriorate substantially.
MAE measures average absolute error. The 95th percentile describes an error bound covering 95% of evaluated samples.
Maximum observed error concerns only the evaluated set, not every possible input.

All examples share parameters: ten updates may improve many predictions slightly, a few substantially, or improve some while worsening others.
Paired comparisons on the same rows reveal these changes without requiring an explanation of every neuron.
The fraction improving must be considered alongside the magnitude of changes.
One disappointing early check is insufficient to reject the run.

Check frequency serves decisions, rather than being the main objective.
The schedule 0, 10, 20, 50, 100, then every 100 uses absolute optimizer-update counts.
Checks in the middle remain useful because the end of useful training is not known in advance.
The first observed passing check may occur after the actual threshold crossing between checks.
Repeated checks on the same validation set are not independent statistical confirmations.

## Tolerance and motion groups

Three tolerance levels (0.1, 0.01, 0.001 in the respective physical units) are evaluation milestones, not training stages.
A sample passes only when both displacement and velocity meet their tolerances.
Sariel confirmed that overall training AND validation joint pass rates must each reach at least 95%
at the selected tolerance. Motion-group rates are diagnostic only, with no group acceptance threshold.
This is a project criterion, not a requirement established by a real-world application.
The stopping milestone must be chosen before running; required coverage is fixed at 95%.

Always-resting and moved-then-stopped cases can both have zero final velocity.
Group membership therefore comes from reference behavior, not predictions or final velocity alone.
Group sizes matter: a percentage from a few cases provides weak evidence about rare failures.
The boundary diagnostic overlaps the main motion groups; it is not a fourth disjoint group.
For this pilot it has no separate acceptance threshold. Poor boundary accuracy can coexist with target achievement.

Fixed absolute tolerances keep the initial benchmark simple.
Looser tolerances near difficult boundaries require justification from the intended use, not just model difficulty.
Pure relative error is undefined at zero and unstable near zero.
A future mixed rule could use absolute error ≤ absolute allowance + relative allowance × |reference value|,
with separate rules for displacement and velocity.

## Understanding evaluation and group labels

### Reading the three tolerance rulers

Each validation prediction is evaluated once, then its physical errors are compared with all three
tolerance pairs. For example, displacement error 0.006 m and velocity error 0.008 m/s pass the coarse
and intermediate rulers but fail the fine ruler. The pass rates therefore satisfy coarse >=
intermediate >= fine on the same sample set. A hypothetical result of 99%, 96%, and 72% means
intermediate precision reaches 95% coverage but fine precision does not. The training-set rate must
also reach 95% for the selected ruler to trigger success. These numbers are illustrative, not measurements.
The selected ruler changes the stopping criterion, not the standardized MSE or its gradients.

### Group metadata survives shuffled batches

The current CSV has six numeric columns and no motion-group column. Attach a derived group ID in
the dataset loader, or store it in a sidecar keyed by split, dataset hash and row index; do not silently
rewrite the existing labels. A sample carries its inputs, targets and group ID together when shuffled.
The ID is diagnostic metadata, not an input feature or an extra target for the regression network.

For the current reference and positive force duration, initial rest with force at or below the static
threshold identifies always-resting cases. Among the remaining rows, zero reference final velocity
identifies moved-then-stopped cases; nonzero final velocity identifies moving-at-observation cases.
Use reference stop logic, not the prediction tolerance, and verify these rules on representative cases.
Zero net displacement alone is insufficient because reversal can return an object toward its origin.
Moving at observation is a deliberate name: the group can contain reversals and is not necessarily
in motion at every intermediate instant. Event-specific subdivisions can be added later if useful.

An unreduced squared-error tensor has shape [batch, 2]. Average its two entries per sample to obtain
[batch] losses, select samples by group ID, then accumulate group sums and counts across batches.
The mean across all samples still supplies the training loss. If only that scalar mean was retained,
individual losses cannot be reconstructed; keeping them before reduction solves this problem.
Evaluation uses the whole fixed validation set, so group curves do not depend on which groups happen
to appear in a particular training batch. Missing groups have count zero and undefined rates.

### Why evaluate simple baselines at update zero?

The previous plan's third-from-last validation bullet meant evaluating two non-learning predictors:
zero output [0, 0], and constant velocity [v0 * t, v0]. Neither uses a trained network. Score both on
the same validation rows and rulers; they show whether good-looking scores arise merely from many
stationary or zero-final-velocity examples. The existing evidence below demonstrates that pitfall.
This is separate from the next bullet, which compares the same examples at neural updates 0 and 10.

## Discontinuities and physical limits

The static-friction threshold is a force: 0.4 × 1 kg × 9.81 m/s² = 3.924 N.
The reference mapping from initial inputs to future outcomes is discontinuous at breakaway.
The current linear/ReLU network is continuous, so it cannot approximate that jump with uniformly tiny error everywhere.
A steep continuous transition may nevertheless fit a finite evaluation set well.
For a jump of size J, the supremum error near the boundary is at least J/2 for a continuous approximation.

The initial boundary band is ±0.2 N around the threshold for initially resting inputs.
This is a diagnostic sampling choice, not a physical constant; a broad band can hide narrow failures.
Later diagnostic slices can approach the force threshold from both sides and approach v0 = 0.
Small predicted state errors can trigger different friction branches in repeated predictions.
Endpoint agreement does not establish correct trajectories, rollout accuracy, or real-world accuracy.

## Stopping and checkpoint interpretation

The user's chosen policy is to stop at the first scheduled check meeting the selected target.
The 10,000-update budget is a maximum, not a required duration.
Stopping on target avoids further training after observed success; it does not guarantee prevention of overfitting.
Overfitting may appear before the target is reached: training loss falls while validation loss persistently worsens.

Target reached, progress stalled, budget exhausted, and numerical/implementation failure are distinct outcomes.
A plateau under the current setup does not establish success or prove the architecture is inadequate.
A budget cap while errors still improve does not establish convergence.
The stop checker combines target achievement, 500 updates without meaningful validation-MSE progress,
and a hard 10,000-update cap. The absolute minimum improvement remains to be selected before running.

### How the early stopper works

A validation check measures performance; a checkpoint saves a recoverable model state.
The early stopper remembers progress and decides whether to continue. Logging a best step number
does not preserve that model unless its checkpoint was actually saved.

Use validation MSE as the early-stopping metric, while the joint physical pass rates retain their
separate role in the success gate. Initialize the progress reference and its update counter from
the finite validation result at update 0. At subsequent checks, an improvement is meaningful when
`current_mse < progress_reference_mse - min_delta`, using an absolute `min_delta` in standardized
MSE units. On meaningful improvement, replace the reference and reset the last-improvement update.
Otherwise leave both unchanged, so successive small gains can accumulate against the same reference.

Separately save every new absolute minimum validation MSE, even if its improvement is smaller than
`min_delta`. The best-checkpoint record and the patience reference serve different purposes.
After checking both overall training and validation rates for target achievement, stop for stagnation
when the updates since meaningful improvement reach 500. Start the clock at update 0 without an extra
warm-up, making update 500 the earliest possible stagnation stop.
Check numerical validity first; non-finite metrics are a numerical failure, not a patience event.

Measure patience in optimizer updates, not validation checks: the early validation schedule is denser.
With the confirmed patience of 500 and the last meaningful improvement at update 200, a check at
update 700 would stop if no further meaningful improvement occurred.
An improvement at update 500 would restart the clock there. Stopping occurs only at a validation
check, so the actual wait can exceed the configured patience. The value 500 is now confirmed;
the numerical minimum improvement must still be specified before the run.

Early stopping can respond to a plateau, noisy progress, or deterioration. It does not diagnose
the cause and does not guarantee prevention of overfitting. Recover the lowest-MSE checkpoint if
the success gate was never met, and report the unmet target rather than labeling stagnation as success.

Lowest MSE and highest tolerance pass rate need not occur at the same checkpoint.
The first target-passing checkpoint is the selected successful result; lowest MSE is the fallback if the target is not reached.
An isolated validation pass is not a guarantee of stable or universal accuracy.
The untouched test set evaluates the frozen final choice; test shortfalls must be reported without tuning against it.

## Training mechanics and normalization

A batch supplies one update; an epoch is one full pass over the selected training examples.
With 8,000 rows and batch size 128, one epoch has 62 full batches plus one batch of 64: 63 updates.
Reshuffling changes batch composition, not the number of distinct labels.
Passing examples remain eligible because future shared-parameter updates may worsen their predictions.
A hard pass/fail count provides no useful ordinary gradient almost everywhere.

For each output, z = (value − training mean) / training standard deviation.
The loss is the batch average of [(predicted z_d − z_d)² + (predicted z_v − z_v)²] / 2.
Standard-deviation scaling balances relative variation, not importance relative to physical tolerances.
Tolerance-based weighting is a possible later experiment; changing weights changes the optimization trade-off.
Training/evaluation mode and enabling/disabling gradients are separate controls, even without dropout or batch normalization.
A small-subset fit checks implementation; it is not evidence of generalization or a demand for perfect fit across discontinuities.

## Existing baseline evidence

On 2026-09-24, the zero-output predictor was evaluated on all 1,000 validation examples:

| Milestone | Velocity-only pass rate | Joint pass rate |
|---|---:|---:|
| Coarse | 50.6% | 12.1% |
| Fine | 48.1% | 6.7% |

Its displacement MAE was about 4.57 m and velocity MAE about 2.64 m/s.
There are 481 examples with zero final velocity; the tolerance PDF reports only 64 that remain stationary throughout.
This explains why velocity-only accuracy can look encouraging without learning.
The constant-velocity baseline ignores force and friction; it is a comparison point, not reference physics.
No neural-network training or test-set evaluation was performed for these checks.

## Data size and fair comparisons

One labeled example contains four inputs and both outputs.
Distinct examples, unique examples seen, repeated exposures, optimizer updates, elapsed time, and evaluation labels are separate quantities.
For example, 100 epochs over 8,000 rows use 8,000 distinct training examples and 800,000 exposures.
Time-to-target at 8,000 examples does not discover the minimum number of labels needed.

The dataset already contains 10,000 labels; subset experiments measure labels used, not generation costs already saved.
Normalization fitted outside the chosen training subset would contaminate its label budget.
Warm-starting a smaller-data model from a larger-data model would also contaminate that comparison.
CSV rows do not contain original sampling-category IDs; exact category-stratified subsets cannot be claimed without recovering them.
Three seeded repetitions show variability, not a precise statistical guarantee.
The smallest tested successful subset is not the exact minimum needed; results need not be monotonic.
Equal epochs give different dataset sizes different update counts. Equal updates do not imply equal compute across architectures.
Repeated tuning can overfit validation, which is why the final test set stays reserved.

| Architecture | Parameters including biases |
|---|---:|
| 4 → 16 → 2 | 114 |
| 4 → 32 → 2 | 226 |
| 4 → 16 → 16 → 2 | 386 |
| 4 → 32 → 32 → 2 | 1,282 |

At equal width, changing depth also changes parameter count; the comparison does not isolate depth alone.
A shared recipe measures performance under that recipe, not each architecture's best achievable performance.
Screening on 8,000 rows can miss an architecture that excels with less data.
The full four-architecture × five-size × three-seed grid is 60 runs before loss experiments.
An equally budgeted learning-rate search or parameter-matched comparison can address later specific questions.

## Interpreting failures and extensions

### Supplementary outputs from the first run

Sariel's priorities are tracking prediction ability on unseen conditions, identifying harder motion
groups, and measuring failure severity. Per-group curves distinguish slow learning from persistent
errors; counts prevent a small group's percentage from being mistaken for strong evidence.
Group-level difficulty localizes a problem but does not by itself identify its cause.

For the selected and last valid checkpoints, retain per-example validation results with stable row
IDs tied to the dataset hash: inputs, reference-defined group, targets, predictions, and absolute
displacement/velocity errors. This supports paired comparisons of which examples improved or regressed.
Analyze failed samples separately from the full validation set, using the selected stopping tolerance.
Report failure counts and per-output error summaries, including typical and extreme errors; if none
fail, report zero failures and leave failed-only summary statistics undefined rather than inventing zeros.

For positive tolerances, define severity as `r = max(abs(d_error)/tau_d, abs(v_error)/tau_v)`.
Under the strict pass convention, a sample passes when `r < 1`; `r = 1.1` means the worse normalized
error is 1.1 times its allowance, whereas `r = 20` signals a much larger miss. Keep the original errors
in metres and metres per second alongside this ranking. Inspect representative cases within failing
groups as well as the largest misses; extreme examples alone do not characterize typical failure.

Keep elapsed time and update counts for reproducibility, but do not make efficiency a primary first-run
claim: this small workload can be strongly affected by startup and validation overhead.

### A practical diagnosis after stopping

1. Inspect the raw validation curve and stopper settings. Slow continued improvement may fall below
   `min_delta`; short patience may mistake fluctuations for a plateau. A stop is evidence about the
   configured observation window, not proof that further learning is impossible.
2. Evaluate training and validation data at the same selected and last valid checkpoints, using the
   same preprocessing, evaluation mode, and metrics. Per-batch training losses were measured on
   different examples and changing parameters, so they are not an exact matched comparison.
3. Localize errors by motion group and boundary slice, then inspect failed examples and their severity.
   Broad errors suggest a different investigation from errors concentrated near breakaway or stopping.
4. Write down a leading explanation and a controlled follow-up check. Do not automatically increase
   network size, add data, or extend training merely because the stopper fired.

Examples of follow-up checks include fitting a small fixed set away from difficult boundaries to
check the training pipeline; comparing the original and a lower learning rate from the same saved
training state with matched update budgets and batches; or checking coverage of a difficult region
before adding independently generated training labels. Small-subset success does not prove global
capacity, and failure does not uniquely identify a bug. Change one main factor and keep a baseline
so the result can support or weaken the proposed explanation. These are conditional investigations,
not additional experiments required before the first-model demonstration.

Validation estimates generalization to unseen conditions represented by its distribution, not
understanding of physical laws or reliability at new masses/friction coefficients. Because validation
also guides stopping and later design choices, preserve the untouched test set for the frozen final
choice; never use test failures as inputs to this diagnostic tuning loop.

| Observation | Possible next investigation |
|---|---|
| Training and validation errors both high | Implementation, normalization, optimization, representation, discontinuities |
| Training error low, validation error high | Coverage gaps, representative data, capacity, regularization |
| Errors concentrated near breakaway, stopping, or reversal | Independent reference checks, targeted training data, regime-aware model |
| Both errors still improving at the budget cap | Longer optimization budget or learning-rate schedule |
| Endpoint predictions good, rollouts poor | Accumulated error, state-distribution shift, rollout horizons |

Targeted data may outperform simply adding uniform samples; this is an experimental question.
Any new training labels must be counted, and validation/test cases must not be copied into training.
Physics-informed or residual models remain possible alternatives when their purpose is justified.
Reference-state resets and free rollouts measure different behavior and should be distinguished.

## Repository limitation

`label_preparation_v2.py` imports the absent `label_preparation.py`.
Stored CSVs are available; regeneration requires restoring or reviewing that dependency.
Existing analytical checks are useful but not exhaustive physical verification.

## Sources

- [PyTorch: Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Scikit-learn: Learning curves](https://scikit-learn.org/stable/modules/learning_curve.html)
- [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)
- `learning_documents/transmodel_tolerance_discussion.pdf`
- `learning_documents/front_loaded_validation_review.pdf`
