# Transmodel: supporting notes

Companion to [the execution plan](transmodel_training_plan.md), updated 2026-09-24.
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
The candidate 95% overall and 95% within each main motion group is not yet an established application requirement.
The stopping milestone and coverage must be chosen before running.

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
Additional plateau/deterioration stopping thresholds remain unspecified; they are not automatically enabled.

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
