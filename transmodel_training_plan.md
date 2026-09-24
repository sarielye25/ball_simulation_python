# Transmodel training and comparison plan

Draft for discussion — 2026-09-23. Sariel implements and runs the code; Sary explains and reviews. No training has been run for this plan.

Validation decision structure agreed on 2026-09-24. A label-based baseline check has been run; neural-network training has not.

Read alongside `transmodel_exploration_plan.md`. The requested `tranmodel_training_plan.md` was not present; this document develops the existing exploration plan without replacing it.

## 1. Outcome and first principles

Build a reproducible learned predictor from `[v0, F, t1, t]` to `[d, v]`, and measure how architecture affects held-out error, distinct training examples needed, and computation. Depth of analysis: Standard.

Verified facts:

- The current network is 4 → 32 → 32 → 2, with ReLU hidden activations and linear outputs: 1,282 trainable parameters including biases.
- The current CSV files contain 8,000 training, 1,000 validation, and 1,000 test examples. Outputs are displacement first, velocity second; the physics function returns the opposite order.
- Mass and friction are fixed. Training labels are deterministic outputs of the reference function. Agreement with them measures simulator imitation, not real-world accuracy.
- A forward prediction is necessary before its error can be measured. Loss gradients provide the direction for parameter updates; a tolerance test alone does not.
- Repeated use of an example adds computation, not a new distinct label.

Choices that remain hypotheses:

| Proposed choice | Assessment |
|---|---|
| Both error tolerances are 0.001 | A candidate requirement; units, coverage, operating domain, and intended use determine its meaning. |
| Two hidden layers are better than one | An experimental question; width, parameter count, optimization, and data also matter. |
| More labels will solve poor accuracy | Only if inadequate data or coverage is a relevant limitation. |
| Ignore examples once their errors pass tolerance | Avoid for the baseline: shared parameter updates can make those examples inaccurate again. |
| One count of labels measures efficiency | Separate distinct examples, repeated exposures, optimizer updates, and time. |

The goal is an understandable, informative experiment. A complicated training platform is unnecessary for this small model.

## 2. Validation: direction, achievement, and stopping

Validation provides evidence for whether to continue training, which checkpoint to keep, and when to stop. The training loss and optimizer update the parameters; validation evaluates the resulting behavior on held-out examples. Check frequency serves these decisions. Checks during the middle of training remain necessary because we cannot know in advance when the ending begins.

### 2.1 Match the measurement to the question

| Question | Evidence to record | Limitation |
|---|---|---|
| Are predictions getting closer to the reference? | Validation standardized MSE, plus displacement and velocity mean absolute error (MAE) in physical units | An average can conceal worsening cases; standardized MSE weights outputs by training-data variation, not application tolerances. |
| Are predictions accurate enough? | Joint tolerance pass rate: the proportion satisfying both output tolerances | A pass rate hides how far failures miss the target and how much passing cases improve. |
| Is improvement widespread or concentrated? | Paired per-example error changes and results by motion condition | The fraction improving alone hides the size of improvements and regressions. |
| Is further training worthwhile? | Improvement over a meaningful update window, target status, and computation used | A plateau under one setup does not prove an architecture or dataset is inadequate. |

Use loss to detect progress, tolerances to judge achievement, and grouped errors to expose what averages conceal. A falling loss is evidence of progress under that scoring rule, not proof that every example improves or that the application requirement is met.

### 2.2 Physical tolerances and coverage

For each example, define `e_d = |predicted d - reference d|` and `e_v = |predicted v - reference v|`. It passes only when `e_d <= epsilon_d` **and** `e_v <= epsilon_v`. Evaluate all three provisional milestones with the same training objective:

| Milestone | Displacement tolerance | Velocity tolerance |
|---|---:|---:|
| Coarse | 0.1 m | 0.1 m/s |
| Intermediate | 0.01 m | 0.01 m/s |
| Fine | 0.001 m | 0.001 m/s |

These are experimental milestones, not established application requirements. Equal numerical tolerances for different units are a convenience. Labels provide reference answers; they do not determine acceptable errors or required coverage. A candidate coverage is 95% overall and 95% within each main motion group; confirm the chosen milestone and coverage before enabling target-based stopping or declaring success in comparisons.

At every validation check, report standardized MSE; displacement and velocity MAE, 95th-percentile absolute error, and maximum observed error; and joint pass rates at all three milestones. Report group sizes and group pass rates for always-resting cases, cases that move and then stop, and cases still moving at observation. Determine groups from reference behavior, not model predictions; zero final velocity alone cannot distinguish always-resting from moved-then-stopped cases.

Also report a boundary diagnostic group: initially resting inputs with `abs(abs(F) - 3.924 N) <= 0.2 N`. This is a fixed sampling band for analysis, not a physical constant or a reason to relax tolerances. Show its errors and pass rates explicitly, including any shortfall. A successful overall score must not conceal this limitation.

The reference mapping is discontinuous at breakaway, whereas the current ReLU network is continuous. Finite-set success cannot establish uniformly tiny errors for every possible input near that boundary. Simulator agreement also does not establish real-world or repeated-rollout accuracy.

### 2.3 Beginning: establish evidence of useful learning

Evaluate the untrained model at update 0 and use the same validation rows at subsequent checks. Look for a sustained decrease in validation loss across several checks, supported by physical errors for both outputs. Compare against simple baselines as learning progresses. A single disappointing check after ten updates is insufficient to reject the run; non-finite values or broken data contracts require immediate investigation.

All examples share model parameters. Ten updates may improve many predictions slightly, reduce a few large mistakes, or cause some examples to pass while others fail. There is no assumed sequence in which training finishes one example before moving to another.

For the early diagnostic, retain per-example errors at updates 0 and 10. Compare each output's fraction of examples improving, the magnitudes of improvements and regressions, and where those changes occur. Use paired comparisons on the same rows, not different random validation batches. If the aggregate story remains unclear, repeat this diagnostic at a later scheduled check. A pass rate can remain zero while predictions improve substantially toward its threshold.

### 2.4 Ending: separate success from stopping

| Outcome | Meaning and action |
|---|---|
| Target reached | A saved checkpoint meets the predefined joint tolerances and coverage requirements. Record its actual metrics and group limitations. |
| Progress stalled | Validation improvement is too small over a predefined window under the current setup. Retain the best observed checkpoint and investigate before changing the setup. This is not accuracy success. |
| Budget exhausted | The update or time cap is reached without a target-based or plateau-based stop. Report whether progress was continuing; do not label the cap as convergence. |

For the first pilot, retain the fixed 10,000-update budget to observe the curve; record target crossings and possible plateaus without automatically stopping on them. Stop and diagnose numerical or implementation failures separately. Pilot observations will inform any later early-stopping settings.

Before subsequent comparisons, freeze any early-stopping rule: monitored metric, minimum meaningful improvement, patience in optimizer updates, and the earliest update at which plateau stopping is allowed. Patience counts updates since the last qualifying improvement, and stopping can occur only at scheduled validation checks. Do not invent thresholds from one early fluctuation or change them separately for each candidate.

Keep the lowest-validation-MSE checkpoint and the first observed target-passing checkpoint separately. The stopping checkpoint need not be the best checkpoint, and the lowest-MSE checkpoint need not have the highest pass rate. For the pilot's primary comparison, use the lowest-MSE checkpoint and report whether that specific checkpoint meets the target; report time-to-target separately. If a later task requires selecting a target-qualified checkpoint, specify that selection rule before comparisons.

Report whether a target crossing persists at later pilot checks. Repeated checks on the same validation set are not independent confirmations. Freeze architecture, training protocol, and checkpoint selection before evaluating on the reserved test set; report test shortfalls without tuning against that set.

## 3. The loop, conceptually

There are two cycles, with different responsibilities.

**Learning cycle:** training batch → normalized inputs → model predictions → differentiable loss → clear previous gradients → backpropagation → optimizer updates weights and biases → next batch.

**Evaluation cycle:** fixed validation set → predictions without gradient tracking → inverse output normalization → physical error metrics and joint tolerance check → record progress and checkpoint decisions.

The tolerance check belongs to evaluation and stopping decisions. It does not filter examples out before the baseline loss. A hard pass/fail count provides no useful ordinary gradient almost everywhere; squared error continues to provide training information.

- A batch is a group of examples used for one update.
- An optimizer step is one parameter update.
- An epoch is one complete pass over the selected training examples.
- Validation influences model selection and stopping, but never supplies training gradients.
- The test set is reserved until the architecture, training protocol, and checkpoint-selection rule are frozen.

At the start of each epoch, reshuffle the same selected input–label pairs and divide them into new batches. With 8,000 examples and batch size 128, each epoch contains 62 full batches and one final batch of 64, giving 63 updates. Reshuffling changes batch composition and update order, so examples contribute alongside different examples across epochs. It creates varied optimization experiences but does not create new labels, add physical conditions, or increase the number of distinct training examples. Keep every selected training example eligible in every epoch, including examples that currently meet a candidate tolerance.

Use training mode for updates, and evaluation mode plus disabled gradient tracking for evaluation. These are separate concepts in PyTorch, even though this particular network has no dropout or batch normalization.

## 4. A small initial training recipe

These are starting settings to test, not claims of optimality.

| Item | Initial choice | Reason |
|---|---|---|
| Model | Existing 4 → 32 → 32 → 2 | Establish the planned baseline first. |
| Inputs | Standardize each feature using training-only mean and standard deviation | Prevent input units and magnitudes from dominating conditioning. |
| Outputs | Standardize d and v separately using training-only statistics | Make initial numerical scales manageable. Save the inverse transform. |
| Loss | Mean squared error over both standardized outputs and all examples in a batch | Simple, differentiable, and gives larger errors more influence. |
| Optimizer | Adam, learning rate 0.001, zero weight decay | A practical starting point for this small regression model. |
| Batch size | 128, shuffled; retain the final partial batch | Modest computation with multiple examples per update. |
| Device / precision | CPU and float32 initially | The current requirements use CPU PyTorch; benchmark before adding hardware complexity. |
| Validation | Front-loaded schedule: updates 0, 10, 20, 50, and 100; then every 100 updates and at the final update | Initial observation schedule serving the direction and stopping decisions in section 2; intervals are not themselves the objective. |
| Initial budget | At most 10,000 optimizer updates | A bounded pilot, not a claim that convergence occurs by this point. |
| Checkpoint | Lowest validation standardized MSE | A smooth, predefined selection rule. Report tolerance metrics at that checkpoint. |

If z_d = (d − mean_d)/std_d and similarly for velocity, the loss is the batch mean of [(predicted z_d − z_d)² + (predicted z_v − z_v)²]/2.

Standard-deviation scaling balances relative variation; it does **not** mean equal priority relative to the 0.001 tolerances. Acceptance remains in physical units. If one output remains disproportionately inaccurate, compare tolerance-based loss weighting as a separate, explicitly recorded experiment. Freeze the loss during architecture comparisons.

Treat zero or nearly zero feature scales explicitly rather than dividing by zero. Fit normalization on the selected training subset in data-size experiments; using labels outside that subset would invalidate its claimed label budget.

Follow section 2.4 for termination and checkpoint decisions. After the fixed-budget pilot, decide whether a longer budget, a learning-rate reduction, or a common early-stopping rule is warranted before starting comparisons. A plateau and reaching the target accuracy remain distinct outcomes.

Apply the same front-loaded validation schedule to every candidate in a comparison so each receives the same checkpoint opportunities. The schedule assumes one optimizer update per processed training batch; if gradient accumulation or skipped optimizer steps are introduced later, trigger validation by the optimizer-update counter rather than the batch counter.

Record the first validation check that passes the joint target and its checkpoint separately from the checkpoint with lowest validation MSE. An isolated pass can fluctuate; report whether it persists at subsequent checks. Repeated checks on the same validation set are not independent statistical confirmations.

## 5. Evidence before a full run

Sariel builds these parts in order; Sary reviews each milestone.

1. **Data contract:** inspect training rows, column order, input shape [batch, 4], target shape [batch, 2], finite values, and t1 ≤ t. Demonstrate that normalization followed by its inverse reconstructs the original physical values.
2. **Reference checks:** independently calculate a few rest, sliding, stopping, reversal, and breakaway cases. Matching the label generator to itself is not independent verification. The current physics file hash matches the recorded generation hash, but existing analytical checks are not exhaustive.
3. **One update:** explain what the predictions, scalar loss, gradients, and changed parameters represent. Check finite gradients and actual parameter changes. One Adam step need not improve every output or every example.
4. **Small fixed training subset:** repeatedly fit roughly 32–64 examples. A substantial training-error decrease is a useful implementation check, not evidence of generalization. Do not require perfect fit across a physical discontinuity.
5. **Baseline run:** train on the 8,000 examples, evaluate on validation, save curves and a reloadable checkpoint. Confirm saved-model predictions reproduce those of the selected checkpoint.

Before training, evaluate both a zero-output predictor (`d = 0`, `v = 0`) and the constant-velocity prediction (`d = v0 × t`, `v = v0`) on validation using the same metrics as the network. These are simple performance references, not substitutes for the analytical simulator that supplies the labels.

The 2026-09-24 label-based check illustrates why this matters. On the 1,000 validation rows, the zero-output predictor passes velocity alone on 50.6% of cases at 0.1 m/s and 48.1% at 0.001 m/s, but passes both outputs on only 12.1% and 6.7%, respectively, at the matching milestones. Its displacement MAE is approximately 4.57 m and velocity MAE is 2.64 m/s. Many cases end at rest, so an apparently encouraging velocity pass rate can occur without learning. This check involved no neural-network training and did not use the test set.

An operational issue for later data generation: `label_preparation_v2.py` imports `label_preparation.py`, but that source file is absent. The stored CSVs are available; regeneration needs that dependency restored or reviewed first. No repair is part of this planning step.

## 6. Measuring how many labels are needed

Define one labeled example as one distinct input condition paired with both outputs [d, v]. It contains two scalar targets, but counts as one example.

Record these separately:

| Quantity | Meaning |
|---|---|
| N_train | Distinct labeled examples available to this training run. |
| Unique examples seen | Distinct selected examples actually used so far. |
| Example exposures | Total appearances in training batches, including repetitions. |
| Optimizer updates | Number of parameter updates. |
| Elapsed training time | Actual cost on the recorded device. |
| Validation/test examples | Evaluation labels, reported separately from training labels. |

For example, 8,000 examples used for 100 complete epochs means 8,000 distinct training examples and 800,000 exposures. Counting exposures answers an optimization-efficiency question. It cannot establish that the model needs 800,000 distinct labels.

Training once on all 8,000 and recording the first successful epoch measures **time or updates to accuracy at N = 8,000**. It does not discover the minimum training-set size.

To estimate sample efficiency, construct learning curves:

1. Choose nested subsets of 500, 1,000, 2,000, 4,000, and 8,000 rows from training only. One seeded random permutation and its prefixes are enough initially. Inspect the coverage of identifiable conditions such as initial rest, zero force, and force removal for each subset.
2. Use exactly the same subset memberships for every architecture within a repetition. Keep validation fixed. The CSV does not store original sampling-category IDs, so do not claim exact category-stratified subsets without recovering that information.
3. Train a fresh model and fresh optimizer at each size. Refitting a model previously trained on a larger set would contaminate the smaller label budget; growing one warm-started model instead measures a different, sequential-training procedure.
4. Give candidates a predefined common maximum update budget, validation cadence, batch size, and optimizer protocol. Log actual time and exposures. Equal epochs would give larger datasets more updates; equal updates still do not mean equal compute for differently sized networks.
5. Start with one seed for exploration; repeat meaningful comparisons with three independently seeded subset/order/initialization repetitions. Pair each repetition's subsets across architectures. Three runs show variability, not a precise statistical guarantee.
6. Plot physical validation errors and joint pass rate against N; show results for each seed. Report the smallest **tested** N that passes in all three runs as a conservative practical criterion, including exceptions and the training budget.

If 1,000 fails and 2,000 passes, do not claim that exactly 2,000 labels are necessary. It is the smallest successful tested size under this protocol; stochastic results need not be monotonic. Add an intermediate size only if narrowing the transition is useful. If none passes, record “not reached within tested data and training budgets,” not “impossible” or an invented minimum.

The existing dataset already contains 10,000 generated labels. This experiment measures how many the training procedure uses, not savings in labels already acquired. The fixed 1,000 validation labels also inform decisions and must be disclosed. Repeated tuning can overfit validation; use the untouched test set once for the frozen final comparison.

## 7. Architecture comparison without an oversized sweep

| Hidden layers | Full architecture | Parameters including biases |
|---|---|---:|
| One, width 16 | 4 → 16 → 2 | 114 |
| One, width 32 | 4 → 32 → 2 | 226 |
| Two, width 16 | 4 → 16 → 16 → 2 | 386 |
| Two, width 32 | 4 → 32 → 32 → 2 | 1,282 |

At equal width, depth also changes parameter count. These comparisons test practical architecture choices under the chosen training recipe; they do not isolate a universal causal effect of depth. Parameter-matched comparisons can follow if depth itself becomes the scientific question.

Recommended sequence:

1. Complete one working baseline and understand its errors.
2. Screen the four architectures at N = 8,000 with one fixed protocol and seed. Repeat promising contrasts with three seeds.
3. Build data-size curves for two informative candidates, ideally one shallow and one deep, before expanding further. State that screening at 8,000 may miss a model that performs especially well with little data.
4. Compare losses only after the architecture experiment is interpretable.
5. Evaluate repeated predictions after one-step behavior is understood.

The full four-architecture × five-size × three-seed grid is 60 runs before any loss experiment. Do not begin there. If a result changes after a small, equally budgeted learning-rate search for each candidate, report that sensitivity. A single shared recipe compares performance under that recipe; it does not establish the best possible performance of every architecture.

## 8. When adding labels is inefficient

| Observation | What to investigate next |
|---|---|
| Training and validation errors remain high | Data/normalization bugs, optimization progress, representation, and physical discontinuities before more labels. |
| Training error is low but validation error is high | More representative data, coverage gaps, capacity, or regularization. |
| Errors cluster around breakaway/stopping/reversal | Independent reference checks, targeted training examples, and possibly a regime-aware representation. |
| Both curves are still improving at the step cap | More optimization budget or an explicit learning-rate schedule; dataset size is not yet isolated as the cause. |
| One-step accuracy is good but rollouts drift | State-distribution shift and accumulated error; inspect rollout horizons and revisit the training distribution. |

Targeted sampling may buy more accuracy per label than adding uniform random samples. Treat it as a separate comparison with a fixed evaluation distribution, count every new labeled training example, and do not copy validation/test cases into training. Sophisticated active learning is unnecessary until simple diagnostics show why it would help.

Alternative paths remain open: a physics-informed model can encode known structure, or a residual model can learn corrections when there is a real discrepancy to correct. Since the reference already computes this task analytically, the current MLP's immediate value is learning and a demonstrable experiment; any speed or deployment advantage needs measurement.

## 9. Minimal experiment record

Use one run directory, a configuration record, a CSV of metrics, and saved checkpoints. No external tracking service is needed initially.

Record architecture and parameter count; data hashes and subset row identities; random seeds; normalization statistics; optimizer, loss, learning rate and batch size; software versions and device; stopping budget; physical tolerances and coverage rule; update, epoch and exposure counts; elapsed time; validation loss and physical metrics; chosen and first-passing checkpoints; and failure/termination reason.

Include group definitions and counts, per-group pass rates, and the paired early-error diagnostic from section 2.3. If early stopping is enabled, record its metric, minimum improvement, patience, and earliest eligible update. Record target achievement separately from the reason execution ended: a fixed-budget pilot can reach a target and continue to its cap.

Save model weights, model configuration, and preprocessing together. Save optimizer state and random-generator state if exact continuation is required. Aggregate epoch losses by example count so a small final batch is not over-weighted.

Final comparison: error versus distinct labels, error versus updates/time, and a compact table of accuracy, parameter count, and cost. Include difficult-case results separately from overall averages. For repeated predictions, keep force schedules identical, track absolute position externally, and flag states that leave the trained input range; reference-state resets and free rollouts are different evaluation conditions.

## 10. Sources and next decision

The workflow follows established supervised-learning practice while keeping choices tied to this experiment:

- [PyTorch: Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html): forward prediction, loss, gradient clearing, backpropagation, optimizer updates, and gradient-free evaluation.
- [Scikit-learn: Learning curves](https://scikit-learn.org/stable/modules/learning_curve.html): training and validation performance versus number of training examples, and the bias introduced by tuning on validation.
- [Google Research: Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook): simple initial configurations, scientific versus nuisance hyperparameters, and resource-aware comparisons.

Next decision: confirm which joint per-example tolerance milestone and coverage will define target achievement; average loss remains a separate progress and checkpoint-selection measure. Choose numerical plateau-stopping settings after observing the pilot. Sariel's first implementation milestone is loading one batch and demonstrating the normalization round trip, before writing the full learning cycle.
