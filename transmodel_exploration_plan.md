# Transmodel — Exploration Plan

Updated: 2026-09-19.

## Direction and current task

Current learning task: understand and train a small neural network with acceptable one-step motion predictions. Next, compare repeated predictions with and without reference-state feedback. Along the way, investigate how to judge whether a loss function serves the prediction task well.

Work on one milestone at a time. Sariel writes the implementation; the assistant explains concepts, reviews attempts, and helps interpret evidence. Start with a small working experiment and expand when results justify it. This plan does not imply that the reference simulator or trained network is already complete.

## 1. Establish the prediction contract and reference

Use the initial candidate contract: current signed velocity and applied force predict signed displacement and final velocity over a fixed interval. Keep mass and friction coefficients fixed, and hold force constant within each interval. Absolute position is tracked externally; it is not needed as a network input on the uniform floor. Specify the interval, operating ranges, units, and friction assumptions before generating data.

Build and verify a deterministic physics function that advances arbitrary current position and velocity through one interval. The existing push-and-coast stopping-distance calculation is not yet this function. Check rest below the static-friction threshold, starting, sliding in both directions, coasting, stopping inside an interval, and restarting or reversing under an appropriate force. Friction alone must not reverse motion. Use hand-calculated cases and time-step convergence where numerical integration is used: determinism alone does not establish correctness.

Choose separate displacement, position, and velocity acceptance tolerances based on what motion matters for this experiment. Define a velocity threshold for judging an endpoint approximately stationary. These evaluation tolerances are not automatically the loss scaling constants. No numerical tolerances have been agreed yet.

Completion evidence: a written contract and reproducible, independently checked reference cases. Verified transitions are required before trustworthy training data; a controller and animation are not prerequisites.

## 2. Train and qualify the first one-step network

Generate input–answer pairs covering rest, sliding, stopping, and motion boundaries. Split complete trajectories or experiment conditions into training, validation, and test sets, keeping neighboring transitions together. Use validation results to make choices and reserve test data for final evaluation. Fit data-derived normalization on training data only.

Start with the proposed two-input network, one hidden layer of 16 ReLU neurons, and two unrestricted linear outputs. Use scaled squared error as the initial loss. PyTorch supplies automatic differentiation and standard optimizers; understand the prediction, loss, gradient, and update sequence before implementing it. Record the optimizer, learning rate, training budget, and random seed.

Evaluate predictions from true states against the reference and simple baselines such as constant velocity. Report displacement and final-velocity errors separately in physical units: mean absolute error, 95th-percentile error, maximum observed error, and the fraction meeting both chosen tolerances. Inspect motion categories separately so easy cases cannot hide stopping failures.

Define the required pass fraction and any critical-case limits before judging qualification. If the model fails, inspect reference labels, coverage, scaling, and training before assuming it needs more layers.

Completion evidence: a saved model, configuration, and validation report showing which one-step criteria pass or fail. Confirm the selected model on untouched test data.

## 3. Judge the loss using common evaluation criteria

Question: which training penalty produces the most useful predictions for this task? A smaller reported training loss is insufficient evidence. MSE, MAE, and Huber have different numerical meanings; simply multiplying a loss by a constant changes its reported value.

Keep a common evaluation scorecard independent of the training loss:

| Criterion | Evidence |
|---|---|
| One-step accuracy | Displacement and velocity errors in physical units and tolerance pass fraction |
| Motion behavior | Starting/stopping errors and endpoint stationary/sliding mismatches using a fixed threshold |
| Longer-horizon usefulness | Position and velocity errors over the matched horizons in milestone 4 |
| Practical training | Stability, training time, and variability across repeated runs |

Begin with squared error. Choose one focused comparison based on an observed issue: change displacement-versus-velocity weighting if their trade-off is wrong, or compare a different loss shape such as MAE or Huber. Large errors near genuine stopping events are not automatically noisy outliers to suppress. Leave multi-step training until rollout results justify trying it.

For the first controlled comparison, hold data splits, architecture, normalization, batch size, optimizer, learning rate, and training budget fixed; change only the selected loss choice. Retrain each candidate from matched starting weights and data order for each seed, using a small repeated set of seeds where practical. Compare validation results and category-specific failures. This tests losses under that setup, not their universal superiority. If tuning is needed, give candidates the same small tuning budget and document changes.

Prefer the candidate that meets the acceptance criteria more reliably without unacceptable trade-offs. Record improvements in displacement versus regressions in velocity or stopping explicitly. Select using validation data, then evaluate the final selection on reserved test data. Once test results influence revisions, that set is development evidence; use a fresh held-out set for a new final assessment.

Completion evidence: a compact comparison table and a reasoned choice. Run the initial one-step comparison here and add rollout evidence after milestone 4. Do not wait for an exhaustive search for the “best” loss before proceeding.

## 4. Controlled experiment: repeated prediction with and without feedback

Freeze one trained model. Use identical initial conditions, fixed physical parameters, prediction interval, horizon, and prescribed force sequence for all paths. Maintain an independent physics-reference trajectory. At each step, compare predictions with the reference at the same next time.

| Path | State used before each prediction | Next position |
|---|---|---|
| Physics reference | Previous reference state | Deterministic physics update |
| Without state feedback | Previous predicted velocity | Previous predicted position plus predicted displacement |
| With full reference-state feedback | Current reference velocity | Current reference position plus predicted displacement |

Both model paths also receive force and start from the same true initial state. Reset position as well as velocity in the feedback path: feeding back true velocity while continuing to sum predicted displacements is a different, partial-feedback experiment that can retain position drift.

Plot reference and predicted position/velocity against time, plus error against time. Report errors at selected horizons, tolerance pass fractions, first tolerance crossing when present, and stationary/sliding mismatches. Include rest, persistent sliding, stopping, and force changes at interval boundaries. Repeat across held-out initial conditions and force schedules; keep extrapolation results separate.

Hypothesis: full state feedback reduces propagation of earlier prediction errors. Test this rather than assuming errors always accumulate or grow monotonically. Individual one-step errors remain. The reference provides ideal observations, not evidence that real sensors are perfect.

This is a prediction-feedback experiment, not yet closed-loop control: forces remain prescribed and identical. The reference trajectory is not driven by predictions. Full-feedback results measure repeated local predictions anchored to reality, not an independent long-horizon forecast.

Completion evidence: matched trajectory/error plots and an explanation of when feedback helps, where errors remain, and whether the hypothesis holds. Extend the loss scorecard with these results.

## 5. Choose the next experiment from the evidence

- If useful one-step predictions drift during rollouts, compare one-step training with multi-step loss. Multi-step training feeds predictions forward and penalizes subsequent trajectory errors; keep evaluation consistent.
- If exact feedback helps, test partial feedback, less frequent observations, or specified sensor noise and delay, changing one factor at a time.
- For actual closed-loop control, use observed state and predicted candidate outcomes to choose bounded forces toward a target. Execute in the reference simulator, measure again, and replan. Compare with a simple controller on identical targets and initial conditions; actions may now differ between controllers. Judge target accuracy, final speed, settling duration, and success rate.

Rotation and language-conditioned models remain later directions in the [parent plan](README.md). They are not required to complete this exploration.

## Record learning and results

Use the existing paper notebook for hypotheses, expectations before running experiments, and explanations of surprising results. Save configurations, data splits, model versions, metrics, and plots with the project for reproducibility. For each milestone record what was tested, what happened, what was learned, and the next justified change. A small demonstrable experiment and an honest report are the intended portfolio output.
