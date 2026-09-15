# Learning Notes — transmodel

Last updated: 2026-09-15 15:34:24 +08:00 (Asia/Hong_Kong)

These Q&A notes summarize the discussion of `transmodel_learning_guide.pdf`, especially Lesson 02, “Define the prediction contract.” The timestamp records this revision; update it whenever these notes are revised.

## Q: What is my current goal for this project?

**A:** Train a learned world model named **transmodel** to predict the cuboid's motion over a specified time interval: displacement during the interval and velocity at its end. Here, `d` means signed displacement, not total distance travelled or absolute position.

For now, finish studying the PDF and leave the existing implementation unchanged. Start with fixed physical conditions, establish accurate predictions through evaluation, and then introduce additional varying conditions gradually.

## Q: Does reducing the number of input variables improve prediction accuracy? Aren't three or four inputs easy for a GPU?

**A:** Three or four inputs are computationally tiny. Fewer inputs do not automatically mean better accuracy. Starting with fixed conditions mainly simplifies data collection, evaluation, and diagnosis of mistakes.

The important distinction is between fixing a physical condition and hiding a condition that varies:

| Experiment | Meaning for the model |
|---|---|
| Mass stays fixed and is omitted from the inputs | The model learns motion for that particular mass. |
| Mass varies and is provided as an input | The model can learn how motion depends on mass, with adequate training coverage. |
| Mass varies but is hidden | Identical visible inputs may require different predictions. |
| Mass is supplied but has only one value throughout training | The data do not teach how changing mass affects motion. |

Fixing mass or friction does not remove its physical effect. The model learns a relationship specific to those conditions. More varied conditions require adequate examples of their effects and combinations; a faster GPU cannot replace missing information or training coverage.

## Q: Why can a missing variable make prediction inaccurate?

**A:** A missing variable matters when it varies between cases, changes the required prediction enough to matter, and cannot be inferred from the available inputs or history.

For a cuboid already sliding positively without stopping during the interval:

\[
a = \frac{F}{m} - \mu_k g.
\]

With force 10 N, kinetic friction coefficient 0.2, and gravity 9.8 m/s², acceleration is 3.04 m/s² for a 2 kg cuboid and 0.54 m/s² for a 4 kg cuboid. The same current velocity and force therefore produce different future velocities. A predictor that receives neither mass nor information sufficient to infer its effect cannot uniquely distinguish those cases.

The effect of an omission is not always large. Its importance depends on the operating range and required accuracy.

## Q: Why is target position unnecessary when current position and velocity matter?

**A:** Current position is where the cuboid is now. Next position is where physics takes it. Target position is where we want it to go.

A controller uses the goal and current state to choose a force. The world model predicts motion under the supplied force. If two experiments have identical initial states, physical conditions, and force schedules, changing the desired destination does not change their motion.

“Under a supplied force” means that the action throughout the prediction interval is specified. If a controller changes the force during that interval in response to a target, the force schedule can differ and so can the motion. The first experiment avoids this ambiguity by holding force constant throughout the interval.

## Q: Does current position have to be an input to transmodel?

**A:** On a uniform floor with no boundaries or position-dependent effects, absolute position does not affect displacement or velocity change. A learned function can predict those changes from velocity and force under fixed physical conditions. Current position is then added to predicted displacement to obtain absolute next position.

Moving an entire experiment 100 m to the right should add 100 m to its predicted next position while leaving displacement and velocity unchanged. This assumption fails if position changes the physics, such as near walls or across different surfaces.

## Q: How does the proposed world model differ from the current physics calculation?

**A:** `cuboid_translation_x/physics.py` calculates total displacement from a constant positive push starting at rest, followed by coasting until rest. `simulation.py` uses that displacement to update the stopping position.

Transmodel's proposed task is to predict displacement and velocity after a specified interval from a possibly moving initial state. The cuboid need not have stopped when that interval ends. The existing stopping calculation is a useful analytical case, but does not provide arbitrary fixed-interval transitions.

Simple cases can already be calculated quickly using mechanics. A neural network has not yet demonstrated a speed or accuracy advantage here. The learning goal is to train and evaluate a predictor against a physics reference we understand.

## Q: Should an industrial robot's world model include as many variables as possible?

**A:** It should contain enough relevant information to meet its prediction requirements within its intended operating conditions. Maximizing the input count is not the objective.

Variable mass and surface friction may matter substantially. Paint colour usually does not affect this mechanical prediction. Air resistance may be negligible at the chosen speeds and tolerances. Some unmeasured conditions can be estimated from motion history rather than supplied directly.

Useful inputs can improve prediction, while irrelevant or noisy inputs can introduce misleading correlations and additional data requirements. Ask: “Could this missing information change the answer enough to matter for this task?” A 1 cm stopping tolerance and a 10 μm tolerance can require different models of the same object.

## Q: Is including every possible variable an industrial standard?

**A:** No universal industrial standard prescribes that approach or requires a learned world model. Requirements and applicable standards depend on the robot and application.

A common engineering approach is to define operating conditions and acceptable errors, identify influential factors, measure or estimate them where appropriate, account for uncertainty, and validate representative and difficult cases.

For example, a robot may receive a known payload mass, estimate its effect from measurements, or use control that tolerates a specified mass range. Industrial systems inevitably face unmeasured influences such as wear, disturbances, and sensor errors.

Industrial readiness requires more than prediction accuracy: timing, control stability, fault handling, and behavior outside supported conditions also matter. Agreement with simulator-generated labels alone does not establish accuracy on physical hardware.

## Q: What is the equivalent issue in an LLM?

**A:** The closest analogy is whether the available context contains enough information to determine a useful answer. Asking for the acceleration of a block under 10 N without specifying mass and friction does not determine one numerical answer.

An LLM can explain the dependencies, ask for information, or state assumptions. General training cannot reliably reveal the mass of a particular block that has not been described. Additional context or tools can supply missing facts, just as motion history can help estimate hidden conditions in a dynamics problem.

More context is not automatically better: irrelevant or contradictory information can make the task harder. A typical LLM predicts tokens, whereas transmodel is intended to predict physical quantities. Fluent text alone does not establish reliable physical prediction.

## Q: What is an “unclearly defined physical setup”? Does it only mean hidden variables?

**A:** Hidden varying conditions are one possibility, but ambiguity can also concern the prediction task itself:

- Unspecified horizon: predict velocity after 0.1 s, after 1 s, or after the push ends?
- Unspecified action schedule: hold 10 N throughout the interval or release it halfway?
- Ambiguous output: does `d` mean signed displacement, distance travelled, or absolute position?
- Unspecified mechanics: can the cuboid tip or encounter a wall?
- Unrecorded changing conditions: does mass or friction change between experiments?

A hidden variable does not necessarily make the problem unclear. “Mass is unknown, lies between 1 and 3 kg, and recent motion history is supplied to help estimate its effect” is a clear problem involving incomplete information.

## Q: What is a clear first prediction contract for transmodel?

**A:** Given current velocity and a force held constant throughout a specified interval, predict signed displacement during that interval and velocity at its end, for a fixed mass on a horizontal uniform surface with fixed friction and gravity, constrained to one-dimensional motion without tipping or impacts.

Before collecting training data, specify the interval, numerical physical conditions, allowed velocity and force ranges, and resting/stopping rules. Evaluate displacement and velocity errors separately in physical units, and check repeated predictions as well as one-step predictions.
