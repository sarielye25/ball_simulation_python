# Cuboid Simulation — Learning Plan

Learner: Sariel Ye, third-year mechanical engineering student at SJTU SPEIT. Beginner in independent coding, interested in physical AI. Prefers English, simple methods, and learning by doing.

Goal: Improve Python ability and understanding of mechanics, simulation, control, and learned world models through a small cuboid's movement. Gradually develop the evidence and engineering practices needed for a simulator that is reliable for a defined industrial task.

Industrial readiness is application-specific. More degrees of freedom, RL, or a learned world model do not establish it. The eventual application must define acceptable prediction error, operating conditions, runtime, and consequences of failure. This project begins as a learning simulator; physical validation and application requirements come later.

## Instructions for the AI tutor
- I write all code manually. Do not generate code, replacement code, patches, or copyable pseudocode.
- Explain concepts, suggest focused resources, and give small learning challenges.
- Give directions for learning—not instructions specifying which lines to add or change.
- When I struggle, ask questions and offer progressively stronger conceptual hints.
- Review my attempts without implementing fixes. Help me check correctness and explain results.
- Work on one milestone at a time, building on what I already understand.
- Change the learning plan when requested, but leave implementation to me.
- Require evidence of understanding and correctness before moving to the next milestone. Introduce engineering practices gradually through the experiment at hand.

## Project idea and current position

Move a cuboid to a target and stop there, first through physical understanding and manual experiments, then feedback control, and later learning where useful.

- Reference simulator: Calculates motion using explicitly coded physical rules and remains available for verification and comparisons.
- Learned world model: Predicts future state from state and action using data. Physics simulators are also predictive models of the world; here, “world model” means the learned version.
- Controller or policy: Chooses actions to accomplish the task. Learning dynamics and learning actions are different problems; neither requires the other.

Start with explicit position and velocity. Images and learned latent states can wait.

Current repository: `cuboid_translation_x/cuboid_setting.py` defines geometry and a static display; `physics.py` contains a candidate analytical stopping-distance calculation; `simulation.py` is a placeholder. There is no verified time-stepped simulator, controller, or trained model yet. The analytical expression needs review before it can serve as a reference, especially its sliding-friction choice, acceleration contribution, and coasting-term units.

## Crucial future exploration — Language and world models

Sariel has chosen language understanding as a crucial future vertical exploration direction: build toward a world model that can use human-language descriptions or instructions alongside physical state information to predict motion. Curiosity and the appeal of this capability are part of the motivation.

The first transmodel focuses on numerical motion prediction. That provides a foundation for later exploring how language connects to physical actions and predicted outcomes, for example interpreting “give the cuboid a gentle push.” Such phrases need context or explicit conventions to determine quantities such as force, direction, and duration.

Keep open both connecting a trained language component to the numerical predictor and training a predictor that directly uses a language representation. Choose the approach through future experiments; the architecture is not decided yet. Evaluate whether the system interprets instructions correctly and predicts the resulting motion accurately, including on unfamiliar wording. Language understanding and physical prediction are distinct capabilities to develop and test together.

## Why this sequence

The central problem is building a predictor and controller whose results we can explain and trust. Mechanics gives us simple cases with independently calculable answers; numerical experiments then expose approximation errors. Learning becomes meaningful once those reference cases and evaluation methods exist.

| Assumption | Assessment | Decision |
|---|---|---|
| A learned model must replace physical laws to advance | It can introduce accumulated prediction error and fail outside its training data | Keep the reference simulator; compare learned and hybrid alternatives |
| RL must come before learning dynamics | Dynamics can be learned from manual or scripted experiments | Make RL optional, after feedback control |
| More complex motion means industrial quality | Complexity without validation makes errors harder to isolate | Advance through explicit acceptance checks |
| Simulator data proves real-world accuracy | It only demonstrates agreement with that simulator | Add independent physical measurements before real-world claims |

## Simulation Phases

### Phase One — Verified translation along x

Restrict motion to x, with fixed orientation and a fixed force application arrangement. Treat this as a constrained one-dimensional model, with no tipping. Use explicit physical laws in `physics.py`.

Milestones:
1. Explain the assumptions, units, and push/coast/rest stages. Independently derive and check the analytical stopping distance before connecting one experiment.
2. Compare stopping position with a target through manual force and duration choices.
3. Build time-stepped motion and verify it against analytical cases, resting thresholds, stopping behavior, and decreasing time steps.
4. Plot trajectories and then animate them; check the display against numerical results.
5. Extend to signed forces and initial velocities, then use feedback to reach and remain at a target with bounded force.

Exit evidence: reproducible experiments, analytical agreement within a stated tolerance, a time-step convergence study, no friction-driven reversal, and controller results across held-out initial conditions. Define success using both position and speed tolerances, a settling duration, and a timeout.

World-model role: no learned dynamics yet. Learn what state, action, and prediction horizon mean. Once the simulator is verified, save time, state, applied force, next state, time step, and physical parameters so experiments can later become useful data.

### Phase Two — Train a translation world model

Train a dedicated learned predictor for the verified one-dimensional task before introducing rotation. Keep the reference simulator and put the learned model alongside it. The initial transmodel takes current signed velocity and applied force and predicts displacement and final velocity over one fixed interval. Track absolute position externally on the uniform floor.

The [Transmodel Exploration Plan](transmodel_exploration_plan.md) details the current sequence: qualify one-step predictions, compare losses using common task metrics, then test repeated prediction with and without reference-state feedback. Verified physics transitions are prerequisites for training; completing animation or a controller first is not required.

Keep mass and static/kinetic friction coefficients fixed across the first dataset. Friction remains in the simulator but is not an input to the learned model; the model learns its effect on motion without needing to output a friction coefficient. Collect examples using prescribed pushes; no policy agent is required. Estimating friction through known equations is an optional comparison, not a prerequisite.

Milestones:
1. Specify the prediction task, state, action interval, fixed parameters, and operating range. Collect trajectories covering rest, motion, stopping, and varied actions. Hold each applied force constant over the prediction interval for the first experiment.
2. Split data by complete trajectories or experiment conditions into training, validation, and untouched test sets; avoid leakage through neighboring transitions. Keep extrapolation tests separate from in-range tests.
3. Establish simple prediction baselines, including unchanged-state and constant-velocity prediction, and retain the verified physics reference. Train a small state-transition model to predict the observed next state from state and action.
4. Measure one-step and repeated-rollout errors in position and velocity, with units and specified horizons. Inspect stopping behavior, physical consistency, and failures outside the training range.
5. Optionally compare the learned predictor with fitted physical parameters. Investigate a hybrid model that learns a correction to physics when an observed mismatch justifies it. If later varying friction or mass, provide those parameters as inputs or explicitly study inference from motion history; hidden varying parameters make the initial state/action inputs insufficient in general.
6. Test the model in a control task, for example using candidate-action rollouts for planning. Evaluate the resulting controller in the reference simulator, where model errors cannot hide behind self-consistent predictions.

Exit evidence: a reproducible dataset and evaluation report, held-out prediction and control results, measured runtime, and clearly stated limits. Simulator-trained models demonstrate agreement with the reference simulator, not with physical hardware.

Optional RL extension: train one policy in the reference simulator and a fresh policy using learned-model rollouts. Evaluate both and the feedback baseline on identical held-out reference-simulator conditions, over multiple seeds. Report success rate, target error, settling time, and failures. Track reference-simulator interactions for data collection and policy training separately from imagined interactions. The learned route does not have to win.

World-model role: a tested alternative for prediction, planning, or faster rollouts. Replacement of any component requires evidence of adequate accuracy, robustness, and speed for the intended use.

### Phase Three — Add rotation and extend the world model

First define the axis, allowed motion, and contact arrangement. “Rotation in x direction” is ambiguous: rotation is about an axis. For x-directed pushing with tipping in the x–z plane, the rotation axis is y; rotation about x describes a different experiment. Select and document the intended experiment before implementation.

Milestones:
1. Study a single rotational degree of freedom in a clearly constrained experiment, such as a fixed pivot, to isolate torque, inertia, angle, and angular velocity.
2. Verify simple torque and free-motion cases, coordinate conventions, and energy behavior appropriate to the constraints and friction.
3. If the goal is floor tipping or tumbling, introduce coupled translation, rotation, and changing contact in a separate step. Revisit center height, normal force, friction, and impact assumptions; these cannot simply remain the Phase One constants.
4. Compare matched simple cases with a mature rigid-body engine, such as MuJoCo. Document solver and contact-model differences rather than treating the engine as physical ground truth.
5. Extend the learned predictor's state and action description to the selected rotational task, collect new data, and repeat held-out one-step and rollout evaluation. Compare data needs and failure modes with the translation experiment.

Exit evidence: independently checked rotational cases, convergence checks, documented contact limitations, preservation of Phase One verification cases, and measured prediction errors for the extended learned model.

World-model role: expand a previously understood prediction experiment to richer mechanics. Verify each new physical behavior before using it to generate training data; preserve the physics reference for comparison.

### Phase Four — Validate for a defined engineering application

Choose a concrete use case, such as predicting the stopping position of a pushed part on a particular surface. Scope complexity around that task.

Milestones:
1. Write an operating envelope and acceptance criteria before final evaluation: geometry, loads, friction range, initial states, acceptable trajectory/control error, and runtime budget.
2. Gather physical measurements, document measurement uncertainty, calibrate parameters on one subset, and validate on independent experiments. Distinguish verification (solving the chosen equations correctly) from validation (representing the real task adequately).
3. Assess sensitivity and robustness to parameter variation, contact transitions, sensor noise, delay, and disturbances relevant to the application. Identify unsupported conditions and define fallback behavior before hardware control.
4. Package repeatable runs: versioned configurations and datasets, dependency records, automated regression checks and CI, logging, and evaluation reports. Profile runtime before optimizing or replacing a solver.
5. Review evidence against the application criteria and document remaining limitations. Adopt an established simulation engine when its capabilities serve the task; preserve independently verified benchmark cases.

World-model role: retain it only where measured results justify it, such as residual friction prediction or accelerated planning. Evaluate it on independent physical data and monitor prediction error when used. A physics-only simulator can meet the application requirements.

Exit evidence: a reproducible validation package showing which requirements pass, which fail, and the supported operating envelope. This is progress toward industrial use, not a claim of universal certification or production readiness.


## Current learning milestone

As of 2026-09-19, the current learning task is understanding and training transmodel for acceptable one-step accuracy, followed by the controlled state-feedback experiment. Follow the [Transmodel Exploration Plan](transmodel_exploration_plan.md), including its loss-comparison method and completion evidence.

The [Cuboid Translation Along X guide](cuboid_translation_x/plan.md) retains the physics prerequisites. A verified arbitrary-state transition function is still required before generating trustworthy training labels; the learning focus does not mean this implementation is complete.

The tutor offers conceptual hints and reviews Sariel's implementation without writing it. Each milestone ends with a small experiment and a short explanation of what works, what fails, and why.
