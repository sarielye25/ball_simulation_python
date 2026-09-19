# Cuboid Translation Along X — Next Steps

## Objective and learning approach

Complete Phase One of the [parent learning plan](../README.md): build a verified one-dimensional simulation and control a cuboid so it stops at a target. RL is an optional later extension, not the next required phase. Work on one milestone at a time. Sariel writes all code; the assistant explains concepts and reviews attempts without generating code, replacement code, patches, or copyable pseudocode. Documentation changes are allowed when requested.

## Current foundation and scope

- A cuboid with dimensions, mass, center position, orientation, local vertices, and a static 3D display.
- A candidate analytical function attempts to predict displacement from rest after a constant positive push followed by coasting to rest. Its expression is not yet verified.
- Use SI units: meters, kilograms, seconds, and newtons.
- The cuboid and horizontal floor are rigid. Only x changes; y, height, and orientation remain fixed.
- Assume uniform density, so the center height is half the cuboid height above the floor at z = 0.
- Use ideal static and kinetic friction, ignoring air resistance and contact deformation.
- The analytical calculation requires positive mass and kinetic friction, nonnegative push duration, and static friction at least as large as kinetic friction.

## Current priority — transmodel exploration

As of 2026-09-19, the active learning task is the neural network and its one-step accuracy, followed by comparing repeated prediction with and without reference-state feedback. Follow the [Transmodel Exploration Plan](../transmodel_exploration_plan.md) for the full sequence and loss evaluation. The physics milestones below remain prerequisites and supporting work; animation and a feedback controller need not precede model training.

## 1. Verify and connect one experiment

First explain the resting, pushing-while-sliding, and coasting stages. Independently derive the displacement and check each term's units. Review which friction coefficient applies during sliding, the displacement associated with constant acceleration, and the grouping of the coasting expression. The current function must pass these checks before serving as a reference.

Then design one experiment using the existing cuboid and physics concepts: choose a positive force, push duration, and friction coefficients, and relate displacement to the predicted final center position. Sariel decides how to express this in Python.

Verify a known numerical example and a push below the static-friction threshold. Confirm that y, height, and orientation stay unchanged. No time loop is needed yet.

## 2. Compare with a target

Define a target x coordinate and report signed position error, consistently using final x minus target x. Try forces and push durations manually to observe an exact match, an undershoot, and an overshoot.

The analytical result assumes the object has fully stopped. Later, position alone will not be enough to determine success.

## 3. Simulate motion over time

Add linear velocity and a physics update that advances position and velocity over a small time step. Hold the force during the push interval, then release it. Handle resting, sliding, and stopping, including transitions within a step.

Verify that the stopping position approaches the analytical prediction as the time step decreases. A weak push must leave a resting cuboid stationary, and friction must not reverse motion by itself.

## 4. Plot and animate

Record time, position, velocity, and acceleration. Plot them before adding animation. Move display logic into visualization.py when useful. Animate the recorded center positions while preserving dimensions and orientation.

Check that the display agrees with the data: acceleration during pushing, slowing after release, and persistent rest after stopping.

## 5. Build a feedback controller

Extend and verify the physics for signed forces and already-moving states. Build a simple controller that repeatedly observes target error and velocity and chooses a bounded force. Specify the control interval and distinguish it from the physics time step if multiple physics steps occur per action.

Define position and speed tolerances, a required settling duration, and a maximum episode duration. Test targets in both directions and multiple initial conditions. Keep this controller as a reference for RL evaluation.

## 6. Record Phase One evidence and continue

Keep the experiment settings, SI units, assumptions, numerical tolerances, and results together. Record analytical comparisons, time-step convergence, resting/stopping checks, and feedback-control outcomes on held-out conditions. Gradually turn verified cases into repeatable regression checks.

Once motion updates are verified, record time, state, action, next state, time step, and physical parameters for reproducible experiments and potential later world-model datasets. No learned model is needed to complete this phase.

Continue to the parent plan's Phase Two by training a dedicated translation world model. Its first task is to predict displacement and final velocity from current velocity and applied force over a fixed interval, tracking absolute position externally. Keep mass and friction fixed in the simulator; friction need not be a model input or output. Prescribed pushes can supply training data without a policy agent. Friction estimation is optional; rotation follows in Phase Three.

## Optional later extension — reinforcement learning

Wrap the verified simulator in Gymnasium with reset and step operations. Define observations, bounded continuous actions, reward, success termination, and time-limit truncation. Begin with signed target distance and current velocity as observations, keeping physical parameters fixed.

Connect a Stable-Baselines3 algorithm such as PPO. Vary initial positions and targets during training. Compare the policy with the reference controller on identical held-out conditions using success rate, final error, settling time, and failure cases.

Use the parent plan for learned-model experiments. A verified simulator and evaluation procedure are prerequisites; a trained RL policy is not.

## File responsibilities

| File | Responsibility |
|---|---|
| cuboid_setting.py | Cuboid properties and state |
| physics.py | Analytical reference and later motion updates |
| simulation.py | Experiment settings and coordination |
| visualization.py | Plots and 3D display, when separated |

Add world-model data, training, and evaluation files when reaching Phase Two. Add an RL environment and policy-training files only if pursuing the optional RL extension.

## Python environment

Use one deliberately selected interpreter consistently. The existing D:/5.Physical_AI/mujoco_learning/.venv environment contains NumPy and Matplotlib and was used to check the display. A new terminal does not create a virtual environment. Install additional dependencies into the selected environment only when needed.
