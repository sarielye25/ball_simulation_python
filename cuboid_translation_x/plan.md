# Cuboid Translation Along X — Next Steps

## Objective and learning approach

Build a verified one-dimensional simulation, control a cuboid so it stops at a target, and then train an RL policy. Work on one milestone at a time. Sariel writes the code; the assistant explains concepts and reviews attempts. Provide implementation only when explicitly requested.

## Current foundation and scope

- A cuboid with dimensions, mass, center position, orientation, local vertices, and a static 3D display.
- An analytical function predicts displacement from rest after a constant positive push followed by coasting to rest.
- Use SI units: meters, kilograms, seconds, and newtons.
- The cuboid and horizontal floor are rigid. Only x changes; y, height, and orientation remain fixed.
- Assume uniform density, so the center height is half the cuboid height above the floor at z = 0.
- Use ideal static and kinetic friction, ignoring air resistance and contact deformation.
- The analytical calculation requires positive mass and kinetic friction, nonnegative push duration, and static friction at least as large as kinetic friction.

## 1. Connect one experiment — immediate next task

In simulation.py, import the cuboid class and physics function. Create one cuboid and define a positive force, push duration, and friction coefficients. Call the analytical calculation and add its displacement to the initial x coordinate to obtain the predicted stopping position.

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

## 6. Introduce reinforcement learning

Wrap the verified simulator in Gymnasium with reset and step operations. Define observations, bounded continuous actions, reward, success termination, and time-limit truncation. Begin with signed target distance and current velocity as observations, keeping physical parameters fixed.

Connect a Stable-Baselines3 algorithm such as PPO. Vary initial positions and targets during training. Compare the policy with the reference controller on identical held-out conditions using success rate, final error, settling time, and failure cases.

Defer the learned world model until the reference simulator and policy evaluation work reliably, then revisit the parent project's learning plan.

## File responsibilities

| File | Responsibility |
|---|---|
| cuboid_setting.py | Cuboid properties and state |
| physics.py | Analytical reference and later motion updates |
| simulation.py | Experiment settings and coordination |
| visualization.py | Plots and 3D display, when separated |

Add environment, training, and evaluation files only when reaching RL.

## Python environment

Use one deliberately selected interpreter consistently. The existing D:/5.Physical_AI/mujoco_learning/.venv environment contains NumPy and Matplotlib and was used to check the display. A new terminal does not create a virtual environment. Install additional dependencies into the selected environment only when needed.
