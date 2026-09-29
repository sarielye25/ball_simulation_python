# Cuboid Motion Simulation Model — Progress & Next Steps

Learn to predict one-dimensional cuboid motion: `[initial velocity, force, push duration, observation time] → [displacement, final velocity]`. Mass and friction stay fixed.

## Done

- Implemented push-and-coast physics; analytical spot checks passed.
- Prepared 10,000 examples: 8,000 training / 1,000 validation / 1,000 test.
- Defined the 4 → 32 → 32 → 2 neural network.
- Added checkpoint support; planned training, evaluation, and learning-rate control.
- Built a software for demonstrating the result: loss curve, data.

## For more controlled experiments

1. Optimize the code.
2. Varify the structure of the neural network, for example, use one layer or more than two to compare the training efficiency.
3. Propose a new system for evaluation and conduct the same training.
4. Based on the results of different groups during first training, optimize a training strategy, e.g, specify group.
5. Introduce more complex physical situations.

## Industrial Directions

1. github
2. Gemini Deep Research
3. Jacky Dai
4. youtube
5. google scholar
6. search for more entreprises

Full physics coverage and trained-model evaluation remain pending. Synthetic 1D agreement does not establish real-world accuracy.

**Learning rule:** Sariel writes code; the tutor explains, reviews, and gives hints. One experiment at a time.

