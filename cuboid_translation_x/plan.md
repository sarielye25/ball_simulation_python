#  Stage One: Neural Network Training

## Goal of the simulation

Predict the signed displacement and final velocity of a cuboid at observation time t, given its current signed velocity v0, applied force F, and force duration t1. Motion is one dimensional along the x-axis.

The starting status of cuboid is unknown, it could be static or moving, with velocity given.

The push duration is a supplied input that varies from 0.1 s to 1 s between examples. Apply constant force F until t1, then zero applied force until observation at t, with 0 <= t1 <= t. Friction acts throughout. Both times are measured from the start of the example. Specify the observation-time range before generating the revised labels, and include both t = t1 and t > t1.

Environmental variables, such as gravity, coefficients of kinematic friction & static friction are fixed.

The input variables, in order, are (v0, F, t1, t), and the outputs are (d, v). Input units are m/s, N, s, s; output units are m, m/s.

## Network and controlled experiments

The chosen baseline is 4 → 32 → 32 → 2, with ReLU hidden layers and unrestricted linear outputs. Compare one versus two hidden layers and 16 versus 32 neurons per hidden layer using all four combinations. Keep the data, normalization, loss, and training setup fixed for architecture comparisons; compare loss choices separately. Judge accuracy through displacement and velocity errors in physical units, tolerance pass fractions, and motion-category failures on held-out data. See the [Transmodel Exploration Plan](../transmodel_exploration_plan.md) for controls, repeated seeds, and completion evidence.

## Preparation of labels
Labels should cover all possible situations. 

The four-input v2 labels are produced by `label_preparation_v2.py` using `physics_formula.py` and stored in `labels/v2/`. They include coasting and rest after force removal, with observation up to 2 s after the push ends. See the [v2 dataset README](../labels/v2/README.md) for sampling, verification scope, and regeneration instructions. The original three-input v1 labels and generator remain available separately.
