# Transmodel — Exploration Plan

Updated: 2026-09-23.

Goal: train a small neural network to predict the cuboid's motion, then learn what improves its accuracy.

Sariel writes the code. Sary explains, reviews, and helps interpret results. Work on one experiment at a time.

## Where we are

- **Physics implemented:** `physics_formula.py` predicts displacement and final velocity over a push-and-coast interval. It already supports the prediction task; it does not need replacing with a new simulator. Analytical spot checks have passed, but full case coverage is not established.
- **Data prepared:** `labels/` contains 10,000 four-input examples, split into 8,000 training, 1,000 validation, and 1,000 test examples. Ranges and generation checks are recorded in [labels/README.md](labels/README.md).
- **Network defined:** `neural_network.py` implements 4 → 32 → 32 → 2, with ReLU hidden layers and linear outputs.
- **Next:** write `training_loop.py`, which currently contains only comments. Training and model evaluation remain to be done.

## Prediction task

**Inputs:** `[v0, F, t1, t]` → **outputs:** `[d, v]`.

Starting at velocity `v0`, apply constant force `F` until `t1`, then coast until observation time `t`. Friction acts throughout; `0 <= t1 <= t`. Mass and friction coefficients stay fixed. Position is tracked separately by adding displacement `d`.

The physics function returns `(v, d)`; the dataset and network use `[d, v]`.

## Now — train and evaluate the baseline

1. Load the existing splits and prepare batches. Fit any data-based normalization on training data only.
2. Train the existing network with scaled squared error and a PyTorch optimizer. Understand prediction → loss → gradients → parameter update.
3. Track training and validation loss. Report displacement error in metres and velocity error in m/s, including average and worst errors. Inspect rest, stopping, and reversal cases; independently check questionable physics labels.
4. Choose acceptable displacement and velocity errors in the evaluation setup. Save the model, settings, and results. Use validation data for decisions; reserve test data for final evaluation.

Finish this stage with a trained model and a clear account of where its predictions work or fail.

## Then — three focused experiments

1. **Architecture:** compare one versus two hidden layers and 16 versus 32 neurons per layer. Keep data, loss, and training settings comparable.
2. **Loss:** compare squared error with MAE or Huber, or adjust output weighting if results justify it. Judge using the same physical error metrics, not raw loss values across different losses.
3. **Repeated predictions:** compare feeding back the model's own predicted state with resetting both position and velocity to the physics reference each interval. Use the same prescribed forces and horizons; plot position and velocity errors over time.

Record what changed, what happened, and what to try next. Control, rotation, and language inputs remain later directions.
