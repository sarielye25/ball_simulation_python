# Cuboid Motion — Progress & Next Steps

Learn to predict one-dimensional cuboid motion: `[initial velocity, force, push duration, observation time] → [displacement, final velocity]`. Mass and friction stay fixed.

## Done

- Implemented push-and-coast physics; analytical spot checks passed.
- Prepared 10,000 examples: 8,000 training / 1,000 validation / 1,000 test.
- Defined the 4 → 32 → 32 → 2 neural network.
- Added checkpoint support; planned training, evaluation, and learning-rate control.

## Next

1. **Check:** verify physics cases, data, normalization, and a small training run.
2. **Train:** integrate the loop, Adam, scaled MSE, and learning-rate reductions after stalled progress.
3. **Evaluate:** target ≥95% joint displacement/velocity passes on training and validation at one chosen tolerance. Stop after 500 updates without meaningful improvement or 10,000 total.
4. **Record:** save checkpoints, learning curves, physical errors, and failures. Test only after freezing model selection.
5. **Demonstrate:** animate reference versus learned motion and training progress in 3D.
6. **Compare:** vary network size, dataset size, and loss; test repeated predictions with/without reference-state feedback.
7. **Explore later:** feedback control, optional RL, rotation, language instructions, and physical validation.

Full physics coverage and trained-model evaluation remain pending. Synthetic 1D agreement does not establish real-world accuracy.

**Learning rule:** Sariel writes code; the tutor explains, reviews, and gives hints. One experiment at a time.

Details: [Training guide](transmodel_training_plan_updated.html) · [Further exploration](transmodel_further_exploration_plan.md) · [Dataset](labels/README.md)
