# Learning-rate control and snapshot protocol

Confirmed 2026-09-27. This amendment supersedes the earlier constant-learning-rate assumption in the plan and coding manual. Training-loop integration remains pending.

## Two patience rules

Use PyTorch `torch.optim.lr_scheduler.ReduceLROnPlateau` with Adam. Initial learning rate: 0.001; factor: 0.5; minimum: 0.00001; cooldown: zero scheduler checks (`cooldown=0`, confirmed by the user). This supersedes the earlier one-check cooldown in the plan, notes and coding-manual amendment. Monitor full-validation standardized MSE with `mode="min"`, `threshold_mode="abs"`, and `threshold=config.min_delta` (currently 0.001).

**Learning-rate reduction patience is 200 optimizer updates. Early-stopping patience remains 500 optimizer updates.** Call the scheduler at update 0 to establish its baseline, then at updates 100, 200, 300, etc. With this regular cadence, set PyTorch `patience=1`: it reduces on the second unsuccessful check, after 200 updates. `patience=2` would wait 300 updates, and `patience=200` would count checks rather than optimizer updates. Early evaluations at 10, 20 and 50 are still saved and used by the stop checker, but do not advance the scheduler.

If both progress references were last improved at update 1000, a continuous stall reduces the rate at 1200 and again at 1400, then can stop at 1500. This leaves 300 optimizer updates after the first reduction and 100 after the second, provided the rate is above its minimum and the run does not terminate earlier. Early irregular evaluations can move the stopping reference independently, so this aligned example is not a universal guarantee. Keep both references in the logs.

The scheduler and early stopper compete for the same training budget: stopping before or at a reduction gives the new rate no opportunity. A reduction alone never resets the 500-update stopping clock. A meaningful validation-MSE improvement does. Preserve the 10000-update cap and the first-target success rule. At a terminal check, do not schedule a reduction for training that will not run. Check finite metrics and rank snapshots; decide target/progress/patience/cap; on continuation, step the scheduler at its cadence; persist the resulting optimizer, scheduler and stopping state before the next batch update.

Reducing learning rate is an optimization attempt, not a diagnosis or cure for overfitting. An absolute threshold of 0.001 becomes relatively demanding as MSE shrinks; inspect real curves before revising it. Save every new absolute MSE minimum even if its gain is smaller than this threshold.

## Checkpoint responsibility and integration

`checkpoints.py` saves immutable `training_data/<run>/checkpoints/update_XXXXXX.pt` files with weights, model/run configuration, normalization, validation and optional training metrics, data identity, optional Adam and scheduler states, CPU random states, and caller-provided continuation metadata. `index.json` stores `best_mse`, `best_pass_rate`, `last`, and explicitly `selected` pointers. Highest pass rate uses the selected ruler and lower MSE as its tie-break; exact ties retain the earlier state. The two best criteria may identify different models. The manual's first-target/lowest-MSE fallback remains the final-selection policy until explicitly revised.

For exact CPU continuation, the future training loop must supply and restore epoch, current shuffled permutation, next batch position, any dedicated data-generator state, exposure count, and stopping state. Loading weights alone restores prediction behavior, not the training trajectory. Scheduler control belongs to the training loop; checkpoint saving does not autonomously train, reduce the rate or roll back. Rollback-and-retry requires a new run linked to its parent snapshot.

Record the learning rate used by updates, rate changes, and the next rate after each evaluation. Save scheduler state after its decision. Verify prediction round trips and Adam/scheduler continuation before training. The test split never participates in ranking or scheduling.

Source: https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
