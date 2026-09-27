# Training outputs and checkpoints

Each call to `training_records.create_run_directory()` creates a timestamped run under `config.training_data_dir`, with empty `checkpoints/` and `figures/` directories. No artificial training results are created.

```text
training_data/<timestamp>/
    evaluations.csv
    baselines.csv
    checkpoints/
        update_000000.pt
        update_000100.pt
        index.json
    figures/
```

The checkpoint subdirectory is configured by `config.checkpoint_subdirectory`. Each .pt file stores model parameters, normalization, configuration, metrics, CPU random states, and optional optimizer, scheduler and continuation states. The JSON index tracks best validation MSE, best selected-ruler pass rate, last snapshot and explicitly selected final model. Pass-rate ties use lower MSE; exact ties retain the earlier snapshot.

Use one writer per run. Existing snapshots cannot be overwritten. Rollback-and-retry requires a new run. An interruption between checkpoint and index writes may leave an unindexed file; preserve and reconcile it before resuming.

## Checkpoint API

```python
from checkpoints import save_checkpoint, select_checkpoint, load_checkpoint

path = save_checkpoint(
    run_directory, update, model, validation_result,
    normalization={"input_mean": input_mean, "input_scale": input_scale,
                   "target_mean": target_mean, "target_scale": target_scale},
    model_config={"class": "transmodel", "widths": [4, 32, 32, 2]},
    run_config=recorded_config, train_result=training_result,
    optimizer=optimizer, scheduler=scheduler, continuation=continuation,
)
selected = select_checkpoint(run_directory, "best_mse", reason="NO_PROGRESS")
saved = load_checkpoint(selected, model)
```

The future training loop supplies these variables. On target achievement, select the current snapshot filename with reason TARGET_REACHED. Recording both best metrics does not change the existing first-target/lowest-MSE fallback selection rule.

Loading defaults to evaluation mode. Use the returned normalization for inference. To resume, construct the matching model, optimizer and scheduler, pass the latter two and `restore_rng=True` to load_checkpoint, restore the returned continuation metadata, and call model.train(). Continuation must contain epoch, shuffled permutation, next batch position, dedicated generator state if used, stopping state and sample exposures. Weights alone cannot reproduce a training trajectory.

See [learning-rate protocol](../learning_rate_protocol.md): PyTorch reduction patience is **200 optimizer updates**; early stopping remains **500 updates**. The checkpoint module stores scheduler state; scheduler execution awaits training-loop integration.

## Evaluation records and plots

Create the run and save training baselines once. At each evaluation, use the same unchanged model for the full training and validation sets, then call `save_evaluation(run_directory, update, split, result)` for each split. Baselines come from metrics.baseline_metrics(); model results come from metrics.evaluate(). Supply groups and physical inputs to include the three motion groups and breakaway subset. Avoid duplicate observations on resumption.

Run `python report.py training_data/<timestamp>` to generate PNG and SVG plots. Plotting requires Matplotlib; evaluation and checkpoints require NumPy and PyTorch.

Plots have paired training/validation panels and matched axes. Training shows five model curves plus two baselines; validation shows five model curves. CSV pass rates use 0-1; plots use percentages. Empty groups remain undefined and create gaps. Points retain actual update spacing without smoothing.

Ten plots cover total standardized MSE, each output's standardized MSE, two loss contributions, two physical-error P95 values and three joint pass rates. Under the equal-weight loss, each output contribution is half its MSE. These describe loss values, not gradients, and do not change training weights.

Training-loop integration and real training remain pending.
