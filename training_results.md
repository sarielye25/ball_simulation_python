# Training Results

This file records diagnostic analysis of each run of each training protocol.
The diagnosis format is:

```text 
Status: success/failure [if failure, add stop reason]
SUN: [stopping_update_number]
Observation: [what i see from the data]
PR: [possible reason for results] 
OD: [optimzation direction, both from human and ai]
```

## First Training Protocol


### Run 20260929_142936_371616
```text
Status: failure [ no_progress ]
SUN: 20
Observation: Absolute losses of both v and d prediction are big. Though the loss curve decreases rapidly, the pass rate still gets low. I saw the standardized mse getting below 0.001, but at 20th update, still 7997 samples failed as intermediate level, which is strange.
PR: 
OD: 1. check qt demonstration system, verify the results.
```

#### Verified audit (2026-09-29)

Status: failed the configured accuracy target; training completed normally with `NO_PROGRESS`.
SUN: 2000 optimizer updates (the observation above refers to the 20th regular evaluation).
Selected model: update 1900, which had the lowest validation MSE.

| Checkpoint | Split | Standardized MSE | Displacement MAE (m) | Velocity MAE (m/s) | Intermediate passes |
|---|---|---:|---:|---:|---:|
| 1900 (selected) | Train | 0.007320414 | 0.467125 | 0.329413 | 2 / 8000 |
| 1900 (selected) | Validation | 0.007486291 | 0.464186 | 0.346675 | 1 / 1000 |
| 2000 (last) | Train | 0.007267157 | 0.463188 | 0.328794 | 3 / 8000 |
| 2000 (last) | Validation | 0.007499035 | 0.461257 | 0.347382 | 0 / 1000 |

Intermediate requires BOTH displacement error < 0.01 m and velocity error < 0.01 m/s for each sample. Success requires at least 95% passing in both splits. The observed 7997 failures at update 2000 is correct. Even coarse accuracy at that update is only 281/8000 training and 37/1000 validation samples.

Verification: reloaded all 24 saved checkpoints and reproduced all 216,000 published predictions across training and validation exactly. Independently recomputed overall MSE from physical predictions and saved target scales; the largest difference from logged MSE was 3.02e-10. All three tolerance pass counts matched the viewer and training logs at every checkpoint. Viewer curve values matched the logs. All 9000 saved labels matched the current physics function exactly; this checks consistency with the generator, not an independent proof of its physics. Reserved test data was not evaluated. Reproduction: `.venv\Scripts\python.exe audit_first_training.py`.

Graph diagnosis:

- Neither overall training nor validation standardized MSE ever fell below 0.001. Their minima were 0.007267157 and 0.007486291 respectively.
- An offscreen execution of the Qt window confirmed the graph's last training point is 0.007267157. Its automatic linear vertical range was approximately -0.045 to 1.096, making late errors appear almost zero. The numerical curve is correct; its scale makes late performance difficult to judge.
- Standardized MSE is dimensionless: mean(((predicted displacement - target displacement)/7.800004)^2 + ((predicted velocity - target velocity)/4.786902)^2)/2. Small values do not directly imply centimeter accuracy. At update 2000, training RMSE is 0.632257 m and 0.427185 m/s. Even MSE 0.001 would correspond to approximately 0.247 m and 0.151 m/s RMSE if both normalized output MSEs equaled 0.001.
- The graph always includes the entire history, while the table follows the selected checkpoint. On load, the viewer selects update 1900 and correctly displays 7998 failures; the graph still ends at 2000. There is no marker linking the selected checkpoint to its graph point.
- `completed` describes normal process termination, not achieving the accuracy target. The current status line does not show the stop reason or target achievement.
- Table `error_` columns are signed prediction-minus-target errors; the pass calculation uses absolute errors. Prediction and target columns are correctly ordered and matched by row ID.
- Minor code inconsistency: the viewer uses <= tolerance while training uses < tolerance. It changes none of this run's pass counts.

PR (evidence and possible causes):

1. The optimizer learned substantial structure: training MSE fell from 1.04394 to 0.00727 (about 99.3%). Residual physical errors nevertheless remain tens of times larger than the intermediate tolerances. Failure to meet the target does not mean learning failed completely.
2. Early stopping uses an absolute `min_delta=0.001`. The last qualifying progress reference was validation MSE 0.008097347 at update 1500. Later evaluations needed to fall below 0.007097347 to reset patience. Update 1900 improved to 0.007486291, but this improvement did not qualify. At update 2000, 500 updates had elapsed, so stopping worked as configured. `NO_PROGRESS` therefore means insufficient improvement under this threshold, not literally no improvement.
3. The learning-rate scheduler uses the same absolute threshold and reduced Adam's learning rate at updates 1400, 1700, and 1900: 0.001 -> 0.0005 -> 0.00025 -> 0.000125. This combination can limit continued refinement; a controlled follow-up is needed to establish how much accuracy it costs.
4. Training and validation errors are similar, providing no strong evidence that overfitting is the main limitation. Optimization limits and model approximation error are more plausible. This run alone cannot distinguish them.
5. A 4->32->32->2 ReLU network must approximate products of inputs and transitions among rest, motion, and stopping. It has no mechanism enforcing exact zero output in resting regimes. At update 2000, none of the 561 always-resting training rows passes intermediate accuracy; their MAEs are 0.522 m and 0.285 m/s. Errors also affect moving and stopped groups, so the problem is not confined to one boundary.
6. Standardization balances output scales, but average squared error does not directly optimize the fraction of samples satisfying two strict physical tolerances. Large-error samples can dominate learning while many small physical errors still fail the accuracy criterion.

OD:

1. Improve display clarity: show selected-checkpoint MSE, physical MAEs, both pass rates and tolerances, termination reason, and a checkpoint marker; offer a logarithmic loss axis or a late-update zoom. Align the tolerance comparison operators.
2. Next controlled experiment: keep seed, data, architecture, optimizer, and update budget fixed; lower the shared absolute improvement threshold (for example, to 1e-5), then compare physical errors and pass rates at equal update counts. This is a diagnostic starting value, not a guarantee of reaching 95%.
3. If refinement still plateaus, test model capacity or physics-informed features/regime handling in separate experiments. Preserve the test split until the model and procedure have been selected.

No training configuration, original run artifacts, or viewer code was changed during this audit.
