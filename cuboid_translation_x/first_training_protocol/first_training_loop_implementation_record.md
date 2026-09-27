# First Training Loop Implementation Record

Cuboid Translation Model

## Purpose

This document records how I design and implement the first training loop. For each file, it states the file's responsibility, how I divide that responsibility into smaller coding tasks, and which tools I choose for those tasks. It is a learning record and a decision log; it does not replace the code or the coding manual.

## Working method

- Work through the files in implementation order.
- Break one responsibility into small pieces before writing code.
- Record a tool only after understanding why it fits the task.
- Keep confirmed decisions separate from provisional plans.
- Update this record when a design choice changes.

## Document status

| Field | Current entry |
| --- | --- |
| Project stage | Evaluation metrics and plotting interfaces implemented; training integration pending |
| Current file | `metrics.py`, `training_records.py`, `report.py` |
| Current coding piece | Record the confirmed plotting design and training output layout |
| Last updated | 27 September 2026 |

## File Map

The first ten files form the initial training and evidence pipeline. The final two files are deferred until the first training result has been reviewed.

| Order | File | Responsibility | Status |
| --- | --- | --- | --- |
| 01 | `config.py` | Protocol and run settings | In progress |
| 02 | `data.py` | Load, validate, group, and scale data | Current |
| 03 | `neural_network.py` | Define the prediction network | Existing file to review |
| 04 | `metrics.py` | Evaluate predictions and groups; D/V loss decomposition | Implemented |
| 05 | `checkpoints.py` | Save and restore training state | Planned |
| 06 | `stopping.py` | Apply stopping decisions | Planned |
| 07 | `checks.py` | Run preflight correctness checks | Planned |
| 08 | `training_loop.py` | Orchestrate training and evaluation | Planned |
| 09 | `report.py` | Create evidence and diagnosis | Curve plotting implemented; other reports planned |
| Support | `training_records.py` | Save evaluation and baseline CSV files | Implemented; training-loop integration pending |
| 10 | `final_test.py` | Evaluate the frozen final choice | Planned |
| 11 | `animate.py` | Demonstrate the first model | Deferred |
| 12 | `compare.py` | Run controlled comparisons | Deferred |

## Current training folder structure (2026-09-27)

The tree below distinguishes existing files from outputs generated during a run. `labels/` contains input label data; `training_data/` stores training statistics and figures separately.

```text
first_training_protocol/
├── config.py                         Training and evaluation settings
├── cuboid_setting.py                 Physical environment settings
├── physics_formula.py                Analytical physics reference
├── label_preparation_v2.py           Label generation
├── labels/                           Existing labels, README, and metadata
├── data.py                           Loading, validation, groups, target scaling and inverse scaling
├── neural_network.py                 Network architecture
├── metrics.py                        Evaluation: overall, groups, breakaway, and baselines
├── training_records.py               Evaluation/baseline CSV storage and separate run directories
├── report.py                         Read CSV files and plot with Matplotlib
├── training_data/
│   ├── README.md                     Recording interfaces, metrics, and plotting instructions
│   └── <timestamp>/                  Created per run; no actual training results exist yet
│       ├── evaluations.csv           Metrics by evaluation update, split, and sample scope
│       ├── baselines.csv             Two training baselines calculated once
│       └── figures/                  Created when plotting; PNG and SVG files for each metric
├── transmodel_training_plan_updated.html       Current general training plan
├── tm_first_training_coding_manual_with_responsibilities.pdf
├── first_training_loop_implementation_record.md  This technical and decision record
└── __pycache__/                      Automatically generated Python cache
```

`training_loop.py`, `checkpoints.py`, `stopping.py`, `checks.py`, and `final_test.py` have not yet been implemented in this folder. `animate.py` and `compare.py` are deferred. The File Map above includes these planned files; it does not imply that they already exist.

## 01 `config.py`

**Status:** In progress. The protocol settings exist; open choices should remain explicit.

**Responsibility:** Keep experiment settings in one place so other files read the same protocol rather than repeating values.

### Task breakdown

- Record the model dimensions, optimizer settings, batch size, and maximum updates.
- Record evaluation times, stopping rules, tolerance rulers, and the random seed.
- Record input and target column order.
- Record data and run locations after those locations are deliberately chosen.
- Provide one seeding operation that applies the recorded seed consistently.

### Tools and why they are used

| Tool | Use in this file |
| --- | --- |
| Python dataclasses | Group related settings into a clear configuration object. |
| `pathlib.Path` | Represent data and output paths without manual string handling. |
| `random`, NumPy, PyTorch | Apply the same recorded seed to each random-number system used by the run. |

### Decision record

- Confirmed batch size: 128 rows for a normal training batch.
- Confirmed optimizer: Adam with learning rate 0.001 and zero weight decay.
- The stopping ruler, minimum progress amount, seed, and paths must be recorded before a run.

## 02 `data.py`

**Status:** Current learning task. No implementation has been delegated; the code will be written piece by piece.

**Responsibility:** Turn CSV label files into trustworthy model data while preserving row identity and reference-group metadata. Fit data-derived scaling statistics on training data only.

### Task breakdown

- Label pipeline: open one CSV split and read each row by its column names.
- Convert the six required numeric fields from strings to floating-point values.
- Separate each row into four inputs `[v0, F, t1, t]` and two targets `[d, v]`.
- Assign a stable row ID and retain the split name.
- Validate required columns, finite values, valid times, and final array shapes.
- Attach one reference group to each row: always resting, moved then stopped, or moving at observation.
- Fit input and target means and standard deviations using training rows only.
- Transform all splits with the training statistics and support conversion back to physical units.
- Expose the prepared rows to PyTorch while keeping IDs and groups aligned during shuffling.

### Tools and why they are used

| Tool | Use in this file |
| --- | --- |
| `csv.DictReader` | Read local CSV rows by column name with explicit control over conversion and validation. |
| `float` | Convert CSV text values into numbers before numerical work. |
| NumPy | Store rectangular numerical arrays, check shapes and finite values, and calculate column statistics. |
| `hashlib.sha256` | Record which exact source file was loaded when reproducibility evidence is needed. |
| `torch.utils.data.Dataset` | Present prepared examples to a PyTorch DataLoader later in the pipeline. |

### Decision record

- The built-in `csv` module is selected for the first loading step because the table is simple and explicit row handling supports the learning goal.
- The choice between `csv` and pandas depends on the kind of table operations required, not on whether the source is local or supplied by an API.
- Reference groups will be derived in `data.py`. They are metadata used for evaluation, not inputs or prediction targets.
- Standard deviation is the initial data-derived scale. Statistics are fitted on training data only and then fixed.
- For a normal batch, `N = 128` in shapes `[N, 4]` and `[N, 2]`. The last batch may be smaller because `drop_last` is false.

## 03 `neural_network.py`

**Status:** Existing file to review after `data.py`.

**Responsibility:** Define the function that maps four standardized inputs to two standardized predictions.

### Task breakdown

- Define a PyTorch module.
- Build linear layers with widths 4 to 32 to 32 to 2.
- Apply ReLU after each hidden linear layer.
- Implement the forward calculation.
- Check CPU `float32` predictions for shape `[N, 2]` and finite values.

### Tools and why they are used

| Tool | Use in this file |
| --- | --- |
| `torch.nn.Module` | Create a trainable model with registered parameters. |
| `nn.Linear` | Learn affine transformations between layer widths. |
| `nn.ReLU` | Introduce nonlinear hidden responses. |
| `nn.Sequential` | Express the layer order clearly. |

### Decision record

- Tool choices are provisional until this file is studied.

## 04 `metrics.py`

**Status:** Implemented, including separate standardized D/V MSE and their contributions to the current equal-weight loss.

**Responsibility:** Measure full training and validation performance using standardized loss, physical errors, tolerance rulers, groups, and simple baselines.

### Task breakdown

- Run the unchanged model over a complete split.
- Aggregate squared errors by sample count.
- Convert predictions back to physical units.
- Calculate joint pass rates for all three rulers.
- Calculate overall, group, and boundary-slice diagnostics.
- Calculate update-zero baselines and retain required per-row errors.

### Tools and why they are used

| Tool | Use in this file |
| --- | --- |
| `torch.inference_mode` | Evaluate without constructing gradient history. |
| NumPy | Calculate percentiles, masks, and aggregated diagnostics. |
| Boolean masks | Select reference groups and the breakaway slice without changing the data. |

### Decision record

- Tool choices are provisional until this file is studied.

## 05 `checkpoints.py`

**Status:** Implemented; training-loop integration remains pending. See `learning_rate_protocol.md` for the confirmed scheduler/stopping amendment.

**Responsibility:** Preserve model states and the information required to reproduce predictions or continue a run.

### Task breakdown

- Save every scheduled evaluation checkpoint.
- Track the absolute lowest validation MSE independently from stopping progress.
- Save final and selected checkpoints.
- Restore model, optimizer, normalization, and random states.
- Verify that a reloaded checkpoint reproduces predictions.

### Tools and why they are used

| Tool | Use in this file |
| --- | --- |
| `torch.save` and `torch.load` | Serialize and restore PyTorch states. |
| `state_dict` | Store model and optimizer parameters explicitly. |
| `json` | Store readable checkpoint metadata and run records. |
| RNG state APIs | Continue a run without silently changing its random sequence. |

### Decision record

- Confirmed: immutable PyTorch snapshots, atomic writes, and a JSON index with separate MSE/pass-rate rankings and explicit final selection. CPU RNG, Adam and scheduler states are supported; the caller restores sampler and stopping metadata.
- Checked prediction equality, RNG restoration, exact next Adam update, scheduler reduction after restoration, ranking/tie rules, final selection and rejection of duplicate updates/non-finite MSE using temporary synthetic data.
- Output paths are configured in `config.py`; run directories contain `checkpoints/` and `figures/`. See `training_data/README.md`.
- Use PyTorch ReduceLROnPlateau with 200-update reduction patience (100-update cadence, PyTorch patience=1), alongside unchanged 500-update early stopping. The future training loop will execute this policy; see `learning_rate_protocol.md`.

## 06 `stopping.py`

**Status:** Implemented; training-loop integration remains pending.

**Responsibility:** Turn evaluation results into ordered target, progress, patience, and update-cap decisions.

### Task breakdown

- Represent the stopping state and possible outcomes.
- Check numerical validity first.
- Leave absolute-best checkpoint tracking to `checkpoints.py`.
- Check the selected-ruler target on both train and validation sets.
- Update the progress reference only after a meaningful improvement.
- Apply patience and maximum-update rules in the specified order.

### Tools and why they are used

| Tool | Use in this file |
| --- | --- |
| dataclasses | Store the evolving stopping state clearly. |
| `Enum` | Represent stopping outcomes without fragile free-form strings. |
| State machine logic | Make the required decision order explicit and testable. |

### Decision record

- `StopReason` names outcomes; `StopState` remembers progress MSE/update and the last checked update; immutable `StopDecision` returns the reason, selection policy and limit flags; `check_stop()` validates input and performs the ordered checks while updating state in place.
- Initialize at update zero; check the selected-ruler full-training/full-validation target before progress, patience and cap. Non-finite results abort separately without changing state. Both patience/cap flags are retained when they coincide, with NO_PROGRESS primary.
- The caller maps selection `current` to the just-saved filename, or passes `best_mse` to `select_checkpoint()`. Numerical failure and continuing decisions do not select a final model.
- Save state via `dataclasses.asdict(state)` inside checkpoint continuation metadata; restore via `StopState(**saved_state)`. The scheduler has its own counters. Confirmed cooldown is zero, reduction patience 200 updates, and early-stopping patience 500 updates.

## 07 `checks.py`

**Status:** Planned.

**Responsibility:** Detect data, scaling, model, and learning errors before beginning the real run.

### Task breakdown

- Check data columns, shapes, groups, and reference cases.
- Check training-only scaling and inverse recovery.
- Check that one update changes weights with finite loss and gradients.
- Check that a fresh model can reduce error on a small fixed set.
- Check important stopping-rule boundary cases.

### Tools and why they are used

| Tool | Use in this file |
| --- | --- |
| `unittest` | Organize repeatable preflight checks using the standard library. |
| `numpy.testing` | Compare arrays with explicit floating-point tolerances. |
| `torch.isfinite` | Detect invalid tensors during learning checks. |
| Tensor cloning | Compare parameters before and after an update. |

### Decision record

- Tool choices are provisional until this file is studied.

## 08 `training_loop.py`

**Status:** Planned existing file to edit.

**Responsibility:** Coordinate setup, mini-batch updates, scheduled full-set evaluation, checkpoints, stopping, and evidence logging.

### Task breakdown

- Seed the run and load training and validation data.
- Fit training scalers and create a fresh model and optimizer.
- Evaluate update zero before any gradient update.
- Perform one mini-batch update at a time and count actual row exposures.
- Evaluate on the fixed schedule and at normal termination.
- Invoke checkpoint and stopping decisions in their required order.
- Record losses, metrics, groups, time, and termination evidence.

### Tools and why they are used

| Tool | Use in this file |
| --- | --- |
| `torch.optim.Adam` | Apply the selected parameter-update algorithm. |
| `DataLoader` | Shuffle rows and construct batches while retaining the final partial batch. |
| `time.perf_counter` | Measure elapsed run time with a monotonic clock. |
| `csv.DictWriter` | Write structured evaluation history in a readable format. |

### Decision record

- Tool choices are provisional until this file is studied.

## 09 `report.py`

**Status:** Curve plotting implemented and checked with temporary sample data. Training-loop integration, failure tables and full run diagnosis remain planned.

**Responsibility:** Turn saved run evidence into curves, failure tables, a factual summary, and a focused diagnosis.

### Task breakdown

- Plot validation MSE and matched train and validation pass rates.
- Plot full-validation group results with group counts.
- Create failure tables for selected and last valid checkpoints.
- Record representative and worst physical-unit errors.
- Summarize configuration, data identity, normalization, runtime, and termination.

### Tools and why they are used

| Tool | Use in this file |
| --- | --- |
| Matplotlib (`subplots`, `plot`, `axhline`, `savefig`) | Confirmed: paired train/validation plots, evaluation markers, fixed baseline lines, matched axes, PNG/SVG export. |
| `csv` | Read the structured history produced by the training loop. |
| NumPy | Calculate failure rankings and descriptive statistics. |
| `json` | Write a structured run summary. |

### Decision record

- Matplotlib is confirmed for plotting; CSV keeps the data reusable in Python or MATLAB without retraining.

### Confirmed plotting design (2026-09-27)

- Use actual parameter update counts on the x-axis. Each evaluation supplies one point; connect adjacent observations at their actual spacing without smoothing.
- Create two side-by-side panels per metric: training on the left and validation on the right. Match axis scales, show numerical y-axis labels on both panels, and use consistent colors for each sample scope across figures.
- Show seven training lines: overall, always resting, moved then stopped, moving at observation, the breakaway subset, the zero baseline, and the constant-velocity baseline. Calculate the two baselines once over the full training set and display them as horizontal dashed lines.
- Show five validation lines: overall, the three motion groups, and breakaway. Validation baselines are omitted as requested.
- Emphasize the overall line, use a dash-dot line for breakaway, and show sample counts in legends. Empty groups return `None`, produce blank CSV cells, and leave gaps in plots rather than being replaced with zero. Breakaway is an overlapping diagnostic subset with no separate acceptance gate.
- Produce ten plot types: total standardized MSE, D standardized MSE, V standardized MSE, D loss contribution, V loss contribution, displacement P95 absolute error, velocity P95 absolute error, and joint pass rates under coarse/intermediate/fine tolerances. Store pass rates as 0–1 in CSV and display them as 0–100%. P95 is neither MSE nor a confidence interval.
- The current loss averages the two standardized outputs equally. Each D/V contribution is half its corresponding standardized MSE; the contributions sum to total MSE. This decomposes the loss value, not gradient contributions, and does not automatically change training weights. Use these plots to inspect output differences and inform later controlled experiments.

### Data recording and plotting responsibilities

`metrics.py` returns dictionaries and arrays for the current evaluation. `training_records.py` uses the standard-library `csv`, `pathlib.Path`, and `datetime` modules to create separate run directories and save results. The future training loop will evaluate the full training and validation sets using the same model state, then call the recording interfaces. `report.py` reads saved files to generate plots.

Each row in `evaluations.csv` identifies an update, split, predictor, and scope, and stores sample counts, error metrics, loss contributions, and all three pass rates. Scopes include overall, the three motion groups, and breakaway. `baselines.csv` stores the two training baselines separately. Pass `groups` and `physical_inputs` to each evaluation to obtain all five scopes. Training-log fields such as epoch, sample exposures, and elapsed time still need to be added when implementing the training loop.

Use a separate `training_data/<timestamp>/` directory for each run to avoid mixing experiments. Run `python report.py training_data/<timestamp>` to save plots under that directory's `figures/` folder. Temporary sample data has been used to check loss decomposition, empty groups, CSV reading/writing, all ten plot types, seven training lines versus five validation lines, and matched axes. A sample layout was also inspected; these checks do not represent actual training results.

## 10 `final_test.py`

**Status:** Planned. Execute only after the training protocol and selected checkpoint are frozen.

**Responsibility:** Evaluate the frozen model choice once on the reserved test split and keep those results separate from validation decisions.

### Task breakdown

- Load the selected checkpoint and training-fitted normalization.
- Open the reserved test split only after choices are frozen.
- Reuse the same data and metric definitions.
- Write test results separately and state the limited synthetic one-dimensional scope.

### Tools and why they are used

| Tool | Use in this file |
| --- | --- |
| `data.py` | Load and transform test rows using training statistics. |
| `metrics.py` | Apply the same evaluation definitions used for validation. |
| `checkpoints.py` | Restore the selected frozen model. |
| `torch.inference_mode` | Evaluate without gradient tracking. |

### Decision record

- Tool choices are provisional until this file is studied.

## 11 `animate.py`

**Status:** Deferred until after the first training result is reviewed.

**Responsibility:** Demonstrate the first trained model beside the analytical reference using its saved checkpoints and normalization.

### Task breakdown

- Review the first model result before choosing the demonstration design.
- Load recorded model progress and normalization.
- Calculate matching analytical-reference motion.
- Animate the reviewed comparison design.

### Tools and why they are used

| Tool | Use in this file |
| --- | --- |
| Matplotlib `FuncAnimation` | Provisional tool for a time-based visual demonstration. |
| PyTorch `state_dict` | Load the first model's saved states. |
| Analytical reference | Provide the comparison trajectory. |

### Decision record

- Implementation and final tool choices remain deliberately deferred.

## 12 `compare.py`

**Status:** Deferred until after the first-model demonstration.

**Responsibility:** Run controlled comparisons in which one main factor changes while the remaining data and protocol stay matched.

### Task breakdown

- Compare the specified network architectures.
- Compare nested training-set sizes with fresh models and subset-specific normalization.
- Define any loss comparison only after its candidate losses are chosen.
- Treat multi-step rollouts as a separate study with explicit position tracking.

### Tools and why they are used

| Tool | Use in this file |
| --- | --- |
| `dataclasses.replace` | Create controlled configuration variants. |
| NumPy indexing | Select reproducible nested data subsets. |
| Existing pipeline modules | Reuse training, evaluation, and reporting definitions. |
| Matplotlib | Compare measured outcomes visually. |

### Decision record

- Implementation and final tool choices remain deliberately deferred.

## Update Template

Use this structure whenever a new coding piece is planned or completed.

| Record field | Entry |
| --- | --- |
| File | |
| Responsibility | |
| Small coding piece | |
| Inputs and outputs | |
| Chosen tool | |
| Reason for tool choice | |
| What I learned | |
| Checks performed | |
| Next piece | |
