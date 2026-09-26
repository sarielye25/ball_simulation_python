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
| Project stage | Beginning the data pipeline |
| Current file | `data.py` |
| Current coding piece | Open one CSV split and read rows by column name |
| Last updated | 26 September 2026 |

## File Map

The first ten files form the initial training and evidence pipeline. The final two files are deferred until the first training result has been reviewed.

| Order | File | Responsibility | Status |
| --- | --- | --- | --- |
| 01 | `config.py` | Protocol and run settings | In progress |
| 02 | `data.py` | Load, validate, group, and scale data | Current |
| 03 | `neural_network.py` | Define the prediction network | Existing file to review |
| 04 | `metrics.py` | Evaluate predictions and groups | Planned |
| 05 | `checkpoints.py` | Save and restore training state | Planned |
| 06 | `stopping.py` | Apply stopping decisions | Planned |
| 07 | `checks.py` | Run preflight correctness checks | Planned |
| 08 | `training_loop.py` | Orchestrate training and evaluation | Planned |
| 09 | `report.py` | Create evidence and diagnosis | Planned |
| 10 | `final_test.py` | Evaluate the frozen final choice | Planned |
| 11 | `animate.py` | Demonstrate the first model | Deferred |
| 12 | `compare.py` | Run controlled comparisons | Deferred |

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

**Status:** Planned.

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

**Status:** Planned.

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

- Tool choices are provisional until this file is studied.

## 06 `stopping.py`

**Status:** Planned.

**Responsibility:** Turn evaluation results into ordered target, progress, patience, and update-cap decisions.

### Task breakdown

- Represent the stopping state and possible outcomes.
- Check numerical validity first.
- Recognize a new absolute best checkpoint.
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

- Tool choices are provisional until this file is studied.

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

**Status:** Planned.

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
| Matplotlib | Create training and evaluation curves. |
| `csv` | Read the structured history produced by the training loop. |
| NumPy | Calculate failure rankings and descriptive statistics. |
| `json` | Write a structured run summary. |

### Decision record

- Tool choices are provisional until this file is studied.

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
