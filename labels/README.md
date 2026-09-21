# Cuboid labels v1

Status as of 2026-09-21: these files describe the preserved three-input dataset. The new [four-input v2 dataset](v2/README.md) is available in `labels/v2/`, with separate force duration and observation time and observations after force removal. Use v2 for the current network. The original generator and the command below still generate v1.

10,000 synthetic examples: 8,000 training, 1,000 validation, 1,000 test.

The dataset is stored in the repository-root `labels/` directory: `train.csv`, `validation.csv`, `test.csv`, and `metadata.json`.

Inputs, in order: `v0_m_s`, `force_N`, `duration_s`.
Labels, in order: `displacement_m`, `v_final_m_s`.
The physics function returns velocity first; the generator deliberately reorders its outputs to match the cuboid guide.

Force is constant throughout the supplied duration (0.1–1 s), and prediction ends at that time: `tr = tf = duration_s`. This follows the previous three-input variable-duration contract. There is no additional coast interval. Mass is 1 kg; friction and gravity come from `physics_formula.py`.

Initial velocity ranges from -10 to 10 m/s; force ranges from -20 to 20 N. These are initial experimental choices, not validated limits for physical hardware. Each split contains 60% broad uniform samples and 10% each of rest, zero force, near static-threshold, and near stopping-time samples. Sampling categories describe how inputs are chosen, not mutually exclusive physical outcomes. Exact duplicate inputs are excluded across all splits.

From the repository root, regenerate with:

```powershell
python cuboid_translation_x/label_preparation.py --output-dir labels
```

The command overwrites the generated CSV and metadata files in `labels/`. The explicit `--output-dir labels` is required because the generator's default remains `data/cuboid_translation_x/v1`. Choose another `--output-dir` for a separate dataset and use `--seed` to change the random seed. With unchanged code and the same Python runtime and seed, generation is reproducible. `metadata.json` records parameters, sampling details, column order, and source/data hashes.

All values are unnormalized and stored in physical units. Fit normalization on training data only. Reserve the test split for final evaluation. These labels measure agreement with the supplied physics, not accuracy against real experiments. No training code is included.
