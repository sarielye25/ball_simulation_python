# Cuboid labels v2 — four-input push and coast

Diagnostic subset of the synthetic examples generated using `physics_formula.motion_cal`: 160 training and 40 validation examples sampled across the original files. The 1,000 reserved test examples are unchanged and are not used for diagnostic training or shown in Qt.

Inputs in order: `v0_m_s`, `force_N`, `force_duration_s`, `observation_time_s`.
Targets in order: `displacement_m`, `v_final_m_s`.

These four inputs match the current network. The physics function returns velocity first; the generator reorders the targets to displacement first. Values are unnormalized; fit normalization using training data only.

Apply constant force from time zero until `force_duration_s`, then zero applied force until `observation_time_s`. Friction acts throughout. Times are relative to the start of the example. Mass is 1 kg, gravity is 9.81 m/s², static friction coefficient is 0.4, and kinetic friction coefficient is 0.3.

Initial velocity ranges from -10 to 10 m/s; force ranges from -20 to 20 N; push duration ranges from 0.1 to 1 s. Time after removal ranges from 0 to 2 s, so observation time ranges from 0.1 to 3 s subject to `t1 <= t <= t1 + 2`. The two-second coast range is an initial experimental choice, not a physical limit or a guarantee that every example has stopped.

The original files were generated with 40% broad uniform samples; 10% each from rest, zero force, near static threshold, near stopping during the push, and exactly at force removal; 5% near stopping during coast; and 5% at rest after coast. This diagnostic subset takes evenly spaced rows from the original train and validation files, so those exact percentages are not guaranteed. Categories describe sampling and may overlap in physical outcome. Repeated `(v0, F, t1)` conditions are excluded across and within splits.

From the repository root, generate another copy in a new directory:

```powershell
python cuboid_translation_x/label_preparation_v2.py --output-dir labels/v2_regenerated
```

The default destination is `labels/v2`. Existing dataset files are never overwritten. The default seed is 20260921; use `--seed` for a different sample. The generator reuses the original generator's input sampling and calls the existing physics function with separate force-removal and observation times. `metadata.json` records the contract, sampling, split hashes, and all three source hashes. The original v1 files remain in `labels/`.

The source dataset passed generation checks for finite values, ranges, target ordering, reference targets, condition separation, and reproducibility. The reduced train and validation files have new row counts and hashes in `metadata.json`; they are diagnostic subsets, not byte-identical regenerated source files. This dataset measures agreement with the reference model, not real hardware. Reserve test data for final model evaluation.
