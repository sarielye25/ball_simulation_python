import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random

import physics_formula as physics


MASS_KG = 1.0
VELOCITY_RANGE = (-10.0, 10.0)
FORCE_RANGE = (-20.0, 20.0)
DURATION_RANGE = (0.1, 1.0)
COLUMNS = ["v0_m_s", "force_N", "duration_s", "displacement_m", "v_final_m_s"]
CATEGORIES = {
    "uniform": 600,
    "from_rest": 100,
    "zero_force": 100,
    "static_threshold": 100,
    "stopping_boundary": 100,
}


def sample_inputs(rng, category):
    while True:
        velocity = rng.uniform(*VELOCITY_RANGE)
        force = rng.uniform(*FORCE_RANGE)
        duration = rng.uniform(*DURATION_RANGE)
        if category == "from_rest":
            velocity = 0.0
        elif category == "zero_force":
            force = 0.0
        elif category == "static_threshold":
            velocity = 0.0
            force = rng.choice((-1.0, 1.0)) * (
                physics.mu_s * MASS_KG * physics.g + rng.uniform(-0.2, 0.2)
            )
        elif category == "stopping_boundary":
            stopping_time = duration * rng.uniform(0.95, 1.05)
            force = MASS_KG * (
                math.copysign(physics.mu_k * physics.g, velocity)
                - velocity / stopping_time
            )
        if FORCE_RANGE[0] <= force <= FORCE_RANGE[1]:
            return velocity, force, duration


def generate_split(rng, multiplier, seen):
    rows = []
    counts = {}
    for category, base_count in CATEGORIES.items():
        counts[category] = base_count * multiplier
        for _ in range(counts[category]):
            inputs = sample_inputs(rng, category)
            while inputs in seen:
                inputs = sample_inputs(rng, category)
            seen.add(inputs)
            velocity, force, duration = inputs
            final_velocity, displacement = physics.motion_cal(
                force, MASS_KG, duration, duration, velocity
            )
            row = (*inputs, displacement, final_velocity)
            if not all(math.isfinite(value) for value in row):
                raise ValueError(f"Non-finite example: {row}")
            rows.append(row)
    rng.shuffle(rows)
    return rows, counts


def prepare_labels(output_dir, seed):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    seen = set()
    split_metadata = {}
    for split, multiplier in (("train", 8), ("validation", 1), ("test", 1)):
        rows, counts = generate_split(rng, multiplier, seen)
        path = output_dir / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        split_metadata[split] = {
            "file": path.name,
            "rows": len(rows),
            "sampling_category_counts": counts,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    metadata = {
        "dataset_version": "v1",
        "seed": seed,
        "total_rows": len(seen),
        "input_columns": COLUMNS[:3],
        "label_columns": COLUMNS[3:],
        "units": {
            "v0_m_s": "m/s",
            "force_N": "N",
            "duration_s": "s",
            "displacement_m": "m",
            "v_final_m_s": "m/s",
        },
        "prediction_contract": {
            "description": "Signed displacement and final velocity under constant force for the supplied duration.",
            "force_removal_time": "tr = duration_s",
            "prediction_time": "tf = duration_s",
            "coasting_after_force_removal": False,
            "initial_position": "Displacement is relative to the initial position.",
        },
        "fixed_parameters": {
            "mass_kg": MASS_KG,
            "gravity_m_s2": physics.g,
            "mu_s": physics.mu_s,
            "mu_k": physics.mu_k,
        },
        "input_ranges": {
            "v0_m_s": VELOCITY_RANGE,
            "force_N": FORCE_RANGE,
            "duration_s": DURATION_RANGE,
        },
        "sampling": {
            "uniform": "Independent uniform draws across all input ranges.",
            "from_rest": "Exactly zero initial velocity; uniform force and duration.",
            "zero_force": "Exactly zero force; uniform initial velocity and duration.",
            "static_threshold": "Rest, with signed force within 0.2 N of either static threshold; uniform duration.",
            "stopping_boundary": "Sample velocity and duration uniformly; choose nominal stopping time within 5% of duration and derive force; reject out-of-range forces.",
        },
        "splits": split_metadata,
        "split_policy": "Independent examples with identical category proportions per split; exact duplicate input tuples are excluded across all splits. No trajectories are sampled.",
        "normalization": "None. Values retain physical units. Fit any future normalization using only train.csv.",
        "evaluation_scope": "In-range synthetic examples from the reference physics; not an extrapolation or real-world validation dataset. Keep test.csv unused until final evaluation.",
        "physics_source": {
            "file": Path(physics.__file__).name,
            "function": "motion_cal",
            "sha256": hashlib.sha256(Path(physics.__file__).read_bytes()).hexdigest(),
        },
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {len(seen)} examples to {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Prepare synthetic cuboid labels; no training.")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "cuboid_translation_x" / "v1",
    )
    args = parser.parse_args()
    prepare_labels(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
