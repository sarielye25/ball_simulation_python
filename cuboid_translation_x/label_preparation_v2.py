import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random

import label_preparation as original
import physics_formula as physics


COLUMNS = [
    "v0_m_s", "force_N", "force_duration_s", "observation_time_s",
    "displacement_m", "v_final_m_s",
]
MAX_COAST_S = 2.0
CATEGORIES = {
    "uniform": 400,
    "from_rest": 100,
    "zero_force": 100,
    "static_threshold": 100,
    "stopping_boundary": 100,
    "push_endpoint": 100,
    "coast_stopping_boundary": 50,
    "rest_after_coast": 50,
}


def sample_inputs(rng, category):
    base_category = category if category in original.CATEGORIES else "uniform"
    while True:
        velocity, force, duration = original.sample_inputs(rng, base_category)
        coast = rng.uniform(0.0, MAX_COAST_S)
        if category == "push_endpoint":
            coast = 0.0
        elif category in ("coast_stopping_boundary", "rest_after_coast"):
            push_velocity, _ = physics.motion_cal(
                force, original.MASS_KG, duration, duration, velocity
            )
            stopping_time = abs(push_velocity) / (physics.mu_k * physics.g)
            if not 0.0 < stopping_time < MAX_COAST_S:
                continue
            if category == "coast_stopping_boundary":
                coast = stopping_time * rng.uniform(0.95, 1.05)
                if coast > MAX_COAST_S:
                    continue
            else:
                coast = rng.uniform(stopping_time, MAX_COAST_S)
        return velocity, force, duration, duration + coast


def generate_split(rng, multiplier, seen):
    rows = []
    counts = {}
    for category, count in CATEGORIES.items():
        counts[category] = count * multiplier
        for _ in range(counts[category]):
            inputs = sample_inputs(rng, category)
            condition = inputs[:3]
            while condition in seen:
                inputs = sample_inputs(rng, category)
                condition = inputs[:3]
            seen.add(condition)
            velocity, force, duration, observation = inputs
            final_velocity, displacement = physics.motion_cal(
                force, original.MASS_KG, duration, observation, velocity
            )
            row = (*inputs, displacement, final_velocity)
            if not all(math.isfinite(value) for value in row):
                raise ValueError(f"Non-finite example: {row}")
            rows.append(row)
    rng.shuffle(rows)
    return rows, counts


def source_record(path):
    path = Path(path)
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare_labels(output_dir, seed):
    output_dir = Path(output_dir)
    paths = [output_dir / name for name in (
        "train.csv", "validation.csv", "test.csv", "metadata.json"
    )]
    if any(path.exists() for path in paths):
        raise FileExistsError("Dataset files already exist; choose a new output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    seen = set()
    splits = {}
    for split, multiplier in (("train", 8), ("validation", 1), ("test", 1)):
        rows, counts = generate_split(rng, multiplier, seen)
        path = output_dir / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        splits[split] = {
            "file": path.name,
            "rows": len(rows),
            "sampling_category_counts": counts,
            "sha256": source_record(path)["sha256"],
        }
    metadata = {
        "dataset_version": "v2",
        "seed": seed,
        "total_rows": len(seen),
        "input_columns": COLUMNS[:4],
        "label_columns": COLUMNS[4:],
        "units": dict(zip(COLUMNS, ("m/s", "N", "s", "s", "m", "m/s"))),
        "prediction_contract": {
            "description": "Constant force until t1, then zero applied force until t; friction acts throughout.",
            "force_removal_time": "t1 = force_duration_s",
            "prediction_time": "t = observation_time_s",
            "time_constraint": "0.1 <= t1 <= 1; t1 <= t <= t1 + 2",
            "initial_position": "Displacement is relative to the initial position.",
        },
        "fixed_parameters": {
            "mass_kg": original.MASS_KG, "gravity_m_s2": physics.g,
            "mu_s": physics.mu_s, "mu_k": physics.mu_k,
        },
        "input_ranges": {
            "v0_m_s": original.VELOCITY_RANGE,
            "force_N": original.FORCE_RANGE,
            "force_duration_s": original.DURATION_RANGE,
            "observation_time_s": [original.DURATION_RANGE[0], original.DURATION_RANGE[1] + MAX_COAST_S],
            "time_after_force_removal_s": [0.0, MAX_COAST_S],
        },
        "sampling": {
            "uniform": "Uniform velocity, force, push duration, and time after removal; observation time is their time sum.",
            "from_rest": "Initial velocity zero; otherwise uniform.",
            "zero_force": "Applied force zero; otherwise uniform.",
            "static_threshold": "Initial rest; force within 0.2 N of either static threshold.",
            "stopping_boundary": "Nominal stopping time during the push within 5% of push duration; inherited v1 sampling.",
            "push_endpoint": "Observation exactly at force removal.",
            "coast_stopping_boundary": "Nonzero velocity at removal; observation within 5% of analytical coast stopping time, restricted to the time range.",
            "rest_after_coast": "Nonzero velocity at removal; observation uniformly between coast stopping time and maximum coast duration.",
        },
        "splits": splits,
        "split_policy": "Independent conditions, with no repeated (v0, F, t1) across or within splits; no shared trajectories at different observation times.",
        "normalization": "None; fit normalization on training data only.",
        "evaluation_scope": "Synthetic reference-physics agreement, not real-world validation. Reserve test data for final evaluation.",
        "sources": [source_record(physics.__file__), source_record(original.__file__), source_record(__file__)],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {len(seen)} examples to {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Generate four-input push-and-coast labels.")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "labels" / "v2",
    )
    args = parser.parse_args()
    prepare_labels(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
