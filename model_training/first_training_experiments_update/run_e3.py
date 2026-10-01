"""Run E3 with E1-A scheduler patience and compare it with E2."""

import argparse
import csv
import importlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts - 10000"
SEEDS = (42, 1006, 10086, 104792, 230786)
UPDATES = (2000, 4000, 6000, 8000, 10000)


def read_evaluations(run_directory):
    with (run_directory / "evaluations.csv").open(newline="", encoding="utf-8") as stream:
        return {
            int(row["update"]): row
            for row in csv.DictReader(stream)
            if row["split"] == "validation"
            and row["predictor"] == "model"
            and row["scope"] == "overall"
            and int(row["update"]) in UPDATES
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    args = parser.parse_args()

    sys.path.insert(0, str(SCRIPTS))
    config = importlib.import_module("config")
    config.lr_reduction_patience_updates = 500
    config.lr_scheduler_patience = config.lr_reduction_patience_updates // config.eval_interval - 1
    config.training_data_dir = ROOT.parent / "data" / "e3" / "runs"
    training_loop = importlib.import_module("training_loop")

    group_directory = config.training_data_dir.parent
    group_directory.mkdir(parents=True, exist_ok=True)
    group_path = group_directory / "group.json"
    if not group_path.exists():
        group_path.write_text(json.dumps({
            "protocol_version": 1,
            "group_id": "e3",
            "name": "E3 longer learning-rate patience",
            "description": "E1-A scheduler patience through a fixed 10000-update budget.",
            "source": "model_training/first_training_experiments_update/run_e3.py",
        }, indent=2) + "\n", encoding="utf-8")

    e2_runs = ROOT.parent / "data" / "update_10000" / "runs"
    baseline_by_seed = {}
    for run_directory in e2_runs.iterdir():
        manifest = run_directory / "run.json"
        if manifest.is_file():
            seed = json.loads(manifest.read_text(encoding="utf-8"))["run_config"]["seed"]
            baseline_by_seed[seed] = run_directory

    selected_seeds = args.seed or SEEDS
    for seed in selected_seeds:
        if seed not in baseline_by_seed:
            raise FileNotFoundError(f"No E2 10,000-update run for seed {seed}")
        run_directory = config.training_data_dir / f"seed_{seed}"
        if not run_directory.exists():
            training_loop.run_training(run_directory=run_directory, seed=seed)
        else:
            termination = run_directory / "termination.json"
            if not termination.is_file() or json.loads(termination.read_text(encoding="utf-8"))["status"] != "completed":
                raise ValueError(f"Existing E3 run is incomplete: {run_directory}")

    comparison_path = config.training_data_dir.parent / "e3_vs_e2.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("seed", "update", "schedule", "validation_standardized_mse",
                         "validation_intermediate_pass_rate"))
        for seed in SEEDS:
            e3_run = config.training_data_dir / f"seed_{seed}"
            if not e3_run.exists():
                continue
            for schedule, run_directory in (("E2 original", baseline_by_seed[seed]),
                                            ("E3 E1-A patience", e3_run)):
                evaluations = read_evaluations(run_directory)
                if set(evaluations) != set(UPDATES):
                    raise ValueError(f"Missing fixed-update evaluations: {run_directory}")
                for update in UPDATES:
                    row = evaluations[update]
                    writer.writerow((seed, update, schedule, row["standardized_mse"],
                                     row["pass_rate_intermediate"]))
    print(f"Comparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
