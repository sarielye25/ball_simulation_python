"""Run E3 through 20,000 updates and summarize its 16k and 20k checkpoints."""

import argparse
import csv
import importlib
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts - 10000"
SEEDS = (42, 1006, 10086, 104792, 230786)
UPDATES = (16000, 20000)
MSE_FIELDS = (
    "standardized_mse",
    "displacement_standardized_mse",
    "velocity_standardized_mse",
    "displacement_loss_contribution",
    "velocity_loss_contribution",
)


def completed_run(run_directory):
    termination = run_directory / "termination.json"
    if not termination.is_file():
        return False
    result = json.loads(termination.read_text(encoding="utf-8"))
    return result["status"] == "completed" and result["update"] == 20000


def summarize(group_directory):
    measurements = {}
    for seed in SEEDS:
        run_directory = group_directory / "runs" / f"seed_{seed}_20000"
        if not completed_run(run_directory):
            raise ValueError(f"Missing completed 20,000-update run: {run_directory}")
        with (run_directory / "evaluations.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["predictor"] != "model" or int(row["update"]) not in UPDATES:
                    continue
                key = (int(row["update"]), row["split"], row["scope"])
                measurements.setdefault(key, {})[seed] = row

    summary_path = group_directory / "e3_extended_mse_means.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("update", "split", "scope", "count", "runs", *MSE_FIELDS))
        for (update, split, scope), by_seed in sorted(measurements.items()):
            if set(by_seed) != set(SEEDS):
                raise ValueError(f"Missing seeds for {update}, {split}, {scope}")
            counts = {int(row["count"]) for row in by_seed.values()}
            if len(counts) != 1:
                raise ValueError(f"Different row counts for {update}, {split}, {scope}")
            writer.writerow((update, split, scope, counts.pop(), len(SEEDS),
                             *(statistics.mean(float(by_seed[seed][field]) for seed in SEEDS)
                               for field in MSE_FIELDS)))
    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    args = parser.parse_args()

    sys.path.insert(0, str(SCRIPTS))
    config = importlib.import_module("config")
    config.max_updates = 20000
    config.lr_reduction_patience_updates = 500
    config.lr_scheduler_patience = config.lr_reduction_patience_updates // config.eval_interval - 1
    config.training_data_dir = ROOT.parent / "data" / "e3" / "runs"
    training_loop = importlib.import_module("training_loop")

    for seed in args.seed or SEEDS:
        run_directory = config.training_data_dir / f"seed_{seed}_20000"
        if run_directory.exists():
            if not completed_run(run_directory):
                raise ValueError(f"Existing E3 run is incomplete: {run_directory}")
            print(f"Already completed: {run_directory}", flush=True)
        else:
            training_loop.run_training(run_directory=run_directory, seed=seed)

    if all(completed_run(config.training_data_dir / f"seed_{seed}_20000") for seed in SEEDS):
        print(f"MSE summary saved to {summarize(config.training_data_dir.parent)}", flush=True)


if __name__ == "__main__":
    main()
