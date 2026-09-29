"""Run matched E1-A/E1-B seeds and compare validation MSE at update 2000."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 43, 44, 45, 46)
UPDATE = 2000


def validation_mse(run_directory):
    with (run_directory / "evaluations.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if int(row["update"]) == UPDATE
                and row["split"] == "validation"
                and row["scope"] == "overall"
                and row["predictor"] == "model"]
    if len(rows) != 1:
        raise ValueError(f"Expected one validation MSE at update {UPDATE}: {run_directory}")
    return float(rows[0]["standardized_mse"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "e1_comparison")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for condition in ("e1-a", "e1-b"):
        for seed in SEEDS:
            run_directory = output / condition / f"seed_{seed}"
            if run_directory.exists():
                raise FileExistsError(run_directory)
            subprocess.run([
                sys.executable, str(ROOT / f"scripts_{condition}" / "training_loop.py"),
                "--seed", str(seed), "--run-directory", str(run_directory),
            ], check=True)
            mse = validation_mse(run_directory)
            results.append((condition, seed, mse))
            print(f"{condition} seed={seed}: validation MSE at {UPDATE} = {mse:.8g}", flush=True)

    summary = output / "validation_mse_at_2000.csv"
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("condition", "seed", "update", "validation_standardized_mse"))
        writer.writerows((condition, seed, UPDATE, mse) for condition, seed, mse in results)
    for condition in ("e1-a", "e1-b"):
        values = [mse for name, _, mse in results if name == condition]
        print(f"{condition} mean validation MSE: {sum(values) / len(values):.8g}")
    print(f"Individual results: {summary}")


if __name__ == "__main__":
    main()
