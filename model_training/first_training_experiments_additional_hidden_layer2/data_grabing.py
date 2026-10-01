"""Add E5 seed and learning-rate tables to the October 1 report."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / "data" / "e5" / "runs"
REPORT = ROOT.parent.parent / "training_results" / "26.10.1 controlled experiment.md"
HEADING = "## E5 seed and learning-rate details"
UPDATES = (2000, 4000, 6000, 8000, 10000)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def collect_run(run_directory):
    evaluations = {}
    for row in read_csv(run_directory / "evaluations.csv"):
        if row["predictor"] != "model" or row["scope"] != "overall":
            continue
        key = (row["split"], int(row["update"]))
        if key in evaluations:
            raise ValueError(f"Duplicate overall evaluation {key}: {run_directory}")
        evaluations[key] = float(row["standardized_mse"])

    events = sorted(read_csv(run_directory / "evaluation_events.csv"),
                    key=lambda row: int(row["update"]))
    changes = []
    for position, event in enumerate(events):
        old_rate = float(event["lr_before"])
        new_rate = float(event["lr_next"])
        if new_rate == old_rate:
            continue
        update = int(event["update"])
        if position + 1 >= len(events):
            raise ValueError(f"No evaluation after rate change at {update}: {run_directory}")
        next_update = int(events[position + 1]["update"])
        try:
            current_mse = evaluations[("validation", update)]
            next_mse = evaluations[("validation", next_update)]
        except KeyError as error:
            raise ValueError(f"Missing validation evaluation {error}: {run_directory}") from error
        changes.append((update, next_update, old_rate, new_rate,
                        current_mse, next_mse))
    return evaluations, changes


def make_tables():
    fixed_rows = [
        "| Seed | 2,000 | 4,000 | 6,000 | 8,000 | 10,000 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rate_rows = [
        "| Seed | LR change update | LR before → after | Next validation update | Validation MSE at change | Next validation MSE | Improvement | Improvement % |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    run_directories = sorted(RUNS.glob("seed_*"), key=lambda path: int(path.name.removeprefix("seed_")))
    if not run_directories:
        raise FileNotFoundError(f"No E5 seed runs found in {RUNS}")
    for run_directory in run_directories:
        seed = int(run_directory.name.removeprefix("seed_"))
        evaluations, changes = collect_run(run_directory)
        try:
            values = [evaluations[("train", update)] for update in UPDATES]
        except KeyError as error:
            raise ValueError(f"Missing training evaluation {error}: {run_directory}") from error
        fixed_rows.append(f"| {seed} | " + " | ".join(f"{value:.9f}" for value in values) + " |")
        for update, next_update, old_rate, new_rate, current_mse, next_mse in changes:
            improvement = current_mse - next_mse
            percentage = improvement / current_mse * 100 if current_mse else float("nan")
            rate_rows.append(
                f"| {seed} | {update:,} | {old_rate:.8g} → {new_rate:.8g} | "
                f"{next_update:,} | {current_mse:.9f} | {next_mse:.9f} | "
                f"{improvement:+.9f} | {percentage:+.2f}% |"
            )
    return [
        HEADING,
        "",
        "### Training overall standardized MSE by seed",
        "",
        *fixed_rows,
        "",
        "### Learning-rate changes and the next validation check",
        "",
        *rate_rows,
        "",
        "Improvement is validation standardized MSE at the learning-rate change minus MSE at the next validation check; a positive value means lower error.",
    ]


def main():
    section = "\n".join(make_tables()) + "\n"
    report = REPORT.read_text(encoding="utf-8")
    start = report.find(HEADING)
    if start < 0:
        updated = report.rstrip() + "\n\n" + section
    else:
        end = report.find("\n## ", start + len(HEADING))
        if end < 0:
            end = len(report)
        updated = report[:start] + section.rstrip() + report[end:]
    REPORT.write_text(updated, encoding="utf-8")
    print(f"Updated {REPORT}")


if __name__ == "__main__":
    main()
