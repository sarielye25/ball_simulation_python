"""Run E5 with E4's schedule and one more hidden layer."""

import argparse
import csv
import importlib
import json
from pathlib import Path
import sys
from statistics import mean

# This part sets up the paths and seeds, and updates.
ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts - e5"
SEEDS = (42, 1006, 10086, 104792, 230786)
UPDATES = (2000, 4000, 6000, 8000, 10000)
SCOPES = (
    ("Overall", "overall"),
    ("Moving at observation", "moving_at_observation"),
    ("Always resting", "always_resting"),
    ("Moved then stopped", "moved_then_stopped"),
    ("Breakaway", "breakaway"),
)
REPORT = ROOT.parent.parent / "training_results" / "26.10.1 controlled experiment.md"
REPORT_HEADING = "## E4 and E5 comparison:"
THREE_EXPERIMENT_HEADING = "## E3, E4 and E5 overall MSE comparison:"


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


def read_training_mse(run_directory):
    with (run_directory / "evaluations.csv").open(newline="", encoding="utf-8") as stream:
        values = {
            (int(row["update"]), row["scope"]): float(row["standardized_mse"])
            for row in csv.DictReader(stream)
            if row["split"] == "train"
            and row["predictor"] == "model"
            and int(row["update"]) in UPDATES
            and row["scope"] in {scope for _, scope in SCOPES}
        }
    expected = {(update, scope) for update in UPDATES for _, scope in SCOPES}
    if set(values) != expected:
        raise ValueError(f"Missing fixed-update training evaluations: {run_directory}")
    return values


def replace_report_section(report, heading, lines):
    heading_start = report.find(heading)
    section = "\n".join(lines) + "\n"
    if heading_start < 0:
        return report.rstrip() + "\n\n" + section
    section_end = report.find("\n## ", heading_start + len(heading))
    if section_end < 0:
        section_end = len(report)
    return report[:heading_start] + section + report[section_end:]


def write_markdown_comparison(e3_runs, e4_runs, e5_runs):
    evaluations = {
        experiment: [read_training_mse(runs / f"seed_{seed}") for seed in SEEDS]
        for experiment, runs in (("E3", e3_runs), ("E4", e4_runs), ("E5", e5_runs))
    }
    lines = [
        REPORT_HEADING,
        "",
        "Five-seed mean training standardized MSE at each fixed update:",
        "",
        "| Update | Overall | Moving at observation | Always resting | Moved then stopped | Breakaway |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for update in UPDATES:
        for experiment in ("E4", "E5"):
            values = [
                f"{mean(run[(update, scope)] for run in evaluations[experiment]):.9f}"
                for _, scope in SCOPES
            ]
            lines.append(f"| {update:,} ({experiment}) | " + " | ".join(values) + " |")
    overall_lines = [
        THREE_EXPERIMENT_HEADING,
        "",
        "Five-seed mean training overall standardized MSE at each fixed update:",
        "",
        "| Update | E3 MSE | E4 MSE | E5 MSE |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for update in UPDATES:
        values = [
            f"{mean(run[(update, 'overall')] for run in evaluations[experiment]):.9f}"
            for experiment in ("E3", "E4", "E5")
        ]
        overall_lines.append(f"| {update:,} | " + " | ".join(values) + " |")
    report = REPORT.read_text(encoding="utf-8")
    updated = replace_report_section(report, REPORT_HEADING, lines)
    updated = replace_report_section(updated, THREE_EXPERIMENT_HEADING, overall_lines)
    REPORT.write_text(updated, encoding="utf-8")
    print(f"Markdown comparison saved to {REPORT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    args = parser.parse_args()

    sys.path.insert(0, str(SCRIPTS))
    config = importlib.import_module("config")
    config.lr_reduction_patience_updates = 500
    config.lr_scheduler_patience = config.lr_reduction_patience_updates // config.eval_interval - 1
    config.training_data_dir = ROOT.parent / "data" / "e5" / "runs"
    training_loop = importlib.import_module("training_loop")

    group_directory = config.training_data_dir.parent
    group_directory.mkdir(parents=True, exist_ok=True)
    group_path = group_directory / "group.json"
    if not group_path.exists():
        group_path.write_text(json.dumps({
            "protocol_version": 1,
            "group_id": "e5",
            "name": "E5 additional hidden layer",
            "description": "E4 schedule with one more hidden layer in the neural network.",
            "source": "model_training/first_training_experiments_additional_hidden_layer2/run_e5.py",
        }, indent=2) + "\n", encoding="utf-8")

    e4_runs = ROOT.parent / "data" / "e4" / "runs"
    baseline_by_seed = {}
    for seed in SEEDS:
        run_directory = e4_runs / f"seed_{seed}"
        manifest = run_directory / "run.json"
        if manifest.is_file():
            run_config = json.loads(manifest.read_text(encoding="utf-8"))["run_config"]
            if run_config["seed"] != seed or run_config["max_updates"] != 10000:
                raise ValueError(f"Unexpected E4 baseline configuration: {manifest}")
            baseline_by_seed[seed] = run_directory

    selected_seeds = args.seed or SEEDS
    for seed in selected_seeds:
        if seed not in baseline_by_seed:
            raise FileNotFoundError(f"No E4 10,000-update run for seed {seed}")
        run_directory = config.training_data_dir / f"seed_{seed}"
        if not run_directory.exists():
            training_loop.run_training(run_directory=run_directory, seed=seed)
        else:
            termination = run_directory / "termination.json"
            if not termination.is_file() or json.loads(termination.read_text(encoding="utf-8"))["status"] != "completed":
                raise ValueError(f"Existing E5 run is incomplete: {run_directory}")

    comparison_path = config.training_data_dir.parent / "e5_vs_e4.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("seed", "update", "schedule", "validation_standardized_mse",
                         "validation_intermediate_pass_rate"))
        for seed in SEEDS:
            e5_run = config.training_data_dir / f"seed_{seed}"
            if not e5_run.exists():
                continue
            for schedule, run_directory in (("E4", baseline_by_seed[seed]),
                                            ("E5 additional hidden layer", e5_run)):
                evaluations = read_evaluations(run_directory)
                if set(evaluations) != set(UPDATES):
                    raise ValueError(f"Missing fixed-update evaluations: {run_directory}")
                for update in UPDATES:
                    row = evaluations[update]
                    writer.writerow((seed, update, schedule, row["standardized_mse"],
                                     row["pass_rate_intermediate"]))
    print(f"Comparison saved to {comparison_path}")
    if all((config.training_data_dir / f"seed_{seed}" / "termination.json").is_file()
           for seed in SEEDS):
        e3_runs = ROOT.parent / "data" / "e3" / "runs"
        write_markdown_comparison(e3_runs, e4_runs, config.training_data_dir)


if __name__ == "__main__":
    main()
