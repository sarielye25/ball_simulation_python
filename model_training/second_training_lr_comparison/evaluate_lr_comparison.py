"""Build E4 learning-rate validation tables from saved predictions."""

import csv
import json
from pathlib import Path
from statistics import mean

from run_training import DATA_ROOT, GROUPS, ROOT, SEEDS


OUTPUT = ROOT / "lr_comparison_validation.md"
TOLERANCE = 0.01
SCOPES = (
    ("Overall", None),
    ("Always resting", "always_resting"),
    ("Moved then stopped", "moved_then_stopped"),
    ("Moving at observation", "moving_at_observation"),
    ("Breakaway", "breakaway"),
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def is_breakaway(row):
    return float(row["v0_m_s"]) == 0 and abs(abs(float(row["force_N"])) - 3.924) <= 0.2


def scope_rows(pairs, scope):
    if scope is None:
        return pairs
    if scope == "breakaway":
        return [(label, prediction) for label, prediction in pairs if is_breakaway(label)]
    return [(label, prediction) for label, prediction in pairs if label["motion_group"] == scope]


def rates(pairs):
    if not pairs:
        return "-"
    displacement = [abs(float(label["displacement_m"]) - float(prediction["pred_displacement_m"])) < TOLERANCE
                    for label, prediction in pairs]
    velocity = [abs(float(label["v_final_m_s"]) - float(prediction["pred_v_final_m_s"])) < TOLERANCE
                for label, prediction in pairs]
    counts = (sum(displacement), sum(velocity),
              sum(distance and speed for distance, speed in zip(displacement, velocity)))
    return " / ".join(f"{100 * count / len(pairs):.1f}%" for count in counts)


def report_run(run, lines):
    labels = {row["row_id"]: row for row in read_csv(run / "datasets" / "validation.csv")}
    if len(labels) != 1000:
        raise ValueError(f"Unexpected validation labels: {run}")
    mse = {int(row["update"]): float(row["standardized_mse"])
           for row in read_csv(run / "evaluations.csv")
           if row["split"] == "validation" and row["predictor"] == "model"
           and row["scope"] == "overall"}
    headings = [f"{name} (n={len(scope_rows([(label, None) for label in labels.values()], scope))})"
                for name, scope in SCOPES[1:]]
    lines.extend([f"### {run.name}", "",
                  "| Update | Standardized MSE | Overall | " + " | ".join(headings) + " |",
                  "| ---: | ---: | --- | --- | --- | --- | --- |"])
    for directory in sorted((run / "predictions").glob("update_*")):
        update = int(directory.name.split("_")[1])
        predictions = read_csv(directory / "validation.csv")
        ids = [row["row_id"] for row in predictions]
        if len(ids) != len(labels) or set(ids) != set(labels):
            raise ValueError(f"Prediction rows do not match labels: {directory}")
        pairs = [(labels[row["row_id"]], row) for row in predictions]
        if update not in mse:
            raise ValueError(f"Missing validation MSE: {directory}")
        values = [rates(scope_rows(pairs, scope)) for _, scope in SCOPES]
        lines.append(f"| {update} | {mse[update]:.6g} | " + " | ".join(values) + " |")
    lines.append("")


def main():
    lines = ["# E4 learning-rate comparison: validation results", "",
             "Tolerance: displacement < 0.01 m; velocity < 0.01 m/s. "
             "Rates: displacement / velocity / both.", "",
             "Final MSE is measured at each run's selected checkpoint. "
             "Group means use only completed runs.", "",
             "| Group | Completed seeds | Mean selected validation standardized MSE |",
             "| --- | ---: | ---: |"]
    runs_by_group = {}
    report_groups = {"e4": "E4-1 original", **{group_id: group[0] for group_id, group in GROUPS.items()}}
    for group_id, name in report_groups.items():
        runs = []
        for seed in SEEDS:
            run = DATA_ROOT / group_id / "runs" / f"seed_{seed}"
            termination = run / "termination.json"
            if termination.is_file() and json.loads(termination.read_text(encoding="utf-8"))["status"] == "completed":
                runs.append(run)
        runs_by_group[group_id] = runs
        selected_mse = []
        for run in runs:
            index = json.loads((run / "checkpoints" / "index.json").read_text(encoding="utf-8"))
            selected = next(item for item in index["snapshots"] if item["file"] == index["selected"])
            selected_mse.append(selected["validation_mse"])
        value = f"{mean(selected_mse):.9f}" if selected_mse else "-"
        lines.append(f"| {name} | {len(runs)}/5 | {value} |")
    lines.append("")
    for group_id, name in report_groups.items():
        lines.extend([f"## {name}", ""])
        if not runs_by_group[group_id]:
            lines.extend(["No completed runs yet.", ""])
        for run in runs_by_group[group_id]:
            report_run(run, lines)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
