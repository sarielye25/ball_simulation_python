"""Write an E4 validation pass-rate table for every seed and saved update."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data" / "e4" / "runs"
OUTPUT = Path(__file__).with_name("e4_validation_table.md")
TOLERANCE_D = 0.01
TOLERANCE_V = 0.01
GROUPS = ("always_resting", "moved_then_stopped", "moving_at_observation", "breakaway")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def rates(rows):
    if not rows:
        return "-"
    distance = [abs(float(label["displacement_m"]) - float(prediction["pred_displacement_m"])) < TOLERANCE_D
                for label, prediction in rows]
    velocity = [abs(float(label["v_final_m_s"]) - float(prediction["pred_v_final_m_s"])) < TOLERANCE_V
                for label, prediction in rows]
    percentages = (sum(distance), sum(velocity), sum(d and v for d, v in zip(distance, velocity)))
    return " / ".join(f"{100 * count / len(rows):.1f}%" for count in percentages)


def main():
    lines = ["# E4 validation results", "",
             "Tolerance: displacement < 0.01 m; velocity < 0.01 m/s. Rates: displacement / velocity / both.", ""]
    for run in sorted(ROOT.glob("seed_*"), key=lambda path: int(path.name.split("_")[1])):
        labels = {row["row_id"]: row for row in read_csv(run / "datasets" / "validation.csv")}
        counts = [sum(row["motion_group"] == group for row in labels.values()) for group in GROUPS[:3]]
        counts.append(sum(float(row["v0_m_s"]) == 0
                          and abs(abs(float(row["force_N"])) - 3.924) <= 0.2
                          for row in labels.values()))
        lines.extend([f"## {run.name}", "",
                      "| Update | Standardized MSE | Overall | " + " | ".join(
                          f"{name} (n={count})" for name, count in zip(
                              ("Always resting", "Moved then stopped", "Moving at observation", "Breakaway"), counts)) + " |",
                      "| ---: | ---: | --- | --- | --- | --- | --- |"])
        mse = {int(row["update"]): float(row["standardized_mse"])
               for row in read_csv(run / "evaluations.csv")
               if row["split"] == "validation" and row["predictor"] == "model" and row["scope"] == "overall"}
        for directory in sorted((run / "predictions").glob("update_*")):
            update = int(directory.name.split("_")[1])
            pairs = [(labels[row["row_id"]], row) for row in read_csv(directory / "validation.csv")]
            if len(pairs) != len(labels) or len({label["row_id"] for label, _ in pairs}) != len(labels):
                raise ValueError(f"Prediction rows do not match labels: {directory}")
            scopes = [pairs]
            scopes.extend([(label, prediction) for label, prediction in pairs
                           if label["motion_group"] == group] for group in GROUPS[:3])
            scopes.append([(label, prediction) for label, prediction in pairs
                           if float(label["v0_m_s"]) == 0
                           and abs(abs(float(label["force_N"])) - 3.924) <= 0.2])
            lines.append("| " + " | ".join((str(update), f"{mse[update]:.6g}",
                                              *(rates(scope) for scope in scopes))) + " |")
        lines.append("")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
