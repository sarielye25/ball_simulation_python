"""Plot saved evaluation history without rerunning the model."""

import argparse
import csv
from pathlib import Path


SCOPES = {
    "overall": ("Overall", "#222222", "-"),
    "always_resting": ("Always resting", "#0072B2", "-"),
    "moved_then_stopped": ("Moved then stopped", "#009E73", "-"),
    "moving_at_observation": ("Moving at observation", "#CC79A7", "-"),
    "breakaway": ("Breakaway", "#D55E00", "-."),
}
PLOTS = {
    "standardized_mse": "Total standardized MSE",
    "displacement_standardized_mse": "Displacement standardized MSE",
    "velocity_standardized_mse": "Velocity standardized MSE",
    "displacement_loss_contribution": "Displacement contribution to total MSE",
    "velocity_loss_contribution": "Velocity contribution to total MSE",
    "displacement_p95_m": "Displacement P95 absolute error (m)",
    "velocity_p95_m_s": "Velocity P95 absolute error (m/s)",
    **{f"pass_rate_{name}": f"Joint pass rate: {name} (%)"
       for name in ("coarse", "intermediate", "fine")},
}


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot_curves(run_directory):
    """Use matched axes, observed evaluation points, and training-only baselines."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_directory = Path(run_directory)
    rows = _read_rows(run_directory / "evaluations.csv")
    baselines = _read_rows(run_directory / "baselines.csv")
    if not rows:
        raise ValueError("No evaluations to plot.")
    output_directory = run_directory / "figures"
    output_directory.mkdir(exist_ok=True)
    outputs = []
    for metric, title in PLOTS.items():
        figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True,
                                    layout="constrained")
        multiplier = 100 if metric.startswith("pass_rate_") else 1
        for axis, split in zip(axes, ("train", "validation")):
            for scope, (label, color, style) in SCOPES.items():
                selected = sorted((row for row in rows if row["split"] == split
                                   and row["scope"] == scope), key=lambda row: int(row["update"]))
                if not selected:
                    continue
                counts = sorted({int(row["count"]) for row in selected})
                count_label = str(counts[0]) if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"
                axis.plot([int(row["update"]) for row in selected],
                          [float(row[metric]) * multiplier if row[metric] else float("nan")
                           for row in selected], label=f"{label} (n={count_label})",
                          color=color, linestyle=style, marker="o", markersize=3,
                          linewidth=2.3 if scope == "overall" else 1.4)
            if split == "train":
                for name, color in (("zero", "#888888"), ("constant_velocity", "#B59B00")):
                    reference = [row for row in baselines if row["split"] == "train"
                                 and row["scope"] == "overall" and row["predictor"] == name]
                    if len(reference) != 1:
                        raise ValueError(f"Expected one training overall baseline for {name}.")
                    if reference[0][metric]:
                        axis.axhline(float(reference[0][metric]) * multiplier, color=color,
                                     linestyle="--", label=f"Baseline: {name.replace('_', ' ')}")
            axis.set(title=split.capitalize(), xlabel="Update", ylabel=title)
            axis.tick_params(axis="y", labelleft=True)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
            axis.set_ylim(bottom=0)
            if multiplier == 100:
                axis.set_ylim(0, 100)
        figure.suptitle(title)
        try:
            for extension in ("png", "svg"):
                destination = output_directory / f"{metric}.{extension}"
                figure.savefig(destination, dpi=160)
                outputs.append(destination)
        finally:
            plt.close(figure)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    arguments = parser.parse_args()
    for output in plot_curves(arguments.run_directory):
        print(output)
