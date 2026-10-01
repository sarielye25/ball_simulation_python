"""Run E4 with three hidden layers and the selected first-stage schedule."""

import argparse
import importlib
import json
from pathlib import Path
import sys

# This part sets up the paths and seeds, and updates.
ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts - e4"
SEEDS = (42, 1006, 10086, 104792, 230786)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    args = parser.parse_args()

    sys.path.insert(0, str(SCRIPTS))
    config = importlib.import_module("config")
    config.training_data_dir = ROOT.parent.parent / "data" / "e4" / "runs"
    training_loop = importlib.import_module("training_loop")

    group_directory = config.training_data_dir.parent
    group_directory.mkdir(parents=True, exist_ok=True)
    group_path = group_directory / "group.json"
    if not group_path.exists():
        group_path.write_text(json.dumps({
            "protocol_version": 1,
            "group_id": "e4",
            "name": "E4 additional hidden layer",
            "description": "E3 schedule with an additional hidden layer in the neural network.",
            "source": "model_training/first_training_protocol/training_E4/run_e4.py",
        }, indent=2) + "\n", encoding="utf-8")

    selected_seeds = args.seed or SEEDS
    for seed in selected_seeds:
        run_directory = config.training_data_dir / f"seed_{seed}"
        if not run_directory.exists():
            training_loop.run_training(run_directory=run_directory, seed=seed)
        else:
            termination = run_directory / "termination.json"
            if not termination.is_file() or json.loads(termination.read_text(encoding="utf-8"))["status"] != "completed":
                raise ValueError(f"Existing E4 run is incomplete: {run_directory}")

if __name__ == "__main__":
    main()
