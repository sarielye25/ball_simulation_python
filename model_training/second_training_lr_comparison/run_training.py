"""Run E4 learning-rate comparison groups with five paired seeds."""

import argparse
import importlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data"
SEEDS = (42, 1006, 10086, 104792, 230786)
GROUPS = {
    "e4_2": ("E4-2 current scheduler", "plateau", "abs", 0.001),
    "e4_3": ("E4-3 constant LR", "constant", "abs", 0.001),
    "e4_4": ("E4-4 relative threshold", "plateau", "rel", 0.001),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS, action="append",
                        help="Run only selected groups; default is all three.")
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append",
                        help="Run only selected seeds; default is all five.")
    parser.add_argument("--max-updates", type=int,
                        help="Diagnostic cap; writes to separate diagnostic directories.")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "training_scripts"))
    config = importlib.import_module("config")
    training_loop = importlib.import_module("training_loop")
    config.lr_reduction_patience_updates = 500
    config.lr_scheduler_patience = config.lr_reduction_patience_updates // config.eval_interval - 1

    for group_id in args.group or GROUPS:
        name, schedule, threshold_mode, threshold = GROUPS[group_id]
        config.lr_schedule = schedule
        config.lr_threshold_mode = threshold_mode
        config.lr_threshold = threshold
        config.training_data_dir = DATA_ROOT / group_id / "runs"
        group_directory = config.training_data_dir.parent
        group_directory.mkdir(parents=True, exist_ok=True)
        group_path = group_directory / "group.json"
        metadata = {
            "protocol_version": 1,
            "group_id": group_id,
            "name": name,
            "description": f"E4 architecture; {schedule} LR, threshold mode {threshold_mode}, threshold {threshold}.",
            "source": "model_training/second_training_lr_comparison/run_training.py",
        }
        if group_path.exists():
            if json.loads(group_path.read_text(encoding="utf-8")) != metadata:
                raise ValueError(f"Group metadata differs: {group_path}")
        else:
            group_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        for seed in args.seed or SEEDS:
            suffix = f"diagnostic_{args.max_updates}_seed_{seed}" if args.max_updates else f"seed_{seed}"
            run_directory = config.training_data_dir / suffix
            if run_directory.exists():
                termination = run_directory / "termination.json"
                if not termination.is_file() or json.loads(termination.read_text(encoding="utf-8"))["status"] != "completed":
                    raise ValueError(f"Existing run is incomplete: {run_directory}")
                print(f"Already completed: {run_directory}")
                continue
            training_loop.run_training(run_directory=run_directory, seed=seed,
                                       max_updates=args.max_updates)


if __name__ == "__main__":
    main()
