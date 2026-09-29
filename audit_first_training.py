import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model_training" / "first_training_protocol"))
sys.path.insert(0, str(ROOT / "qt_software_building"))
from neural_network import transmodel
from physics_formula import motion_cal
from viewer.reader import load_run
from viewer.curves import prepare_curve
from viewer.failures import analyze_failures


def audit():
    snapshot = load_run(ROOT / "model_training" / "data", "first_training_protocol", "20260929_142936_371616")
    run = ROOT / "model_training" / "data" / snapshot.group.id / "runs" / snapshot.run_id
    metrics = {(record.update, record.split_id, record.scope, record.name): record.value for record in snapshot.metrics if record.predictor == "model"}
    output = {"termination": snapshot.termination, "selected_checkpoint": snapshot.selected_checkpoint, "checkpoints": [], "max_prediction_difference": 0.0, "max_metric_difference": 0.0}
    for checkpoint in snapshot.checkpoints:
        saved = torch.load(run / "checkpoints" / f"{checkpoint.id}.pt", weights_only=True, map_location="cpu")
        model = transmodel()
        model.load_state_dict(saved["model_state"])
        model.eval()
        normalization = {name: np.asarray(value) for name, value in saved["normalization"].items()}
        output["normalization"] = saved["normalization"]
        output["run_config"] = saved["run_config"]
        checkpoint_result = {"update": checkpoint.update, "splits": {}}
        for split in snapshot.splits:
            samples = snapshot.samples[split.id]
            inputs = np.asarray([sample.inputs for sample in samples])
            targets = np.asarray([sample.targets for sample in samples])
            by_id = {row.row_id: row.values for row in snapshot.predictions[checkpoint.id, split.id]}
            predictions = np.asarray([by_id[sample.row_id] for sample in samples])
            standardized = (inputs - normalization["input_mean"]) / normalization["input_scale"]
            with torch.inference_mode():
                reloaded = np.concatenate([model(torch.tensor(batch, dtype=torch.float32)).numpy() for batch in np.array_split(standardized, range(1024, len(inputs), 1024))])
            reloaded = reloaded * normalization["target_scale"] + normalization["target_mean"]
            difference = float(np.max(np.abs(reloaded - predictions)))
            output["max_prediction_difference"] = max(output["max_prediction_difference"], difference)
            np.testing.assert_allclose(reloaded, predictions, rtol=0, atol=1e-10)
            absolute = np.abs(predictions - targets)
            mse = float(np.mean(((predictions - targets) / normalization["target_scale"]) ** 2))
            summary = {"standardized_mse": mse, "mae": absolute.mean(axis=0).tolist(), "rmse": np.sqrt(np.mean(absolute ** 2, axis=0)).tolist(), "p95": np.percentile(absolute, 95, axis=0).tolist(), "passes": {ruler: int(np.all(absolute < limits, axis=1).sum()) for ruler, limits in snapshot.tolerances.items()}}
            for ruler, passed in summary["passes"].items():
                assert passed == analyze_failures(snapshot, split.id, checkpoint.id, ruler).passed
                assert passed / len(samples) == metrics[checkpoint.update, split.id, "overall", f"pass_rate_{ruler}"]
            difference = abs(mse - metrics[checkpoint.update, split.id, "overall", "standardized_mse"])
            output["max_metric_difference"] = max(output["max_metric_difference"], difference)
            assert difference < 1e-7
            curve = dict(prepare_curve(snapshot, split.id, "standardized_mse").points)
            assert curve[checkpoint.update] == metrics[checkpoint.update, split.id, "overall", "standardized_mse"]
            summary["groups"] = {}
            for group in sorted({sample.motion_group for sample in samples}):
                mask = np.asarray([sample.motion_group == group for sample in samples])
                summary["groups"][group] = {"count": int(mask.sum()), "mae": absolute[mask].mean(axis=0).tolist(), "intermediate_passes": int(np.all(absolute[mask] < snapshot.tolerances["intermediate"], axis=1).sum())}
            checkpoint_result["splits"][split.id] = summary
            if checkpoint.update == 0:
                physics = np.asarray([motion_cal(force, 1.0, duration, observation, velocity)[::-1] for velocity, force, duration, observation in inputs])
                output[f"{split.id}_physics_label_max_difference"] = float(np.max(np.abs(physics - targets)))
        output["checkpoints"].append(checkpoint_result)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    audit()
