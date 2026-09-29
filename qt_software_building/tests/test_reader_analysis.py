import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from viewer.curves import prepare_curve
from viewer.failures import analyze_failures
from viewer.reader import DataError, METRIC_HEADER, list_groups, list_runs, load_run


def _csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _descriptor(path, run_dir, append_only=False):
    content = path.read_bytes()
    result = {"file": path.relative_to(run_dir).as_posix(), "sha256": hashlib.sha256(content).hexdigest()}
    result["committed_bytes" if append_only else "bytes"] = len(content)
    return result


def _metric_row(update, split, mse):
    return [update, split, "model", "overall", 2, mse, mse, mse, mse, mse, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5]


class ReaderAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data_root = Path(self.temporary.name)
        self.group_dir = self.data_root / "demo"
        self.run_dir = self.group_dir / "runs" / "first"
        self.run_dir.mkdir(parents=True)
        (self.group_dir / "group.json").write_text(json.dumps({"protocol_version": 1, "group_id": "demo", "name": "Demo", "description": "", "source": "test"}), encoding="utf-8")
        self.manifest = {
            "protocol_version": 1, "group_id": "demo", "run_id": "first", "generation": 1,
            "status": "running", "purpose": "diagnostic", "splits": [],
            "columns": {"inputs": ["input"], "targets": ["target"], "units": {"input": "m", "target": "m"}},
            "tolerances": {"coarse": [0.5]}, "stopping_ruler": "coarse",
            "artifacts": {}, "checkpoints": [], "selected_checkpoint": None, "termination": None,
        }
        for split_id, role in (("train", "train"), ("validation", "validation"), ("validation_extra", "validation")):
            path = self.run_dir / "datasets" / f"{split_id}.csv"
            _csv(path, ["row_id", "motion_group", "input", "target"], [["a", "moving", 1, 2], ["b", "resting", 2, 4]])
            self.manifest["splits"].append({"id": split_id, "name": split_id, "role": role, "rows": 2, "dataset": _descriptor(path, self.run_dir)})
        metrics = self.run_dir / "evaluations.csv"
        _csv(metrics, METRIC_HEADER, [_metric_row(0, "train", 2), _metric_row(0, "validation", 3), _metric_row(0, "validation_extra", 4)])
        self.manifest["artifacts"]["evaluations"] = _descriptor(metrics, self.run_dir, True)
        self.metrics_path = metrics
        predictions = {}
        for split_id in ("train", "validation", "validation_extra"):
            path = self.run_dir / "predictions" / "update_000000" / f"{split_id}.csv"
            _csv(path, ["row_id", "pred_target"], [["b", 5], ["a", 2.2]])
            predictions[split_id] = _descriptor(path, self.run_dir)
        self.manifest["checkpoints"] = [{"id": "update_000000", "update": 0, "predictions": predictions, "model_file": None}]
        self.publish()

    def publish(self):
        (self.run_dir / "run.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_reader_and_analysis(self):
        self.assertEqual([group.id for group in list_groups(self.data_root)], ["demo"])
        self.assertEqual(list_runs(self.data_root, "demo"), ("first",))
        snapshot = load_run(self.data_root, "demo", "first")
        curve = prepare_curve(snapshot, "validation_extra", "standardized_mse")
        self.assertEqual(curve.points, ((0, 4.0),))
        report = analyze_failures(snapshot, "validation", "update_000000", "coarse")
        self.assertEqual((report.total, report.passed, report.failed), (2, 1, 1))
        self.assertEqual([row.row_id for row in report.rows], ["a", "b"])
        self.assertEqual(report.rows[1].signed_errors, (1.0,))

    def test_optional_experiment_metadata(self):
        self.manifest["run_config"] = {"learning_rate": 0.001, "seed": 42}
        self.manifest["model_config"] = {"widths": [1, 1]}
        self.manifest["normalization"] = {
            "input_mean": [-1], "input_scale": [2],
            "target_mean": [-3], "target_scale": [4],
        }
        self.publish()
        snapshot = load_run(self.data_root, "demo", "first")
        self.assertEqual(snapshot.run_config["seed"], 42)
        self.assertEqual(snapshot.normalization["target_mean"], (-3.0,))

    def test_tolerance_boundary_is_failure(self):
        descriptor = self.manifest["checkpoints"][0]["predictions"]["validation"]
        path = self.run_dir / descriptor["file"]
        _csv(path, ["row_id", "pred_target"], [["b", 4.5], ["a", 2.2]])
        self.manifest["checkpoints"][0]["predictions"]["validation"] = _descriptor(path, self.run_dir)
        self.publish()
        report = analyze_failures(load_run(self.data_root, "demo", "first"),
                                  "validation", "update_000000", "coarse")
        self.assertEqual(report.failed, 1)

    def test_uncommitted_tail_is_ignored(self):
        with self.metrics_path.open("ab") as stream:
            stream.write(b"unfinished,not,a,complete,row")
        snapshot = load_run(self.data_root, "demo", "first")
        self.assertEqual(prepare_curve(snapshot, "train", "standardized_mse").points, ((0, 2.0),))

    def test_new_generation_reads_new_committed_prefix(self):
        with self.metrics_path.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(_metric_row(1, "validation", 1.5))
        self.manifest["generation"] = 2
        self.manifest["artifacts"]["evaluations"] = _descriptor(self.metrics_path, self.run_dir, True)
        self.publish()
        snapshot = load_run(self.data_root, "demo", "first")
        self.assertEqual(prepare_curve(snapshot, "validation", "standardized_mse").points, ((0, 3.0), (1, 1.5)))

    def test_initial_generation_has_no_metrics_or_checkpoint(self):
        self.manifest["generation"] = 0
        self.manifest["artifacts"]["evaluations"] = None
        self.manifest["checkpoints"] = []
        self.publish()
        snapshot = load_run(self.data_root, "demo", "first")
        self.assertEqual(snapshot.metrics, ())
        self.assertEqual(snapshot.checkpoints, ())

    def test_corrupt_committed_prefix_is_rejected(self):
        content = self.metrics_path.read_bytes()
        self.metrics_path.write_bytes(content.replace(b",train,", b",other,", 1))
        with self.assertRaisesRegex(DataError, "SHA-256 mismatch"):
            load_run(self.data_root, "demo", "first")

    def test_missing_prediction_id_is_rejected(self):
        descriptor = self.manifest["checkpoints"][0]["predictions"]["validation"]
        path = self.run_dir / descriptor["file"]
        _csv(path, ["row_id", "pred_target"], [["a", 2.2]])
        self.manifest["checkpoints"][0]["predictions"]["validation"] = _descriptor(path, self.run_dir)
        self.publish()
        with self.assertRaisesRegex(DataError, "prediction row IDs"):
            load_run(self.data_root, "demo", "first")

    def test_unsupported_version_is_rejected(self):
        self.manifest["protocol_version"] = 99
        self.publish()
        with self.assertRaisesRegex(DataError, "unsupported run version"):
            load_run(self.data_root, "demo", "first")

    def test_escaping_artifact_path_is_rejected(self):
        self.manifest["splits"][0]["dataset"]["file"] = "../datasets/train.csv"
        self.publish()
        with self.assertRaisesRegex(DataError, "invalid artifact path"):
            load_run(self.data_root, "demo", "first")

    def test_completed_run_requires_selected_checkpoint(self):
        self.manifest["status"] = "completed"
        self.manifest["termination"] = {"reason": "limit", "final_update": 0, "selected_checkpoint": None}
        self.publish()
        with self.assertRaisesRegex(DataError, "completion selection"):
            load_run(self.data_root, "demo", "first")

    def test_moved_data_root(self):
        moved = self.data_root / "moved"
        moved.mkdir()
        self.group_dir.rename(moved / "demo")
        self.assertEqual(load_run(moved, "demo", "first").run_id, "first")


if __name__ == "__main__":
    unittest.main()
