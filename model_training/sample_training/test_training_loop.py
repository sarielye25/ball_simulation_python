"""Exercise orchestration with disposable runs, never the reserved test split."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import config
from checkpoints import load_checkpoint
from neural_network import transmodel
import training_loop


class TrainingLoopChecks(unittest.TestCase):
    def test_schedule_partial_batch_and_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "smoke"
            training_loop.run_training(max_updates=64, run_directory=run)
            metadata = json.loads((run / "run.json").read_text())
            self.assertEqual(metadata["protocol_version"], 1)
            self.assertEqual(metadata["status"], "completed")
            for folder in ("datasets", "checkpoints", "figures", "exports"):
                self.assertTrue((run / folder).is_dir())
            self.assertFalse((run / "datasets" / "test.csv").exists())
            for split, expected_count in (("train", 160), ("validation", 40)):
                identity = next(item for item in metadata["splits"] if item["id"] == split)
                snapshot = run / identity["dataset"]["file"]
                self.assertEqual(hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                                 identity["dataset"]["sha256"])
                with snapshot.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), expected_count)
                self.assertEqual([row["row_id"] for row in rows],
                                 training_loop.prepare_data()[0][split]["row_ids"])
                snapshot_inputs, snapshot_targets = training_loop.data.load_split(snapshot)
                source_inputs, source_targets = training_loop.data.load_split(
                    Path(__file__).parent / "labels" / f"{split}.csv")
                self.assertEqual(snapshot_inputs, source_inputs)
                self.assertEqual(snapshot_targets, source_targets)
            with (run / "batches.csv").open(newline="") as stream:
                batches = list(csv.DictReader(stream))
            self.assertEqual(len(batches), 64)
            self.assertEqual(int(batches[62]["batch_rows"]), 128)
            self.assertEqual(int(batches[63]["exposures"]), 5120)
            with (run / "evaluation_events.csv").open(newline="") as stream:
                events = list(csv.DictReader(stream))
            self.assertEqual([int(row["update"]) for row in events], [0, 10, 20, 50, 64])
            self.assertEqual([row["scheduler_checked"] for row in events],
                             ["True", "False", "False", "False", "False"])
            termination = json.loads((run / "termination.json").read_text())
            self.assertEqual(termination["reason"], "MAX_UPDATES")
            self.assertTrue(termination["max_updates_reached"])
            index = json.loads((run / "checkpoints" / "index.json").read_text())
            self.assertEqual(index["selected"], index["best_mse"])

            model = transmodel()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            payload = load_checkpoint(run / "checkpoints" / "update_000010.pt", model,
                                      optimizer=optimizer, scheduler=scheduler)
            self.assertIn("per_row", payload["validation_result"])
            splits, normalization = training_loop.prepare_data()
            self.assertEqual(normalization["input_mean"], payload["normalization"]["input_mean"])
            inputs = torch.tensor(splits["train"]["inputs"], dtype=torch.float32)
            targets = torch.tensor(splits["train"]["targets"], dtype=torch.float32)
            continuation = payload["continuation"]
            permutation = continuation["permutation"]
            position = continuation["next_position"]
            sampler = torch.Generator(device="cpu")
            sampler.set_state(continuation["sampler_rng"])
            for update in range(10):
                if position == len(permutation):
                    permutation = torch.randperm(len(inputs), generator=sampler)
                    position = 0
                indices = permutation[position:position + config.batch_size]
                training_loop.train_one_batch(model, optimizer, torch.nn.MSELoss(),
                                              inputs[indices], targets[indices])
                position += len(indices)
            expected = torch.load(run / "checkpoints" / "update_000020.pt", weights_only=True)
            for name, parameter in model.state_dict().items():
                torch.testing.assert_close(parameter, expected["model_state"][name], rtol=0, atol=0)

    def test_target_at_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "target"
            with patch.object(config, "target_pass_rate", 0):
                training_loop.run_training(max_updates=1, run_directory=run)
            self.assertFalse((run / "batches.csv").exists())
            termination = json.loads((run / "termination.json").read_text())
            self.assertEqual(termination["reason"], "TARGET_REACHED")
            self.assertEqual(termination["selected_checkpoint"], "update_000000.pt")

    def test_numerical_abort_keeps_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "abort"
            with patch.object(training_loop, "train_one_batch",
                              side_effect=FloatingPointError("Non-finite loss")):
                with self.assertRaises(FloatingPointError):
                    training_loop.run_training(max_updates=1, run_directory=run)
            termination = json.loads((run / "termination.json").read_text())
            self.assertEqual(termination["reason"], "NUMERICAL_FAILURE")
            self.assertEqual(termination["last_valid_checkpoint"], "update_000000.pt")
            self.assertIsNone(termination["selected_checkpoint"])

    def test_scheduler_cadence_and_patience(self):
        original = training_loop.evaluate_splits

        def stalled_results(*arguments):
            results = original(*arguments)
            for result in results.values():
                result["overall"]["standardized_mse"] = 1.0
                result["overall"]["pass_rates"][config.stopping_ruler] = 0.0
            return results

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "stall"
            with patch.object(training_loop, "evaluate_splits", side_effect=stalled_results):
                training_loop.run_training(max_updates=600, run_directory=run)
            with (run / "evaluation_events.csv").open(newline="") as stream:
                events = {int(row["update"]): row for row in csv.DictReader(stream)}
            self.assertEqual(float(events[100]["lr_next"]), config.learning_rate)
            self.assertEqual(float(events[200]["lr_next"]),
                             config.learning_rate * config.lr_reduction_factor)
            self.assertEqual(events[500]["scheduler_checked"], "False")
            termination = json.loads((run / "termination.json").read_text())
            self.assertEqual(termination["reason"], "NO_PROGRESS")
            self.assertEqual(termination["update"], 500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
