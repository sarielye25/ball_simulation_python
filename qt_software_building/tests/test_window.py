import os
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from viewer.window import MainWindow


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_latest_sample_run_displays_validation_failures(self):
        data_root = Path(__file__).resolve().parents[2] / "model_training" / "data"
        if not (data_root / "sample_training" / "runs" / "imported_20260928_214346_696295" / "run.json").is_file():
            self.skipTest("Imported diagnostic run not available")
        window = MainWindow(data_root)
        try:
            deadline = time.monotonic() + 10
            while window.snapshot is None and time.monotonic() < deadline:
                self.application.processEvents()
                time.sleep(0.01)
            self.assertIsNotNone(window.snapshot)
            self.assertEqual(window.snapshot.status, "completed")
            window.split_combo.setCurrentIndex(window.split_combo.findData("validation"))
            self.assertEqual(window.table.rowCount(), 40)
            self.assertIn("40 failed / 40 samples", window.summary_label.text())
        finally:
            window.close()

    def test_selectors_recenter_plot(self):
        data_root = Path(__file__).resolve().parents[2] / "model_training" / "data"
        if not (data_root / "sample_training" / "runs" / "imported_20260928_214346_696295" / "run.json").is_file():
            self.skipTest("Imported diagnostic run not available")
        window = MainWindow(data_root)
        try:
            deadline = time.monotonic() + 10
            while window.snapshot is None and time.monotonic() < deadline:
                self.application.processEvents()
                time.sleep(0.01)
            self.assertIsNotNone(window.snapshot)
            changes = (
                (window.split_combo, "validation"),
                (window.metric_combo, "displacement_mae_m"),
                (window.checkpoint_combo, "update_000000"),
                (window.ruler_combo, "coarse"),
            )
            for selector, value in changes:
                window.plot.setRange(xRange=(100, 110), yRange=(100, 110), padding=0)
                selector.setCurrentIndex(selector.findData(value))
                x_range, y_range = window.plot.getViewBox().viewRange()
                self.assertLess(x_range[0], 1)
                self.assertGreater(x_range[1], 0)
                self.assertLess(y_range[0], 20)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
