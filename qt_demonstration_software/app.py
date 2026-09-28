"""Launch the read-only Qt training result viewer."""
import argparse
import os
from pathlib import Path
import sys
import logging

os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"

from PySide6.QtWidgets import QApplication

from viewer.report_qt import MainWindow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        parser.error("--demo requires a bundled demo fixture, which is not available yet")
    log_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "BallTrainingViewer" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / "viewer.log", level=logging.INFO, encoding="utf-8")
    app = QApplication(sys.argv[:1])
    window = MainWindow(args.run, args.labels, args.demo)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
