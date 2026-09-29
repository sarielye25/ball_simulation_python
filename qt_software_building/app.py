"""Launch the Qt training result viewer."""

import argparse
import importlib.util
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent / "model_training" / "data")
    args = parser.parse_args()
    if importlib.util.find_spec("PySide6") is None:
        local_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
        if local_python.is_file() and Path(sys.executable).resolve() != local_python.resolve():
            os.execv(str(local_python), [str(local_python), str(Path(__file__).resolve()), *sys.argv[1:]])
        parser.error("PySide6 is missing; install requirements-gui.txt in this folder's .venv")
    os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
    from PySide6.QtWidgets import QApplication
    from viewer.window import MainWindow

    application = QApplication(sys.argv[:1])
    window = MainWindow(args.data_root)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
