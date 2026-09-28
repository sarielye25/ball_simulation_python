"""The only boundary through which training modules enter the viewer."""
import importlib
from pathlib import Path
import sys

from .contracts import ViewerError

TRAINING_ROOT = Path(__file__).resolve().parents[2] / "model_training" / "first_training_protocol"
MODULE_NAMES = ("config", "data", "metrics", "checkpoints", "neural_network", "training_records")


def training_modules():
    root = TRAINING_ROOT.resolve()
    for name in MODULE_NAMES:
        existing = sys.modules.get(name)
        if existing is not None and Path(existing.__file__).resolve().parent != root:
            raise ViewerError(f"同名训练模块冲突: {name}: {existing.__file__}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    modules = {name: importlib.import_module(name) for name in MODULE_NAMES}
    for name, module in modules.items():
        if Path(module.__file__).resolve().parent != root:
            raise ViewerError(f"训练模块路径错误: {name}: {module.__file__}")
    return modules
