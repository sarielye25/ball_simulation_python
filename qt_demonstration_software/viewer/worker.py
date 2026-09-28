"""One active Qt worker and one latest pending request."""
from collections import OrderedDict
import threading

from PySide6.QtCore import QObject, QThread, Signal

from .inference import infer_checkpoint
from .report_data import load_run


class Task(QObject):
    result = Signal(int, int, object)
    error = Signal(int, int, str)
    finished = Signal()

    def __init__(self, generation, request_id, kind, arguments, cancel_event):
        super().__init__()
        self.generation, self.request_id, self.kind = generation, request_id, kind
        self.arguments, self.cancel_event = arguments, cancel_event

    def run(self):
        try:
            if not self.cancel_event.is_set():
                value = load_run(*self.arguments) if self.kind == "load" else infer_checkpoint(*self.arguments, self.cancel_event)
                if not self.cancel_event.is_set():
                    self.result.emit(self.generation, self.request_id, value)
        except InterruptedError:
            pass
        except Exception as error:
            if not self.cancel_event.is_set():
                self.error.emit(self.generation, self.request_id, str(error))
        finally:
            self.finished.emit()


class WorkerController(QObject):
    result = Signal(int, int, object)
    error = Signal(int, int, str)
    idle = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self.cancel_event = None
        self.pending = None
        self.cache = OrderedDict()

    def clear_cache(self):
        self.cache.clear()

    def submit(self, generation, request_id, kind, arguments):
        if self.cancel_event is not None:
            self.cancel_event.set()
            self.pending = (generation, request_id, kind, arguments)
            return
        self._start(generation, request_id, kind, arguments)

    def _start(self, generation, request_id, kind, arguments):
        self.cancel_event = threading.Event()
        self.thread = QThread()
        task = Task(generation, request_id, kind, arguments, self.cancel_event)
        task.moveToThread(self.thread)
        self.thread.started.connect(task.run)
        task.result.connect(self.result)
        task.error.connect(self.error)
        task.finished.connect(self.thread.quit)
        task.finished.connect(task.deleteLater)
        self.thread.finished.connect(self._finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _finished(self):
        self.thread = None
        self.cancel_event = None
        if self.pending is not None:
            pending, self.pending = self.pending, None
            self._start(*pending)
        else:
            self.idle.emit()

    def close(self):
        self.pending = None
        if self.cancel_event is not None:
            self.cancel_event.set()
