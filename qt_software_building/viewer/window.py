"""Qt window for published training results."""

from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .curves import available_metrics, prepare_curve
from .failures import analyze_failures
from .reader import list_groups, list_runs, load_run


class LoadSignals(QObject):
    loaded = Signal(int, object)
    failed = Signal(int, str)


class LoadTask(QRunnable):
    def __init__(self, data_root, group_id, run_id, request_id):
        super().__init__()
        self.data_root = data_root
        self.group_id = group_id
        self.run_id = run_id
        self.request_id = request_id
        self.signals = LoadSignals()

    def run(self):
        try:
            snapshot = load_run(self.data_root, self.group_id, self.run_id)
        except Exception as error:
            self.signals.failed.emit(self.request_id, str(error))
        else:
            self.signals.loaded.emit(self.request_id, snapshot)


class MainWindow(QMainWindow):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = Path(data_root).resolve()
        self.snapshot = None
        self.request_id = 0
        self.pool = QThreadPool.globalInstance()
        self.tasks = []
        self.setWindowTitle("Training Result Viewer")
        self.resize(1250, 820)

        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)
        selectors = QHBoxLayout()
        self.group_combo = QComboBox()
        self.run_combo = QComboBox()
        self.split_combo = QComboBox()
        self.metric_combo = QComboBox()
        self.checkpoint_combo = QComboBox()
        self.ruler_combo = QComboBox()
        refresh = QPushButton("Refresh")
        for label, widget in (("Group", self.group_combo), ("Run", self.run_combo), ("Split", self.split_combo), ("Metric", self.metric_combo), ("Checkpoint", self.checkpoint_combo), ("Tolerance", self.ruler_combo)):
            selectors.addWidget(QLabel(label))
            selectors.addWidget(widget)
        selectors.addWidget(refresh)
        layout.addLayout(selectors)
        self.status_label = QLabel("Ready")
        self.summary_label = QLabel("")
        layout.addWidget(self.status_label)
        layout.addWidget(self.summary_label)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.plot = pg.PlotWidget(title="Training metrics")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        splitter.addWidget(self.plot)
        self.table = QTableWidget()
        self.table.setSortingEnabled(True)
        splitter.addWidget(self.table)
        layout.addWidget(splitter)

        self.group_combo.currentIndexChanged.connect(self._group_changed)
        self.run_combo.currentIndexChanged.connect(self._load_selected)
        self.split_combo.currentIndexChanged.connect(self._split_changed)
        self.metric_combo.currentIndexChanged.connect(self._render)
        self.checkpoint_combo.currentIndexChanged.connect(self._render)
        self.ruler_combo.currentIndexChanged.connect(self._render)
        refresh.clicked.connect(self.refresh)
        self.refresh()

    def refresh(self):
        self.request_id += 1
        try:
            groups = list_groups(self.data_root)
        except Exception as error:
            self.status_label.setText(str(error))
            return
        current = self.group_combo.currentData()
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        for group in groups:
            self.group_combo.addItem(group.name, group.id)
        if current:
            position = self.group_combo.findData(current)
            if position >= 0:
                self.group_combo.setCurrentIndex(position)
        else:
            for position, group in enumerate(groups):
                if list_runs(self.data_root, group.id):
                    self.group_combo.setCurrentIndex(position)
                    break
        self.group_combo.blockSignals(False)
        self._group_changed()

    def _group_changed(self):
        self.request_id += 1
        group_id = self.group_combo.currentData()
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        if group_id:
            try:
                runs = list_runs(self.data_root, group_id)
            except Exception as error:
                self.status_label.setText(str(error))
                runs = ()
            for run_id in reversed(runs):
                self.run_combo.addItem(run_id, run_id)
        self.run_combo.blockSignals(False)
        self._load_selected()

    def _load_selected(self):
        self.request_id += 1
        group_id = self.group_combo.currentData()
        run_id = self.run_combo.currentData()
        self.snapshot = None
        self.plot.clear()
        self.table.setRowCount(0)
        if not group_id or not run_id:
            self.status_label.setText("No published run selected")
            return
        self.status_label.setText(f"Loading {group_id}/{run_id}…")
        task = LoadTask(self.data_root, group_id, run_id, self.request_id)
        self.tasks.append(task)
        task.signals.loaded.connect(self._loaded)
        task.signals.failed.connect(self._failed)
        self.pool.start(task)

    def _loaded(self, request_id, snapshot):
        if request_id != self.request_id:
            return
        self.snapshot = snapshot
        self.status_label.setText(f"{snapshot.group.name} / {snapshot.run_id} — {snapshot.status}, generation {snapshot.generation}")
        for combo in (self.split_combo, self.checkpoint_combo, self.ruler_combo):
            combo.blockSignals(True)
            combo.clear()
        for split in snapshot.splits:
            self.split_combo.addItem(f"{split.name} ({split.role})", split.id)
        for checkpoint in snapshot.checkpoints:
            self.checkpoint_combo.addItem(f"{checkpoint.id} (update {checkpoint.update})", checkpoint.id)
        if snapshot.selected_checkpoint:
            self.checkpoint_combo.setCurrentIndex(self.checkpoint_combo.findData(snapshot.selected_checkpoint))
        for ruler in snapshot.tolerances:
            self.ruler_combo.addItem(ruler, ruler)
        self.ruler_combo.setCurrentIndex(self.ruler_combo.findData(snapshot.stopping_ruler))
        for combo in (self.split_combo, self.checkpoint_combo, self.ruler_combo):
            combo.blockSignals(False)
        self._split_changed()

    def _failed(self, request_id, message):
        if request_id == self.request_id:
            self.status_label.setText(message)

    def _split_changed(self):
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        if self.snapshot and self.split_combo.currentData():
            for name in available_metrics(self.snapshot, self.split_combo.currentData()):
                self.metric_combo.addItem(name, name)
            position = self.metric_combo.findData("standardized_mse")
            if position >= 0:
                self.metric_combo.setCurrentIndex(position)
        self.metric_combo.blockSignals(False)
        self._render()

    def _render(self):
        self.plot.clear()
        if not self.snapshot:
            return
        split_id = self.split_combo.currentData()
        metric_name = self.metric_combo.currentData()
        if split_id and metric_name:
            curve = prepare_curve(self.snapshot, split_id, metric_name)
            self.plot.setTitle(f"{curve.split_name}: {curve.metric}")
            self.plot.setLabel("bottom", "Optimizer update")
            self.plot.setLabel("left", curve.metric, units=curve.unit or None)
            if curve.points:
                self.plot.plot([point[0] for point in curve.points], [point[1] for point in curve.points], pen=pg.mkPen("#268bd2", width=2), symbol="o")
                self.plot.getViewBox().autoRange()
        checkpoint_id = self.checkpoint_combo.currentData()
        ruler = self.ruler_combo.currentData()
        if not (split_id and checkpoint_id and ruler):
            self.summary_label.setText("Predictions unavailable")
            self.table.setRowCount(0)
            return
        try:
            report = analyze_failures(self.snapshot, split_id, checkpoint_id, ruler)
        except ValueError:
            self.summary_label.setText("Predictions unavailable for this split/checkpoint")
            self.table.setRowCount(0)
            return
        self.summary_label.setText(f"{report.failed} failed / {report.total} samples ({ruler})")
        headers = ["row_id", "motion_group", *self.snapshot.input_columns, *self.snapshot.target_columns, *(f"pred_{name}" for name in self.snapshot.target_columns), *(f"error_{name}" for name in self.snapshot.target_columns), "severity"]
        self.table.setSortingEnabled(False)
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        failed_rows = [row for row in report.rows if not row.passed]
        self.table.setRowCount(len(failed_rows))
        for index, row in enumerate(failed_rows):
            values = (row.row_id, row.motion_group, *row.inputs, *row.targets, *row.predictions, *row.signed_errors, row.severity)
            for column, value in enumerate(values):
                item = QTableWidgetItem(f"{value:.6g}" if isinstance(value, float) else str(value))
                self.table.setItem(index, column, item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
