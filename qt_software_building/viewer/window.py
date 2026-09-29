"""Qt window for published training results."""

from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
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
        self.reference_snapshot = None
        self.request_id = 0
        self.pool = QThreadPool.globalInstance()
        self.tasks = []
        self.setWindowTitle("Training Result Viewer")
        self.resize(1250, 820)

        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)
        primary_selectors = QHBoxLayout()
        analysis_selectors = QHBoxLayout()
        self.group_combo = QComboBox()
        self.run_combo = QComboBox()
        self.split_combo = QComboBox()
        self.metric_combo = QComboBox()
        self.checkpoint_combo = QComboBox()
        self.ruler_combo = QComboBox()
        self.scope_combo = QComboBox()
        self.reference_combo = QComboBox()
        self.log_axis = QCheckBox("Log MSE axis")
        self.log_axis.setChecked(True)
        refresh = QPushButton("Refresh")
        for label, widget in (("Group", self.group_combo), ("Run", self.run_combo),
                              ("Split", self.split_combo), ("Tolerance", self.ruler_combo)):
            primary_selectors.addWidget(QLabel(label))
            primary_selectors.addWidget(widget)
        primary_selectors.addWidget(refresh)
        for label, widget in (("Scope", self.scope_combo), ("Metric", self.metric_combo),
                              ("Checkpoint", self.checkpoint_combo)):
            analysis_selectors.addWidget(QLabel(label))
            analysis_selectors.addWidget(widget)
        layout.addLayout(primary_selectors)
        layout.addLayout(analysis_selectors)
        self.status_label = QLabel("Ready")
        self.summary_label = QLabel("")
        layout.addWidget(self.status_label)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        overview_layout.addWidget(self.log_axis)
        self.plot = pg.PlotWidget(title="Training metrics")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend()
        overview_layout.addWidget(self.plot)
        self.metrics_toggle = QPushButton("▲ Group results")
        self.metrics_toggle.setCheckable(True)
        self.metrics_toggle.setChecked(True)
        overview_layout.addWidget(self.metrics_toggle)
        self.metrics_table = QTableWidget()
        overview_layout.addWidget(self.metrics_table)
        self.tabs.addTab(overview, "Learning and groups")
        comparison = QWidget()
        comparison_layout = QVBoxLayout(comparison)
        comparison_selectors = QHBoxLayout()
        comparison_selectors.addWidget(QLabel("Reference run"))
        comparison_selectors.addWidget(self.reference_combo)
        self.compare_button = QPushButton("Compare at selected update")
        comparison_selectors.addWidget(self.compare_button)
        comparison_layout.addLayout(comparison_selectors)
        self.comparison_label = QLabel("Select a reference run")
        comparison_layout.addWidget(self.comparison_label)
        self.comparison_table = QTableWidget()
        comparison_layout.addWidget(self.comparison_table)
        self.tabs.addTab(comparison, "Run comparison")
        failures = QWidget()
        self.failures_tab = failures
        failures_layout = QVBoxLayout(failures)
        failures_layout.addWidget(self.summary_label)
        self.table = QTableWidget()
        self.table.setSortingEnabled(True)
        failures_layout.addWidget(self.table)
        self.tabs.addTab(failures, "Failed samples")
        self.tabs.currentChanged.connect(self._render)

        self.group_combo.currentIndexChanged.connect(self._group_changed)
        self.run_combo.currentIndexChanged.connect(self._load_selected)
        self.split_combo.currentIndexChanged.connect(self._split_changed)
        self.metric_combo.currentIndexChanged.connect(self._render)
        self.checkpoint_combo.currentIndexChanged.connect(self._render)
        self.ruler_combo.currentIndexChanged.connect(self._render)
        self.scope_combo.currentIndexChanged.connect(self._render)
        self.log_axis.toggled.connect(self._render)
        self.metrics_toggle.toggled.connect(self._toggle_metrics)
        self.compare_button.clicked.connect(self._load_reference)
        refresh.clicked.connect(self.refresh)
        self.refresh()

    def _toggle_metrics(self, expanded):
        self.metrics_table.setVisible(expanded)
        self.metrics_toggle.setText(
            "▲ Group results" if expanded else "▼ Group results"
        )

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
        self.reference_combo.clear()
        for group in groups:
            for run_id in reversed(list_runs(self.data_root, group.id)):
                self.reference_combo.addItem(f"{group.name} / {run_id}", (group.id, run_id))
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
        self.reference_snapshot = None
        self.plot.clear()
        self.table.setRowCount(0)
        self.metrics_table.setRowCount(0)
        self.comparison_table.setRowCount(0)
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
        reason = snapshot.termination.get("reason") if snapshot.termination else "in progress"
        self.status_label.setText(f"Stopping reason: {reason}    Generation: {snapshot.generation}")
        for combo in (self.split_combo, self.checkpoint_combo, self.ruler_combo):
            combo.blockSignals(True)
            combo.clear()
        for split in snapshot.splits:
            self.split_combo.addItem(f"{split.name} ({split.role})", split.id)
        if any(split.role == "train" for split in snapshot.splits) and any(
                split.role == "validation" for split in snapshot.splits):
            self.split_combo.addItem("Train + Validation", "train+validation")
            self.split_combo.setCurrentIndex(self.split_combo.findData("train+validation"))
        for checkpoint in snapshot.checkpoints:
            self.checkpoint_combo.addItem(f"{checkpoint.id} (update {checkpoint.update})", checkpoint.id)
        if snapshot.selected_checkpoint:
            self.checkpoint_combo.setCurrentIndex(self.checkpoint_combo.findData(snapshot.selected_checkpoint))
        for ruler in snapshot.tolerances:
            self.ruler_combo.addItem(ruler, ruler)
        self.ruler_combo.setCurrentIndex(self.ruler_combo.findData(snapshot.stopping_ruler))
        for combo in (self.split_combo, self.checkpoint_combo, self.ruler_combo):
            combo.blockSignals(False)
        self.scope_combo.blockSignals(True)
        self.scope_combo.clear()
        scopes = dict.fromkeys(metric.scope for metric in snapshot.metrics if metric.predictor == "model")
        for scope in scopes:
            self.scope_combo.addItem(scope.replace("_", " ").title(), scope)
        self.scope_combo.blockSignals(False)
        self._split_changed()

    def _load_reference(self):
        if not self.snapshot or not self.reference_combo.currentData():
            return
        self.reference_snapshot = None
        self.comparison_label.setText("Loading reference run…")
        group_id, run_id = self.reference_combo.currentData()
        task = LoadTask(self.data_root, group_id, run_id, self.request_id)
        self.tasks.append(task)
        task.signals.loaded.connect(self._reference_loaded)
        task.signals.failed.connect(self._reference_failed)
        self.pool.start(task)

    def _reference_loaded(self, request_id, snapshot):
        if request_id == self.request_id:
            self.reference_snapshot = snapshot
            self._render_comparison()

    def _reference_failed(self, request_id, message):
        if request_id == self.request_id:
            self.comparison_label.setText(message)

    def _failed(self, request_id, message):
        if request_id == self.request_id:
            self.status_label.setText(message)

    def _split_changed(self):
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        if self.snapshot and self.split_combo.currentData():
            split_ids = self._visible_split_ids()
            names = dict.fromkeys(name for split_id in split_ids
                                  for name in available_metrics(self.snapshot, split_id))
            for name in names:
                self.metric_combo.addItem(name, name)
            position = self.metric_combo.findData("standardized_mse")
            if position >= 0:
                self.metric_combo.setCurrentIndex(position)
        self.metric_combo.blockSignals(False)
        self._render()

    def _visible_split_ids(self):
        selected = self.split_combo.currentData()
        if selected == "train+validation":
            return tuple(split.id for split in self.snapshot.splits
                         if split.role in ("train", "validation"))
        return (selected,) if selected else ()

    def _render(self):
        self.plot.clear()
        if not self.snapshot:
            return
        split_id = self.split_combo.currentData()
        metric_name = self.metric_combo.currentData()
        scope = self.scope_combo.currentData() or "overall"
        self.plot.setLogMode(x=False, y=self.log_axis.isChecked() and metric_name == "standardized_mse")
        if split_id and metric_name:
            self.plot.setTitle(f"{scope.replace('_', ' ').title()}: {metric_name}")
            self.plot.setLabel("bottom", "Optimizer update")
            self.plot.setLabel("left", metric_name)
            for split_id_to_plot in self._visible_split_ids():
                split = self.snapshot.split(split_id_to_plot)
                color = "#268bd2" if split.role == "train" else "#dc6600"
                curve = prepare_curve(self.snapshot, split.id, metric_name, scope=scope)
                if curve.points:
                    self.plot.plot([point[0] for point in curve.points],
                                   [point[1] for point in curve.points],
                                   pen=pg.mkPen(color, width=2), name=split.name)
            checkpoint_id = self.checkpoint_combo.currentData()
            if checkpoint_id:
                marker = pg.InfiniteLine(pos=self.snapshot.checkpoint(checkpoint_id).update,
                                         angle=90, pen=pg.mkPen("#555555", style=Qt.PenStyle.DashLine))
                self.plot.addItem(marker)
            self.plot.getViewBox().autoRange()
        checkpoint_id = self.checkpoint_combo.currentData()
        ruler = self.ruler_combo.currentData()
        self._render_metrics()
        self._render_comparison()
        if self.tabs.currentWidget() is not self.failures_tab:
            return
        if not (split_id and checkpoint_id and ruler):
            self.summary_label.setText("Predictions unavailable")
            self.table.setRowCount(0)
            return
        if split_id == "train+validation":
            self.summary_label.setText("Select Train or Validation to inspect failed samples")
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

    @staticmethod
    def _metric(snapshot, update, split_id, scope, name):
        return next((record for record in snapshot.metrics
                     if record.update == update and record.split_id == split_id
                     and record.predictor == "model" and record.scope == scope
                     and record.name == name), None)

    @staticmethod
    def _cell(value):
        if value is None:
            return QTableWidgetItem("—")
        return QTableWidgetItem(f"{value:.6g}" if isinstance(value, float) else str(value))

    def _render_metrics(self):
        checkpoint_id = self.checkpoint_combo.currentData()
        if not self.snapshot or not checkpoint_id:
            self.metrics_table.setRowCount(0)
            return
        update = self.snapshot.checkpoint(checkpoint_id).update
        scopes = dict.fromkeys(record.scope for record in self.snapshot.metrics
                               if record.update == update and record.predictor == "model")
        headers = ("Scope", "Train n", "Validation n", "Train MSE", "Validation MSE",
                   "Train d MAE (m)", "Validation d MAE (m)",
                   "Train v MAE (m/s)", "Validation v MAE (m/s)",
                   "Train pass", "Validation pass")
        table = self.metrics_table
        table.setSortingEnabled(False)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(scopes))
        ruler = self.ruler_combo.currentData() or self.snapshot.stopping_ruler
        splits = {split.role: split.id for split in self.snapshot.splits}
        for row, scope in enumerate(scopes):
            values = [scope.replace("_", " ")]
            records = {role: self._metric(self.snapshot, update, split_id, scope, "standardized_mse")
                       for role, split_id in splits.items()}
            values.extend(records.get(role).count if records.get(role) else None
                          for role in ("train", "validation"))
            for name in ("standardized_mse", "displacement_mae_m", "velocity_mae_m_s",
                         f"pass_rate_{ruler}"):
                for role in ("train", "validation"):
                    split_id = splits.get(role)
                    record = self._metric(self.snapshot, update, split_id, scope, name) if split_id else None
                    values.append(record.value if record else None)
            for column, value in enumerate(values):
                table.setItem(row, column, self._cell(value))
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()

    def _render_comparison(self):
        if not self.snapshot or not self.reference_snapshot or not self.checkpoint_combo.currentData():
            return
        reference = self.reference_snapshot
        candidate = self.snapshot
        update = candidate.checkpoint(self.checkpoint_combo.currentData()).update
        if not any(checkpoint.update == update for checkpoint in reference.checkpoints):
            self.comparison_label.setText(f"Reference has no checkpoint at update {update}")
            self.comparison_table.setRowCount(0)
            return
        samples_differ = ({split.role: {row.row_id: (row.motion_group, row.inputs, row.targets)
                          for row in reference.samples[split.id]}
             for split in reference.splits} !=
            {split.role: {row.row_id: (row.motion_group, row.inputs, row.targets)
                          for row in candidate.samples[split.id]}
             for split in candidate.splits})
        same_normalization = (reference.normalization == candidate.normalization
                              if reference.normalization and candidate.normalization else None)
        settings_note = ""
        if reference.run_config and candidate.run_config:
            differences = sorted(name for name in set(reference.run_config) | set(candidate.run_config)
                                 if reference.run_config.get(name) != candidate.run_config.get(name))
            settings_note = f"; differing settings: {', '.join(differences) or 'none'}"
        else:
            settings_note = "; settings unavailable for at least one run"
        if not reference.normalization or not candidate.normalization:
            settings_note += "; normalization metadata unavailable"
        elif not same_normalization:
            settings_note += "; normalization differs"
        if samples_differ:
            settings_note += "; samples or targets differ"
        self.comparison_label.setText(
            f"Update {update}: {candidate.group.id}/{candidate.run_id} vs "
            f"{reference.group.id}/{reference.run_id}; lower MSE is better{settings_note}"
        )
        scopes = dict.fromkeys(record.scope for record in candidate.metrics
                               if record.update == update and record.predictor == "model")
        table = self.comparison_table
        table.setSortingEnabled(False)
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(("Split", "Scope", "Reference MSE", "Candidate MSE", "Change %"))
        rows = [(split, scope) for split in candidate.splits for scope in scopes]
        table.setRowCount(len(rows))
        for row, (split, scope) in enumerate(rows):
            matching = next((item for item in reference.splits if item.role == split.role), None)
            before = self._metric(reference, update, matching.id, scope, "standardized_mse") if matching else None
            after = self._metric(candidate, update, split.id, scope, "standardized_mse")
            old = before.value if before else None
            new = after.value if after else None
            change = 100 * (new / old - 1) if old and new is not None else None
            for column, value in enumerate((split.name, scope.replace("_", " "), old, new, change)):
                table.setItem(row, column, self._cell(value))
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()
