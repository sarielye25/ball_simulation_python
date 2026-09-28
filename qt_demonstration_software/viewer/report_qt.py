"""Qt Widgets interface for saved training runs."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QSplitter, QTableView, QTextEdit, QVBoxLayout, QWidget)
from PySide6.QtGui import QStandardItem, QStandardItemModel

from .contracts import GROUPS, RULERS, SCOPES, State
from .report_data import build_summary
from .report_table import FailureFilterProxy, FailureTableModel, export_filtered
from .worker import WorkerController
from .training_adapter import TRAINING_ROOT

METRICS = ("standardized_mse", "displacement_standardized_mse", "velocity_standardized_mse", "displacement_loss_contribution", "velocity_loss_contribution", "displacement_mae_m", "displacement_p95_m", "displacement_max_m", "velocity_mae_m_s", "velocity_p95_m_s", "velocity_max_m_s", "pass_rate_coarse", "pass_rate_intermediate", "pass_rate_fine")
COLORS = {"overall": "#e8e8e8", "always_resting": "#4caf50", "moved_then_stopped": "#ff9800", "moving_at_observation": "#2196f3", "breakaway": "#e91e63"}


class MainWindow(QMainWindow):
    def __init__(self, run=None, labels=None, demo=False):
        super().__init__()
        pg.setConfigOptions(useOpenGL=False)
        self.setWindowTitle("训练结果查看器")
        self.resize(1440, 900)
        self.setMinimumSize(960, 640)
        self.generation = 0
        self.request_id = 0
        self.bundle = None
        self.prediction = None
        self.demo = demo
        self.state = State.EMPTY
        self.closing = False
        self.worker = WorkerController(self)
        self.worker.result.connect(self._result)
        self.worker.error.connect(self._error)
        self.worker.idle.connect(self._idle)
        self._build()
        if run:
            self.runPathEdit.setText(str(Path(run).resolve()))
            self.labelsPathEdit.setText(str(Path(labels).resolve()) if labels else "")
            self.refresh()

    def _build(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)
        paths = QHBoxLayout()
        self.runPathEdit = QLineEdit(objectName="runPathEdit", placeholderText="实验目录（含 run.json）")
        self.labelsPathEdit = QLineEdit(objectName="labelsPathEdit", placeholderText="标签目录（可留空）")
        browse = QPushButton("选择实验")
        browse.clicked.connect(self._browse)
        self.refreshButton = QPushButton("刷新", objectName="refreshButton")
        self.refreshButton.clicked.connect(self.refresh)
        for widget in (self.runPathEdit, browse, self.labelsPathEdit, self.refreshButton):
            paths.addWidget(widget)
        layout.addLayout(paths)
        self.statusLabel = QLabel("空窗口", objectName="statusLabel")
        layout.addWidget(self.statusLabel)
        self.mainSplitter = QSplitter(Qt.Horizontal)
        self.summary = QTextEdit(readOnly=True)
        self.mainSplitter.addWidget(self.summary)
        charts = QSplitter(Qt.Vertical)
        self.trainPlot = pg.PlotWidget(objectName="trainPlot", title="训练")
        self.validationPlot = pg.PlotWidget(objectName="validationPlot", title="验证")
        self.validationPlot.setXLink(self.trainPlot)
        self.validationPlot.setYLink(self.trainPlot)
        charts.addWidget(self.trainPlot)
        charts.addWidget(self.validationPlot)
        self.mainSplitter.addWidget(charts)
        layout.addWidget(self.mainSplitter, 3)
        controls = QHBoxLayout()
        self.checkpointCombo = QComboBox(objectName="checkpointCombo")
        self.splitCombo = QComboBox(objectName="splitCombo")
        self.splitCombo.addItems(("validation", "train"))
        self.toleranceCombo = QComboBox(objectName="toleranceCombo")
        self.toleranceCombo.addItems(RULERS)
        self.metricCombo = QComboBox(objectName="metricCombo")
        self.metricCombo.addItems(METRICS)
        self.scopeCombo = QComboBox()
        self.scopeCombo.addItems(("全部", *SCOPES))
        self.baselineCheck = QCheckBox("基线")
        self.baselineCheck.setChecked(True)
        for widget in (self.checkpointCombo, self.splitCombo, self.toleranceCombo, self.metricCombo, self.scopeCombo, self.baselineCheck):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.groupTable = QTableView(objectName="groupTable")
        self.groupModel = QStandardItemModel(self)
        self.groupModel.setHorizontalHeaderLabels(("组", "总数", "通过", "失败", "通过率"))
        self.groupTable.setModel(self.groupModel)
        self.groupTable.setMaximumHeight(140)
        layout.addWidget(self.groupTable)
        filters = QHBoxLayout()
        self.groupFilter = QComboBox()
        self.groupFilter.addItems(("全部", *GROUPS))
        self.failureFilter = QComboBox()
        self.failureFilter.addItems(("全部", "仅位移失败", "仅速度失败", "两者失败"))
        self.breakawayFilter = QComboBox()
        self.breakawayFilter.addItems(("全部", "是", "否"))
        self.rowSearch = QLineEdit(placeholderText="搜索 row ID")
        self.countLabel = QLabel("失败 0 / 筛选 0")
        self.exportButton = QPushButton("导出筛选结果", objectName="exportButton")
        self.exportButton.setEnabled(False)
        for widget in (self.groupFilter, self.failureFilter, self.breakawayFilter, self.rowSearch, self.countLabel, self.exportButton):
            filters.addWidget(widget)
        layout.addLayout(filters)
        self.failureTable = QTableView(objectName="failureTable")
        self.tableModel = FailureTableModel(self)
        self.proxy = FailureFilterProxy(self)
        self.proxy.setSourceModel(self.tableModel)
        self.failureTable.setModel(self.proxy)
        layout.addWidget(self.failureTable, 2)
        self.checkpointCombo.currentIndexChanged.connect(self._diagnose)
        self.splitCombo.currentIndexChanged.connect(self._diagnose)
        self.toleranceCombo.currentIndexChanged.connect(self._show_prediction)
        self.metricCombo.currentIndexChanged.connect(self._draw)
        self.scopeCombo.currentIndexChanged.connect(self._draw)
        self.baselineCheck.toggled.connect(self._draw)
        for signal in (self.groupFilter.currentIndexChanged, self.failureFilter.currentIndexChanged, self.breakawayFilter.currentIndexChanged, self.rowSearch.textChanged):
            signal.connect(self._filter)
        self.exportButton.clicked.connect(self._export)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.mainSplitter.setOrientation(Qt.Vertical if self.width() < 1200 else Qt.Horizontal)

    def _browse(self):
        chosen = QFileDialog.getExistingDirectory(self, "选择实验目录", self.runPathEdit.text())
        if chosen:
            self.runPathEdit.setText(chosen)
            self.refresh()

    def refresh(self):
        self.generation += 1
        self.request_id += 1
        self.bundle = self.prediction = None
        self.tableModel.beginResetModel()
        self.tableModel.rows = []
        self.tableModel.endResetModel()
        self.exportButton.setEnabled(False)
        self.summary.clear()
        self.trainPlot.clear()
        self.validationPlot.clear()
        self.checkpointCombo.clear()
        self.worker.clear_cache()
        path = self.runPathEdit.text().strip()
        if not path:
            self.state = State.EMPTY
            self.statusLabel.setText("空窗口")
            return
        self.state = State.LOADING
        self.statusLabel.setText("正在加载…")
        labels = self.labelsPathEdit.text().strip() or None
        self.worker.submit(self.generation, self.request_id, "load", (Path(path).resolve(), Path(labels).resolve() if labels else None))

    def _result(self, generation, request_id, value):
        if self.closing or (generation, request_id) != (self.generation, self.request_id):
            return
        if hasattr(value, "manifest"):
            self.bundle = replace(value, generation=generation, demo=self.demo)
            self.state = State.READY
            self.statusLabel.setText(("演示数据" if self.demo else "真实实验") + f"  {value.run_dir}")
            self.summary.setPlainText(build_summary(self.bundle))
            self.toleranceCombo.setCurrentText(value.manifest["config"].get("stopping_ruler", "intermediate"))
            self._draw()
            self.checkpointCombo.blockSignals(True)
            self.checkpointCombo.addItems([item["file"] for item in (value.index or {}).get("snapshots", [])])
            default = (value.index or {}).get("selected") or (value.index or {}).get("last")
            self.checkpointCombo.setCurrentText(default or "")
            self.checkpointCombo.blockSignals(False)
            self._diagnose()
        else:
            self.prediction = value
            self.state = State.READY
            self._show_prediction()

    def _error(self, generation, request_id, message):
        if self.closing or (generation, request_id) != (self.generation, self.request_id):
            return
        self.state = State.ERROR
        self.statusLabel.setText(f"错误：{message}")
        self.exportButton.setEnabled(False)

    def _idle(self):
        if self.closing:
            self.close()

    def _diagnose(self):
        self.request_id += 1
        self.prediction = None
        self.exportButton.setEnabled(False)
        self.tableModel.beginResetModel()
        self.tableModel.rows = []
        self.tableModel.endResetModel()
        self.groupModel.removeRows(0, self.groupModel.rowCount())
        self._filter()
        if not self.bundle or not self.checkpointCombo.currentText():
            return
        split = self.splitCombo.currentText()
        if split not in self.bundle.splits:
            self.statusLabel.setText(self.bundle.split_errors.get(split, "数据集不可用"))
            return
        checkpoint = self.checkpointCombo.currentText()
        key = (str(self.bundle.run_dir), self.generation, checkpoint, split, self.bundle.splits[split].sha256)
        cached = self.worker.cache.get(key)
        if cached is not None:
            self.worker.cache.move_to_end(key)
            self._result(self.generation, self.request_id, cached)
            return
        self.state = State.INFERENCING
        self.statusLabel.setText(f"正在推理：{checkpoint} / {split}")
        self.worker.submit(self.generation, self.request_id, "infer", (self.bundle, checkpoint, split))
        self._cache_key = key

    def _show_prediction(self):
        if not self.prediction or not self.bundle:
            return
        result = self.prediction
        ruler = self.toleranceCombo.currentText()
        self.groupModel.removeRows(0, self.groupModel.rowCount())
        for scope in SCOPES:
            summary = result.summaries[scope]
            count = summary["count"]
            rate = summary["pass_rates"][ruler]
            passed = round(rate * count) if rate is not None else 0
            self.groupModel.appendRow([QStandardItem(str(value)) for value in (scope, count, passed, count - passed, "无定义" if rate is None else f"{rate * 100:.2f}%")])
        self.tableModel.tolerances = self.bundle.manifest["config"]["tolerances"]
        self.tableModel.protected_paths = tuple(path for path in (self.bundle.run_dir, TRAINING_ROOT / "labels", self.bundle.splits[result.split].source.parent, Path(self.labelsPathEdit.text()).resolve() if self.labelsPathEdit.text() else None) if path is not None)
        self.tableModel.set_result(result, ruler)
        self.proxy.sort(17, Qt.DescendingOrder)
        self._filter()
        self.exportButton.setEnabled(result.verified)
        self.statusLabel.setText("核对通过" if result.verified else "不一致：" + "; ".join(result.verification[:3]))
        if result.verified:
            key = (str(self.bundle.run_dir), self.generation, result.checkpoint_file, result.split, self.bundle.splits[result.split].sha256)
            self.worker.cache[key] = result
            while len(self.worker.cache) > 4 or sum(sum(array.nbytes for array in (item.inputs, item.targets, item.predictions, item.signed_errors, item.absolute_errors)) for item in self.worker.cache.values()) > 128 * 1024 * 1024:
                self.worker.cache.popitem(last=False)

    def _filter(self):
        self.proxy.set_filters(self.groupFilter.currentText(), self.failureFilter.currentText(), self.breakawayFilter.currentText(), self.rowSearch.text())
        self.countLabel.setText(f"失败 {self.tableModel.rowCount()} / 筛选 {self.proxy.rowCount()}")

    def _export(self):
        if not self.prediction or not self.prediction.verified:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", str(Path.home() / "failures.csv"), "CSV (*.csv)")
        if path:
            try:
                count = export_filtered(self.proxy, path)
                self.statusLabel.setText(f"已导出 {count} 行：{path}")
            except Exception as error:
                self.statusLabel.setText(f"导出失败：{error}")

    def _draw(self):
        if not self.bundle:
            return
        metric = self.metricCombo.currentText()
        scope = self.scopeCombo.currentText()
        for split, plot in (("train", self.trainPlot), ("validation", self.validationPlot)):
            plot.clear()
            plot.addLegend()
            for selected_scope in SCOPES if scope == "全部" else (scope,):
                rows = [row for row in self.bundle.evaluations if row["split"] == split and row["scope"] == selected_scope]
                if rows:
                    plot.plot([row["update"] for row in rows], [np.nan if row[metric] is None else row[metric] * (100 if metric.startswith("pass_rate") else 1) for row in rows], pen=pg.mkPen(COLORS[selected_scope], width=2), connect="finite", name=selected_scope)
            if split == "train" and self.baselineCheck.isChecked():
                for predictor in ("zero", "constant_velocity"):
                    row = next((item for item in self.bundle.baselines if item["predictor"] == predictor and item["scope"] == "overall"), None)
                    if row and row[metric] is not None:
                        plot.addItem(pg.InfiniteLine(pos=row[metric] * (100 if metric.startswith("pass_rate") else 1), angle=0, pen=pg.mkPen("#888888", style=Qt.DashLine)))
            for event in self.bundle.events:
                if float(event["lr_next"]) < float(event["lr_before"]):
                    plot.addItem(pg.InfiniteLine(pos=int(event["update"]), angle=90, pen=pg.mkPen("#ffc107", style=Qt.DotLine)))
            if self.bundle.index and self.bundle.index.get("selected"):
                entry = next(item for item in self.bundle.index["snapshots"] if item["file"] == self.bundle.index["selected"])
                plot.addItem(pg.InfiniteLine(pos=entry["update"], angle=90, pen=pg.mkPen("#f44336", style=Qt.DashLine)))
            plot.setLabel("bottom", "optimizer update")
            plot.setLabel("left", metric + (" (%)" if metric.startswith("pass_rate") else ""))
            if metric.startswith("pass_rate"):
                plot.setYRange(0, 100)

    def closeEvent(self, event):
        if self.worker.thread is not None:
            self.closing = True
            self.state = State.CLOSING
            self.worker.close()
            event.ignore()
        else:
            event.accept()
