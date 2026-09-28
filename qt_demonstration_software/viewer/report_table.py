"""Sortable, intersecting filters and exact-value CSV export."""
import csv
from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt

from .contracts import ViewerError

HEADERS = ("split", "row_id", "motion_group", "breakaway", "v0_m_s", "force_N", "force_duration_s", "observation_time_s", "displacement_m", "v_final_m_s", "pred_displacement_m", "pred_v_final_m_s", "signed_displacement_m", "signed_velocity_m_s", "abs_displacement_m", "abs_velocity_m_s", "failure_type", "severity")


class FailureTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []

    def set_result(self, result, ruler):
        self.beginResetModel()
        self.rows = []
        if result is not None:
            tolerance = self.tolerances[ruler]
            for index, passed in enumerate(result.pass_masks[ruler]):
                if passed:
                    continue
                displacement_fail = result.absolute_errors[index, 0] >= tolerance[0]
                velocity_fail = result.absolute_errors[index, 1] >= tolerance[1]
                failure = "两者失败" if displacement_fail and velocity_fail else "仅位移失败" if displacement_fail else "仅速度失败"
                self.rows.append((result.split, result.row_ids[index], result.groups[index], bool(result.breakaway_mask[index]), *result.inputs[index], *result.targets[index], *result.predictions[index], *result.signed_errors[index], *result.absolute_errors[index], failure, max(result.absolute_errors[index, 0] / tolerance[0], result.absolute_errors[index, 1] / tolerance[1])))
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return len(HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return HEADERS[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        value = self.rows[index.row()][index.column()]
        if role == Qt.UserRole:
            return value
        if role == Qt.DisplayRole:
            return f"{value:.6g}" if isinstance(value, float) else str(value)
        return None


class FailureFilterProxy(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.group = "全部"
        self.failure = "全部"
        self.breakaway = "全部"
        self.row_id = ""
        self.setSortRole(Qt.UserRole)

    def set_filters(self, group="全部", failure="全部", breakaway="全部", row_id=""):
        self.group, self.failure, self.breakaway, self.row_id = group, failure, breakaway, row_id.casefold()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        row = self.sourceModel().rows[source_row]
        return (self.group == "全部" or row[2] == self.group) and (self.failure == "全部" or row[16] == self.failure) and (self.breakaway == "全部" or row[3] == (self.breakaway == "是")) and self.row_id in row[1].casefold()


def export_filtered(proxy, destination):
    path = Path(destination).resolve()
    source = proxy.sourceModel()
    protected = getattr(source, "protected_paths", ())
    if any(path == item or path.is_relative_to(item) for item in protected):
        raise ViewerError(f"不可覆盖训练产物或标签: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADERS)
        for index in range(proxy.rowCount()):
            writer.writerow(source.rows[proxy.mapToSource(proxy.index(index, 0)).row()])
    return proxy.rowCount()
