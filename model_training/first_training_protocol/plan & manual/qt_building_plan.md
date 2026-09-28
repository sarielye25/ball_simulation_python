# Qt 训练结果查看器：建设计划

## 总体职责

使用 PySide6 + PyQtGraph 构建本地只读查看器，读取已结束或已中止的 transmodel 训练产物，解释运行结果、展示完整学习过程，并定位各 checkpoint 的运动组及逐样本失败。训练进程独立运行；查看器不修改权重、模型排名或原始记录，不读取保留测试集。第一版不提供实时监视、恢复训练、多实验比较、模型动画或打包。

顶部提供训练目录、标签目录、刷新、当前路径和真实/演示标志；中部用可拖动 `QSplitter` 放置总结与训练/验证双图；底部放置 checkpoint、数据集、容差选择器、运动组汇总和失败表。小窗口允许上下排列。

## 第一部分：运行概览与可信数据

### 功能 1：加载、校验与状态展示

- **职责：** 快速展示实验状态；只有数据身份通过校验才生成逐样本详情。
- **选用模块：** `report.py`（入口）、`report_data.py`（解析与校验）、`report_qt.py`（状态界面）；复用现有 `data.py`、`checkpoints.py`。
- **技术路线：** 先读 `run.json`、`evaluations.csv`、`evaluation_events.csv`、`batches.csv`、`baselines.csv`、`termination.json` 和 `checkpoints/index.json`；校验 schema、列、维度与有限值，报告具体文件及错误。标签默认取脚本旁 `labels/`，也可指定；按 `run.json` 中 SHA-256、行数和稳定 row ID 校验训练/验证 CSV。不匹配时禁用失败详情。未结束实验标为“未知/未完成”。

### 功能 2：确定性文字总结

- **职责：** 用已保存的事实解释运行、停止和模型选择，不推断失败原因。
- **选用模块：** `report_data.py` 生成结构化摘要，`report_qt.py` 显示。
- **技术路线：** 展示完成 update/epoch、曝光次数、耗时、停止原因及全部停止标志；区分 `selected` 与最后有效 checkpoint。读取本次 `run_config` 的容差和目标通过率，列出选中模型的训练/验证通过率、双集合达标状态、验证失败类型及各组失败率；比较 selected/last 的验证 MSE 与通过率。按记录翻译 `TARGET_REACHED`、`NO_PROGRESS`、`MAX_UPDATES`、`NUMERICAL_FAILURE`；中断或一般错误显示原始类型与信息。没有 selected 时标注“无正式选中模型”。

## 第二部分：完整学习曲线

### 功能 3：训练/验证双图与交互

- **职责：** 按真实 optimizer update 展示全部已保存评估点，比较训练与验证趋势、组差异和停止位置。
- **选用模块：** `report_data.py` 整理 `evaluations.csv`、`evaluation_events.csv` 和 `baselines.csv`；`report_qt.py` 用 PyQtGraph `PlotWidget`。参考 [linkedViews.py](https://github.com/pyqtgraph/pyqtgraph/blob/master/pyqtgraph/examples/linkedViews.py)、[crosshair.py](https://github.com/pyqtgraph/pyqtgraph/blob/master/pyqtgraph/examples/crosshair.py)、[InfiniteLine.py](https://github.com/pyqtgraph/pyqtgraph/blob/master/pyqtgraph/examples/InfiniteLine.py) 与 [ScatterPlot.py](https://github.com/pyqtgraph/pyqtgraph/blob/master/pyqtgraph/examples/ScatterPlot.py)。
- **技术路线：**

- X 轴为 optimizer update。显示全部已保存评估点，不平滑、不截去早期数据。
- Y 轴通过下拉或按钮切换：总标准化 MSE，位移/速度各自标准化 MSE，两输出对总损失的贡献，位移/速度 MAE、P95、最大绝对误差，以及 coarse/intermediate/fine 联合通过率。
- 同一指标的训练/验证图共享 Y 轴范围。通过率始终 0–100%；其他指标显示单位，不能把标准化 MSE 标为米或米/秒。
- 支持整体、三个互斥运动组、临界力切片的显示/隐藏；固定配色，图例显示样本数。临界力切片与运动组重叠，不参与组计数求和。
- 零预测和恒速预测基线读取 baselines.csv，仅在已有对应数据的训练图显示，不臆造验证基线。
- 标记 selected checkpoint、最后评估位置、实际学习率下降位置。支持平移、缩放、重置视图和点悬停显示 update、指标值、组名与样本数。
- 空组指标显示无定义，曲线留缺口；不得用零替代。默认线性坐标。

## 第三部分：checkpoint 诊断与失败样本

### 功能 4：每个 checkpoint 的组表现

- **职责：** 在指定数据集和容差下核对整体、运动组及 breakaway 切片的表现。
- **选用模块：** `report_data.py` 复用 `metrics.py`、`checkpoints.py`、`neural_network.py`、`data.py`；`report_qt.py` 显示汇总表。
- **技术路线：**

切换 checkpoint 后，分别显示整体、always_resting、moved_then_stopped、moving_at_observation，以及单独标明的 breakaway 切片。

列：组名称、样本总数、通过数、失败数、当前容差下通过率。通过率必须是位移与速度同时满足严格小于容差的联合通过率，不能取两输出通过率的平均。空组显示 count=0、通过率无定义。

快速显示保存的评估摘要；推理结果加载后应核对一致性，若不一致，显示错误而不是悄悄覆盖。

### 功能 5：失败表、筛选与导出

- **职责：** 追溯任一 checkpoint 在训练或验证集上的失败输入，并按严重程度排序。
- **选用模块：** `report_table.py` 使用 `QTableView`、`QAbstractTableModel`、`QSortFilterProxyModel`；参考 [PySide6 表格模型](https://github.com/pyside/pyside-setup/blob/dev/examples/external/pandas/dataframe_model.py) 与 [过滤示例](https://github.com/pyside/pyside-setup/blob/dev/examples/widgets/itemviews/basicfiltermodel/basicsortfiltermodel.py)。
- **技术路线：**

每个 checkpoint 均可查看训练集和验证集失败。使用 QTableView + QAbstractTableModel，配合 QSortFilterProxyModel；避免为每个单元格创建 widget。

必备列：

| 类别 | 字段 |
| --- | --- |
| 追溯 | split、稳定 row ID、运动组、是否临界力切片 |
| 四个输入 | v0（m/s）、force（N）、force duration（s）、observation time（s） |
| 两个目标 | 真实位移（m）、真实末速度（m/s） |
| 两个预测 | 预测位移（m）、预测末速度（m/s） |
| 误差 | 位移/速度的带符号误差及绝对误差 |
| 分类 | 仅位移失败、仅速度失败、两者失败 |
| 严重程度 | max(abs(d_error)/tau_d, abs(v_error)/tau_v) |

误差定义为预测减真实；严重程度 >= 1 即失败。全部判断使用未格式化数值，显示舍入不能改变通过结果。

默认严重程度降序，数值列必须按数值排序。支持组、失败类型、临界力切片和 row ID 搜索；表格上方同时显示总失败数与筛选后数量。提供筛选后全部行的 CSV 导出，不限于可见行。可选显示筛选后失败样本的均值/P95/最大误差，明确统计范围；零失败时失败统计无定义。

### 功能 6：后台计算与交互一致性

- **职责：** 保持界面可响应，防止旧请求覆盖当前 checkpoint。
- **选用模块：** PySide6 `QThread`/worker；参考 [官方线程信号示例](https://github.com/pyside/pyside-setup/tree/dev/examples/widgets/thread_signals)。[pytest-qt](https://github.com/pytest-dev/pytest-qt) 用于交互及异步测试；[superqt worker](https://github.com/pyapp-kit/superqt/blob/main/src/superqt/utils/_qthreading.py) 仅作可选参考。
- **技术路线：**

打开实验先读取小型 JSON/CSV，快速展示曲线与运行状态。仅在选择 checkpoint 时加载模型并推理；一次预测服务于三套容差切换。缓存以实验路径、checkpoint 标识、数据哈希为键，采用有限容量，避免所有 checkpoint 的逐样本数据常驻内存。

QThread/worker 负责耗时计算，主线程才可操作 Qt 控件。快速切换时用请求编号丢弃过期结果，避免旧 checkpoint 覆盖新选择。忙碌状态明确显示，重复请求可合并。关闭窗口时请求取消并安全退出工作线程，不强杀线程。CPU 推理使用 eval 和 inference_mode，不扰动训练随机状态。

遇到文件缺失、损坏、不支持的 schema、维度不符、哈希不符和非有限数值，报告具体文件与原因。空实验显示空状态。第一版允许要求“训练已停止再打开”，但不能因读取中间状态崩溃。仅加载可信的本地产物，沿用现有 weights_only=True 加载方式。

参考项目的许可证、版本限制与示例细节见 [qt_github_references.txt](qt_github_references.txt)。实施时核对所装版本；PySide6 `dev` 分支的过滤示例不可直接当作稳定 API。
