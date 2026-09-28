# Qt 训练结果查看器：建设计划

> AI 实施规范；本地核对日期：2026-09-28。先执行“本地实施约束”，再按功能 1–6 实现。下文路径与字段以已核对的训练源码为准。

## 本地实施约束

### 1. 目录与交付物

- 项目根目录：`D:\.physical_ai\ball_simulation_python`。
- 软件目录：`D:\.physical_ai\ball_simulation_python\qt_demonstration_software`；检查时为空。全部新增软件、测试、说明文件放在这里。
- 训练源码：`<项目根>\model_training\first_training_protocol`，下文简称 `TRAINING_ROOT`。
- 默认实验选择起点：`TRAINING_ROOT\training_data`；用户须选择含 `run.json` 的单次实验子目录。检查时只有 README，无可用于端到端验证的真实实验。
- 默认外部标签：`TRAINING_ROOT\labels`，只允许 `train.csv`、`validation.csv`。禁止读取 `test.csv`。
- 保留训练侧已有 `report.py`：它是 Matplotlib 静态图导出脚本，会写入实验目录；查看器不得调用它。
- 交付结构如下；不复制训练模块，不在训练目录新建同名 GUI 文件。

```text
qt_demonstration_software/
  app.py                         # 参数、路径、Qt 后端、应用入口
  viewer/
    __init__.py
    contracts.py                 # dataclass、状态枚举、字段/单位表
    training_adapter.py          # 唯一训练模块导入与推理适配层
    report_data.py               # 文件解析、身份校验、确定性总结
    report_qt.py                 # 主窗口与图表
    report_table.py              # 表格模型、过滤、导出
    worker.py                    # 单工作线程、取消、缓存
  tests/
    fixtures/                    # 明确标记 demo 的小型产物
    test_data.py
    test_inference.py
    test_table.py
    test_ui.py
  requirements-gui.txt
  requirements-dev.txt
  requirements-lock.txt
  pytest.ini
  README.md
```

### 2. 版本与 Windows 环境

| 项目 | 固定选择 |
| --- | --- |
| Python | 已有 CPython 3.14.7，Windows x64；先检查 `struct.calcsize('P') == 8` |
| Qt / Python 绑定 | `PySide6==6.10.2`；使用其自带 Qt 6.10.2 |
| Qt 子包 | `shiboken6`、`PySide6_Essentials`、`PySide6_Addons` 全部 6.10.2，由 PySide6 解析安装 |
| 绘图 | `pyqtgraph==0.14.0`，固定 `PYQTGRAPH_QT_LIB=PySide6`，关闭 OpenGL |
| 数值 / 推理 | 沿用根目录 requirements：`numpy==2.5.3`、`torch==2.14.0+cpu` |
| UI | Qt Widgets；程序式布局；不引入 QML、Qt WebEngine、pandas、qtpy、superqt |
| 开发测试 | `pytest`、`pytest-qt`；首次成功解析后冻结精确版本；`pytest.ini` 设置 `qt_api = pyside6` |

PyPI 已核对：PySide6 6.10.2 要求 Python `>=3.9,<3.15`，提供 `cp39-abi3-win_amd64` wheel；PyQtGraph 0.14.0 要求 Python `>=3.10`、NumPy `>=1.25.0`。这是候选构建基线，尚未安装或做 GUI 运行验证；不声称为最新版本或 LTS。参考：[PySide6 元数据](https://pypi.org/pypi/PySide6/6.10.2/json)、[PyQtGraph 元数据](https://pypi.org/pypi/pyqtgraph/0.14.0/json)。

`requirements-gui.txt` 仅列 PySide6、PyQtGraph 上述精确版本；`requirements-dev.txt` 包含 `-r requirements-gui.txt`、pytest、pytest-qt。建立软件独立虚拟环境，不修改根目录训练环境。无需单独安装 Qt SDK、Qt Creator、CMake 或 C++ 编译器。

AI 创建文件后执行以下 PowerShell 命令；每条命令失败即停止该阶段，解决后重试：

```powershell
Set-Location -LiteralPath 'D:\.physical_ai\ball_simulation_python\qt_demonstration_software'
& '..\.venv\Scripts\python.exe' -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --only-binary=:all: -r '..\requirements.txt' -r requirements-dev.txt
& '.\.venv\Scripts\python.exe' -m pip check
& '.\.venv\Scripts\python.exe' -c "import struct, PySide6, pyqtgraph, torch, numpy; from PySide6.QtCore import qVersion; assert struct.calcsize('P') == 8; print(PySide6.__version__, qVersion(), pyqtgraph.__version__, torch.__version__, numpy.__version__)"
& '.\.venv\Scripts\python.exe' -m pip freeze | Set-Content -Encoding utf8 requirements-lock.txt
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' app.py
```

锁文件必须保留 CPU wheel 索引配置，或在 README 的锁文件安装命令显式加入 `--extra-index-url https://download.pytorch.org/whl/cpu`。依赖解析失败不得静默升级训练依赖；记录冲突并更新基线后重新验证。

### 3. 启动与模块边界

- `app.py [--run PATH] [--labels PATH] [--demo]`；无参数启动空窗口。路径均转为绝对路径；不依赖当前工作目录。
- 从 `Path(__file__).resolve().parent.parent` 求项目根，再定位 `TRAINING_ROOT`。README 提供从软件目录和任意目录启动的命令。
- `training_adapter.py` 在导入前将唯一的 `TRAINING_ROOT` 放入 `sys.path`，检查 `config/data/metrics/checkpoints/neural_network` 的 `__file__` 均来自该目录；发现同名模块冲突即报错。禁止导入 `training_loop` 或调用训练入口。
- `contracts.py` 至少定义 `RunBundle`、`SplitData`、`PredictionResult`、`ViewerError`；数据对象不包含 QWidget。数组跨线程传递后不得修改。
- 对外接口：`load_run(run_dir, labels_dir=None) -> RunBundle`；`infer_checkpoint(bundle, checkpoint_file, split, cancel_event) -> PredictionResult`；`build_summary(bundle) -> str`；`export_filtered(proxy, destination) -> int`。
- `PredictionResult` 至少含 run generation、checkpoint 文件名/update、split、row IDs、物理输入/目标/预测、带符号/绝对误差、运动组、breakaway mask、三套 pass masks、核对状态。
- 软件日志写 `%LOCALAPPDATA%\BallTrainingViewer\logs`；导出路径由用户选择。不得覆盖实验产物或标签文件，写前校验目标路径。文件导出不是修改训练记录的入口。

### 4. 数据契约与读取顺序

| 来源 | 必须使用的字段 / 规则 |
| --- | --- |
| `run.json` | `format_version == 1`；`config`（不是 `run_config`）、`normalization`、`model_config`、`fixed_parameters`、`data_identity`、`software`、`artifacts` |
| `evaluations.csv` | 列来自 `training_records.FIELDS`；主键 `(update, split, predictor, scope)` 不重复；split 只取 train/validation，predictor=model；同组按整数 update 排序 |
| `baselines.csv` | 同一列规范；predictor 为 zero/constant_velocity；只使用已保存的 train 数据 |
| `evaluation_events.csv` | `update/epoch/exposures/elapsed_seconds/lr_before/lr_next/scheduler_checked/reason/should_stop/patience_reached/max_updates_reached`；`lr_next < lr_before` 才画降 LR 标记 |
| `batches.csv` | `update/epoch/batch_rows/exposures/elapsed_seconds/learning_rate/batch_mse_before_update`；不能作为全训练集评估曲线 |
| `termination.json` | completed 用 `update`；aborted 用 `last_completed_update`；读取 `reason/message/last_valid_checkpoint/selected_checkpoint`；不假设中止记录有全部停止标志 |
| `checkpoints/index.json` | `format_version == 1`、`snapshots[{file,update,validation_mse,validation_pass_rate}]`、`last/selected/best_mse/best_pass_rate/ruler`；无 selected 不自行补选 |
| checkpoint payload | `format_version == 1`、`model_state/model_config/normalization/run_config/data_identity/train_result/validation_result/update`；对照 run.json 的对应字段 |
| 数据快照 | `data_identity[split].snapshot/snapshot_sha256/rows/row_ids`；优先读取实验 `datasets/train.csv`、`datasets/validation.csv` |

- JSON 的路径引用必须解析后仍位于实验目录内；checkpoint 文件必须来自 index；拒绝越界路径。
- 快照哈希对比 `snapshot_sha256`；外部原始标签哈希对比 `sha256`。两者不可混用。
- 快照逐行读出 `row_id`、`motion_group`、四输入、两目标；核对行数、row ID 顺序/唯一性及组定义。原始标签未存 row ID 时，按当前训练规则生成 `train:2`、`validation:2` 起的 CSV 行号标识，并与清单逐项比对。
- 快照缺失时才尝试外部标签；快照存在但校验失败则阻止该 split 详情，不静默回退。
- 物理列顺序固定：`v0_m_s, force_N, force_duration_s, observation_time_s` → `displacement_m, v_final_m_s`。使用本次记录列名校验，维度 4→2。
- run/index/checkpoint 的版本字段均检查；CSV 没有独立版本字段，按 v1 表头校验。未知 schema 显示明确错误。
- count=0 允许指标空白；count>0 的必需指标空白、NaN/Inf、负计数、越界通过率均报错；空指标转换为图表缺口。
- 缺 `termination.json` 显示“未知/未完成”；缺基线或 batch/events 时禁用对应信息并提示；缺 index 仍可显示有效历史曲线。缺 run.json 或损坏的核心身份信息不能进入可信诊断。
- 读取前后比较相关文件大小与 mtime；发生变化则丢弃本次结果并提示停止训练后刷新。对截断 CSV 显示错误，不拼接推测记录。
- 显示的 epoch 保留记录语义（已进入的 epoch），不称“完整完成 epoch 数”。完成 update、最后评估 update、最后有效 checkpoint update 分别显示。
- `purpose=explicit_update_cap` 是真实短程训练，不自动等同 demo。`--demo` 只加载软件测试 fixture，并永久显示“演示数据”。

### 5. 推理与一致性实现约束

1. 后台使用 `torch.load(..., map_location='cpu', weights_only=True)` 读可信 checkpoint 元数据；不回退 `weights_only=False`。
2. v1 仅支持已存在的 `transmodel()`：`widths=[4,32,32,2]`、`activation=ReLU`、CPU float32。其他结构报“不支持的模型结构”。在 `torch.random.fork_rng(devices=[])` 内构建模型；`load_checkpoint(..., restore_rng=False)` 严格载入权重。
3. 用 checkpoint normalization 标准化，不重新 fit；参数形状为 4/4/2/2，scale 有限且 >0。容差从 checkpoint `run_config.tolerances` 读取且须 >0；目标通过率与停止 ruler 同样读保存值。
4. 每批默认 128 行，与当前训练评估一致；每批前检查 `threading.Event`。可逐批调用 `metrics.predict_full_split` 后拼接；保持训练侧误差计算精度和转换路径。
5. 复用 `calculate_errors/calculate_pass_masks/group_metrics/summarize_metrics`。不要直接依赖 `summarize_predictions` 的默认 breakaway 参数；它目前使用导入时配置。显式用 `fixed_parameters.mu_s * mass_kg * gravity_m_s2` 与本次 `config.breakaway_band_N` 调用 `make_breakaway_selection_mask`。
6. 比较保存摘要与重算摘要：count 和通过/失败数严格一致；连续指标初始门限 `rtol=1e-6, atol=1e-8`。门限仅用于核对浮点指标，绝不用于放宽逐样本通过判定。计数缺失时由保存通过率与 count 验证整数一致性，不编造精确失败类型。
7. 核对失败时保留原始摘要，标红显示字段、保存值、重算值及差值；重算详情标记“不一致”，禁用导出，不能当作已验证结果。
8. 若所选 checkpoint 的 CSV 评估缺失，允许使用 checkpoint 内保存摘要；两种摘要同时存在必须互相核对。缺失项显示“未记录”。

### 6. UI 与后台状态

- 初始窗口 1440×900，最小 960×640；宽度 <1200 时总结/图表区改为上下排列。中文文本采用系统字体，长路径可选择复制。
- 控件 objectName 固定：`runPathEdit/labelsPathEdit/refreshButton/statusLabel/checkpointCombo/splitCombo/toleranceCombo/metricCombo/trainPlot/validationPlot/groupTable/failureTable/exportButton`。
- 默认 checkpoint 为 selected，否则 last；默认 split=validation；默认容差=本次 stopping_ruler；默认指标=standardized_mse；默认显示整体及三个运动组，breakaway 可选。
- 状态：EMPTY → LOADING → READY；推理为 INFERENCING；错误为 ERROR；关闭为 CLOSING。目录切换立即清空旧诊断并禁用导出。
- CSV 解析、哈希与推理均在 worker；主线程只更新控件。一个活动任务、一个最新待处理任务；替换待处理项时取消旧任务。
- 每次打开/刷新递增 run generation；每次诊断选择递增 request ID；result/error/progress 信号均带二者。主线程丢弃不匹配信号。
- 缓存键包含规范化实验路径、generation、checkpoint 文件身份、split、数据哈希；LRU 上限 4 项且总数组字节数不超过 128 MiB。缓存预测和误差，不缓存全部模型；刷新清空缓存。
- 窗口关闭时设置取消事件，暂缓 closeEvent；收到 worker finished 后清理 QObject/QThread 并真正关闭。不得在 UI 线程长时间 wait，不调用 terminate。
- 表格 DisplayRole 负责格式化，UserRole 返回原始数值作为 sortRole；筛选条件取交集；row ID 为不区分大小写的字面子串搜索；导出遵循当前筛选与排序。
- CSV 使用 UTF-8 BOM、原始精度数值、完整列名；0 行时仍导出表头。图表使用 `connect='finite'` 保留空组缺口；X 轴联动，非等距 update 悬停按最近真实点查找。

### 7. 实施顺序与完成门槛

| 阶段 | 产出 | 必须通过 |
| --- | --- | --- |
| P0 环境 | requirements、入口、空窗口 | pip check；Qt 版本打印一致；中文标题与窗口正常打开/关闭 |
| P1 数据 | contracts、loader、训练适配层 | 正常/中止/缺文件/坏哈希/未知 schema fixture；不读取 test.csv；快照与外部标签哈希区分正确 |
| P2 概览/曲线 | 总结、双图、控制器 | update=0 保留；非等距点悬停正确；训练/验证同 Y 范围；空组缺口；仅真实 LR 下降有标记 |
| P3 诊断 | checkpoint 推理、表格、筛选导出 | 严格小于容差边界；三种失败类型；数值排序；过滤后全部行导出；摘要核对失败禁用导出 |
| P4 异步 | worker、取消、缓存 | A→B 快切后 A 的迟到结果/错误不覆盖 B；刷新使缓存失效；推理中关闭无遗留线程 |
| P5 交付 | README、锁文件、验收记录 | 全测试通过；实际 Windows 100%/150%/200% DPI、最小窗口、中文与空格路径人工或桌面验证 |

- 测试 fixture 必须含 update 0/10/100、空运动组、临界容差相等值、breakaway 重叠、无 selected、中止运行、损坏 checkpoint。临时目录保存，明确标记 demo。
- 在测试中监测打开的文件路径，确保没有读取 test.csv；对测试运行目录和标签的内容哈希做前后比较，确保查看操作无写入。
- 性能验收数据规模：train 8000、validation 1000、101 个评估点；记录机器 CPU、首次加载/推理时间。窗口在后台任务期间仍可拖动/取消；不得为追求响应跳过完整评估点。
- 当前没有真实 run；先完成 fixture 验收。真实端到端验收使用以后提供的已停止实验，或显式独立执行训练脚本生成短程运行；查看器本身不启动训练。未完成真实 run 验收必须在交付记录注明，不写“已全面验证”。
- README 必须包含：安装/锁文件重建、启动、路径选择、真实/演示标识、导出、常见错误、实测环境与尚未验证项。

## 总体职责

使用 PySide6 + PyQtGraph 构建本地只读查看器，读取已结束或已中止的 transmodel 训练产物，解释运行结果、展示完整学习过程，并定位各 checkpoint 的运动组及逐样本失败。训练进程独立运行；查看器不修改权重、模型排名或原始记录，不读取保留测试集。第一版不提供实时监视、恢复训练、多实验比较、模型动画或打包。

顶部提供训练目录、标签目录、刷新、当前路径和真实/演示标志；中部用可拖动 `QSplitter` 放置总结与训练/验证双图；底部放置 checkpoint、数据集、容差选择器、运动组汇总和失败表。小窗口允许上下排列。

## 第一部分：运行概览与可信数据

### 功能 1：加载、校验与状态展示

- **职责：** 快速展示实验状态；只有数据身份通过校验才生成逐样本详情。
- **选用模块：** `app.py`（入口）、`viewer/report_data.py`（解析与校验）、`viewer/report_qt.py`（状态界面）；经 `training_adapter.py` 复用现有 `data.py`、`checkpoints.py`。
- **技术路线：** 按本地数据契约读取 `run.json`、历史 CSV、termination 和 checkpoint index；校验 schema、列、维度与有限值，报告具体文件及错误。优先读取实验内数据快照，缺失时使用指定或默认训练标签目录；分别按 snapshot_sha256/sha256、行数和稳定 row ID 校验。不匹配时禁用对应失败详情。未结束实验标为“未知/未完成”。

### 功能 2：确定性文字总结

- **职责：** 用已保存的事实解释运行、停止和模型选择，不推断失败原因。
- **选用模块：** `report_data.py` 生成结构化摘要，`report_qt.py` 显示。
- **技术路线：** 展示完成 update、记录 epoch、曝光次数、耗时、停止原因及已记录停止标志；区分 `selected` 与最后有效 checkpoint。读取 `run.json.config`（checkpoint 中对应 `run_config`）的容差和目标通过率，列出选中模型的训练/验证通过率、双集合达标状态、验证失败类型及各组失败率；比较 selected/last 的验证 MSE 与通过率。未推理且记录缺少失败分类时显示“待计算”。按记录翻译 `TARGET_REACHED`、`NO_PROGRESS`、`MAX_UPDATES`、`NUMERICAL_FAILURE`；中断或一般错误显示原始类型与信息。没有 selected 时标注“无正式选中模型”。

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
