# Qt 训练结果查看器：顶层设计与实现交接

## 1. 目标与边界

使用 PySide6 + PyQtGraph 构建独立本地桌面查看器，替代 HTML 报告。读取现有训练产物，显示运行总结、完整学习曲线、每个 checkpoint 的运动组通过率及失败样本输入。无需浏览器、网络或服务器。

训练进程与查看器独立。关闭窗口不影响训练。查看器不更新权重、不改变 checkpoint 排名、不选择新模型、不读取保留测试集、不修改原始实验记录。第一版查看已结束或已中止的实验；实时训练监视、恢复训练按钮、多实验比较、模型动画和 exe 打包不在第一版范围。

## 2. 已确认的页面布局

顶部：打开训练目录、选择标签目录、刷新、当前实验路径，以及明确的真实/演示标志。

中部用 QSplitter 左右并排：左侧可读总结和停止原因，右侧完整学习曲线。分隔线可拖动；训练集与验证集曲线在右侧并排，保持一致坐标范围。小窗口可以改上下排列。

底部：checkpoint、训练/验证数据集、容差标准选择器；运动组通过率表；失败样本过滤器和大表格。默认 checkpoint 为 index.json 的 selected；没有 selected 时默认最后有效 checkpoint，并明确标注“无正式选中模型”。默认显示验证集及保存的停止容差标准。

## 3. 曲线

- X 轴为 optimizer update。显示全部已保存评估点，不平滑、不截去早期数据。
- Y 轴通过下拉或按钮切换：总标准化 MSE，位移/速度各自标准化 MSE，两输出对总损失的贡献，位移/速度 MAE、P95、最大绝对误差，以及 coarse/intermediate/fine 联合通过率。
- 同一指标的训练/验证图共享 Y 轴范围。通过率始终 0–100%；其他指标显示单位，不能把标准化 MSE 标为米或米/秒。
- 支持整体、三个互斥运动组、临界力切片的显示/隐藏；固定配色，图例显示样本数。临界力切片与运动组重叠，不参与组计数求和。
- 零预测和恒速预测基线读取 baselines.csv，仅在已有对应数据的训练图显示，不臆造验证基线。
- 标记 selected checkpoint、最后评估位置、实际学习率下降位置。支持平移、缩放、重置视图和点悬停显示 update、指标值、组名与样本数。
- 空组指标显示无定义，曲线留缺口；不得用零替代。默认线性坐标。

## 4. 每个 checkpoint 的组表现

切换 checkpoint 后，分别显示整体、always_resting、moved_then_stopped、moving_at_observation，以及单独标明的 breakaway 切片。

列：组名称、样本总数、通过数、失败数、当前容差下通过率。通过率必须是位移与速度同时满足严格小于容差的联合通过率，不能取两输出通过率的平均。空组显示 count=0、通过率无定义。

快速显示保存的评估摘要；推理结果加载后应核对一致性，若不一致，显示错误而不是悄悄覆盖。

## 5. 失败表

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

## 6. 可读总结

使用确定性模板与真实统计量生成，不调用语言模型，不猜测因果。

1. 运行状态、停止原因、完成 update、epoch、样本曝光次数、耗时。
2. 最终选中 checkpoint 和最后有效 checkpoint，解释二者可能不同。
3. 保存的停止标准、位移/速度容差、目标通过率、选中模型的训练/验证通过率，以及双集合目标是否达到。
4. 验证失败总数、仅位移/仅速度/两者失败数，各组失败率与样本量。
5. selected 与 last 的验证 MSE/通过率差异；不存在 selected 时不声称模型已选定。
6. 建议检查方向仅依据现象，例如“该组失败率最高，可优先检查输入分布”，不得断言网络容量不足或自动建议扩大模型。

停止原因翻译：TARGET_REACHED 双集合达标；NO_PROGRESS 达到无显著改善耐心；MAX_UPDATES 达预算上限；NUMERICAL_FAILURE 数值异常。保留所有停止标志，耐心和预算可同时达到。中断或一般错误显示实际错误类型/信息。未结束实验只能标为状态未知/未完成，不能推断成功。

## 7. 输入契约与现有模块

| 文件 | 用途 |
| --- | --- |
| run.json | 本次配置、normalization、data_identity、软件信息、purpose |
| evaluations.csv | 每次评估的 split/scope 指标，曲线和各组摘要 |
| evaluation_events.csv | update、epoch、曝光、耗时、学习率前后值、停止标志 |
| batches.csv | 每批更新前 loss、实际批大小、使用的学习率 |
| baselines.csv | 训练集两种基线 |
| termination.json | 完成/中止状态、原因、selected 与最后有效模型 |
| checkpoints/index.json | snapshots、best_mse、best_pass_rate、last、selected |
| checkpoints/update_XXXXXX.pt | 权重、normalization、保存的评估结果、运行配置及继续训练状态 |

依赖 data.py 的加载与标准化函数、metrics.py 的 evaluate/误差/分组计算、checkpoints.py 的 load_checkpoint、neural_network.py 的 transmodel。实现前读取实际接口，不凭本文臆造参数。读取本次 run_config 而非当前 config.py 中可能已变化的容差。当前确认 cooldown=0。

标签默认位于查看器脚本旁的 labels/，与进程工作目录无关。支持明确选择其他标签目录。加载前按 run.json 的 SHA-256 校验 train.csv 和 validation.csv，校验行数和 row ID 对齐；不匹配则禁止生成失败详情，并给出具体错误。不能重新拟合 scaler：每个 checkpoint 使用其保存的 normalization。当前固定质量与摩擦参数见标签协议；未来如支持其他物理参数，必须将其纳入运行元数据再泛化。

## 8. 推荐代码边界

- report.py：Qt 程序入口，解析 run_directory 和 --labels-directory；不在 import 时打开窗口。
- report_data.py：读取/校验实验产物、构造曲线数据、单 checkpoint 推理、失败统计和总结数据。尽量不依赖 Qt，便于单元测试。
- report_qt.py：主窗口、布局、交互事件和状态显示。
- report_table.py：Qt 表格模型、类型正确的排序过滤、CSV 导出。
- 独立后台 worker：执行标签校验与 checkpoint 推理；UI 仅接收结果/进度/错误。

不必为了文件数量强制拆分；避免把训练实现复制到查看器。原 Matplotlib 静态曲线可作为可选导出能力保留，HTML 模板和网页脚本不再继续维护。

## 9. 性能、线程与错误处理

打开实验先读取小型 JSON/CSV，快速展示曲线与运行状态。仅在选择 checkpoint 时加载模型并推理；一次预测服务于三套容差切换。缓存以实验路径、checkpoint 标识、数据哈希为键，采用有限容量，避免所有 checkpoint 的逐样本数据常驻内存。

QThread/worker 负责耗时计算，主线程才可操作 Qt 控件。快速切换时用请求编号丢弃过期结果，避免旧 checkpoint 覆盖新选择。忙碌状态明确显示，重复请求可合并。关闭窗口时请求取消并安全退出工作线程，不强杀线程。CPU 推理使用 eval 和 inference_mode，不扰动训练随机状态。

遇到文件缺失、损坏、不支持的 schema、维度不符、哈希不符和非有限数值，报告具体文件与原因。空实验显示空状态。第一版允许要求“训练已停止再打开”，但不能因读取中间状态崩溃。仅加载可信的本地产物，沿用现有 weights_only=True 加载方式。