# Qt 训练结果查看器

只读查看 `model_training/first_training_protocol` 生成的单次实验。支持训练/验证历史曲线、checkpoint 逐样本诊断、筛选与 CSV 导出；不会读取 `test.csv` 或启动训练。

## 安装

在本目录运行 PowerShell：

```powershell
& '..\.venv\Scripts\python.exe' -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --only-binary=:all: -r '..\requirements.txt' -r requirements-dev.txt
& '.\.venv\Scripts\python.exe' -m pip check
& '.\.venv\Scripts\python.exe' -m pip freeze | Set-Content -Encoding utf8 requirements-lock.txt
```

锁文件安装需指定 CPU wheel 索引：`python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt`。

## 启动

本目录：`& '.\.venv\Scripts\python.exe' app.py`。任意目录：`& 'D:\.physical_ai\ball_simulation_python\qt_demonstration_software\.venv\Scripts\python.exe' 'D:\.physical_ai\ball_simulation_python\qt_demonstration_software\app.py' --run '实验目录'`。可加 `--labels '标签目录'`；通常优先使用实验内 `datasets` 快照。选择含 `run.json` 的单次实验目录。导出按钮只在核对通过后可用，路径由用户选择。

真实实验标为“真实实验”。`--demo` 当前会明确报错；演示 fixture 尚未打包。

## 常见错误与验证状态

缺 `run.json`、版本不支持、路径越界或数据哈希不符会显示具体错误。快照存在但不一致时，该 split 的详情停用。读取时文件变化表示训练可能仍在写入；停止训练后刷新。

计划要求的 Windows GUI 运行、DPI（100%/150%/200%）、真实实验端到端、性能规模验收仍待完成。当前环境网络安装被自动审批系统配置错误阻断，无法冻结依赖或运行 Qt 测试。
