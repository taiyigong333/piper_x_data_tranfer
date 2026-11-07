# Piper-X → TEMPO

将指定采集批次的正向 HDF5 转成 TEMPO 使用的 LeRobot v2.1 数据集：
30Hz，224×224 等比缩放与黑边填充，单臂关节空间 7D，两路相机映射到 head / left_wrist。

```bash
uv sync --locked
uv run piperx-to-tempo --audit-only
uv run piperx-to-tempo
uv run tempo-validate ../after_transfer/piperx_rotating_red_square
```

本次数据已完成转换，上述转换命令再次执行会拒绝覆盖。新转换请指定新的 `--output`。

- [使用方法](docs/使用方法.md)：安装、配置、修改语言、运行与验证。
- [TEMPO 训练参数与数据表示](docs/TEMPO训练参数与数据表示.md)：预测目标、相机映射、30Hz、训练接入。
- [交接文档](docs/handoff.md)：实际结果、代码入口、验证范围与待办。

源数据没有记录下发的目标命令；本项目以**下一时刻观测关节/夹爪状态**作为动作代理。
这与 TEMPO 使用的关节空间兼容，但不等于论文采集的命令标签。
