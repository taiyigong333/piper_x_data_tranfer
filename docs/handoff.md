# 项目交接

- 仓库目录：`/home/cv/gcj/project/data_transfer/to_pi/code`
- 分支：`piperx_to_openpi_vla`
- 远程：`git@github.com:taiyigong333/piper_x_data_tranfer.git`
- 转换入口：`piperx_to_openpi_vla/convert.py`
- 默认来源：`/home/cv/gcj/data_collect/data/rotating_red_square_blue_plate` 的 batch 2、3、4。
- 输入范围：只扫描正式 `forward/trajectory.hdf5`；不转换 reverse 或其他 batch。
- 输出：LeRobot v2.1；两路 RGB 图像 resize + 黑边到 `224x224`，末端 pose 加 gripper 组成 7 维 state/action。
- 时间对齐：优先使用 HDF5 中与相机帧同步的 `_align` 数据；缺少等长对齐数据时，对高频 raw 状态按相机时间戳插值。输出 fps 为 30。
- 语言指令：默认保留 HDF5 的 `metadata/language_instruction`；命令行传 `--language-instruction "..."` 可统一覆盖，结果写入 `meta/tasks.jsonl` 和 `meta/episodes.jsonl`。
- 采集参数副本位于 `config/`；命令和数据格式说明见 `docs/使用方法.md`。

## 当前验证边界

转换器的单元测试覆盖图像尺寸/黑边、时间插值及正向轨迹发现。执行全量转换前，建议先将一个 batch 转到临时输出目录，确认 OpenPI 环境中的 LeRobot loader 可以打开，然后再运行 2、3、4 全量转换。训练配置还需根据所用 pi0.5 模型及数据规范配置 key repack 和 normalization stats。
