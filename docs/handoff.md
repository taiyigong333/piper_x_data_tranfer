# 项目交接

- 分支：`piperx_to_dynamic_vla`
- 转换入口：`code/piperx_to_dynamic_vla/convert.py`
- 输入：可传入单个 `batch_*`，也可传入包含多个 `batch_*` 的数据根目录。
- 输出：单 batch 写入指定目录；多 batch 写入指定目录下对应的 `batch_*` 子目录。
- 当前策略：仅 `forward`；状态读取 `puppet/*_align/data`，按相机时间轴输出 30Hz；图像 resize 后黑边填充至 384x384。
- 支持的轨迹名：新格式 `forward/trajectory.hdf5`，旧格式 `forward/*_trajectory.hdf5`。
- 采集配置副本：`config/rgb_auto_tuned.json`、`config/grab_rotating_red_square_into_blue_plate.yaml`

验证时应检查：输出 episode 数等于 forward HDF5 数；每个 parquet 行数等于两路视频帧数；`meta/info.json` 的 `fps` 为 30；图像尺寸为 384x384。
