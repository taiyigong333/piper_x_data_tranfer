# 项目交接

## 当前状态

- 转换器位于 `piperx_to_streampi/convert.py`，输出 LeRobot v2.1。
- 数据仅从指定 batch 的正式 `forward/trajectory.hdf5` 读取。
- 视频、state、action 使用 30Hz；高频原始关节状态按相机时间戳插值。
- 图像保持比例缩放并黑边填充到 224x224。
- 两相机映射到 StreamPI 的 `cam_high` 与 `cam_left_wrist`，缺失的右腕视角填黑。
- 采集配置副本保存在 `config/`。

## 本地操作

依照 `docs/使用方法.md` 执行 `uv sync` 和转换。当前数据输出位置在仓库外层 `after_transfer/`，不纳入 Git。

## 关键兼容性注意

训练侧应使用 StreamPI Agilex policy 的 14 维关节空间数据变换；不要把本数据集按末端 pose 解读。缺失右腕图像目前是黑色占位，训练 loader 是否支持三视角固定输入需在真实训练配置中确认。训练优化器、学习率、batch size、action horizon 和 temporal context 由 StreamPI 配置决定，详见 `StreamPI训练数据与参数说明.md`。
