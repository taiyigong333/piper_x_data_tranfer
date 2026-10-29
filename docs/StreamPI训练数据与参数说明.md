# StreamPI 训练数据与参数说明

## 数据输入

StreamPI 仓库的 LeRobot 训练配置使用 `observation.images.cam_high`、`observation.images.cam_left_wrist`、`observation.images.cam_right_wrist`、`observation.state` 和 `action`。本项目数据集采用 LeRobot v2.1，30 fps，图像分辨率 `224x224`。固定相机对应 `cam_high`，采集的腕部相机对应 `cam_left_wrist`，未采集的右腕视角为全黑占位图。任务文本写入 `meta/tasks.jsonl` 和 `meta/episodes.jsonl`，每帧通过 `task_index` 关联。

每个样本输入是三路图像、当前关节状态和语言指令 `grab the rotating red square`。StreamPI Agilex policy 使用 14 维 `[左臂6关节, 左夹爪, 右臂6关节, 右夹爪]` 绝对关节位置；本次单臂 Piper-X 数据写入左臂 7 维，右臂 7 维零填充。动作标签预测下一 30Hz 采样点的相同 14 维状态；末帧动作复制末状态，保持 episode 长度一致。关节角沿用采集器的弧度，夹爪值沿用采集器输出，训练前应确认 StreamPI policy 的 Piper-X 数据变换与此状态定义一致。

## 模型与训练参数

StreamPI 是在 OpenPI 的 PI0.5 / PI0 模型基础上进行时序建模的训练代码。仓库中的实际训练参数由 `StreamPI/src/openpi/training/config.py` 的训练配置、模型配置和命令行覆盖共同决定。应优先选用与 PI0.5 权重、Piper-X 14 维关节动作以及三路相机键匹配的配置；本数据转换仓库不复制或擅自锁定 StreamPI 的优化器、学习率、batch size、action horizon、训练步数和 temporal context 等参数，因为它们依模型版本及算力计划而变。

训练启动前至少确认：

- 数据配置映射三路图片键、state、action 和 task prompt。
- 模型动作维度至少为 14，Piper-X state/action 关节顺序与数据变换一致。
- 训练采样率设为 30Hz，历史帧间隔按 30Hz 时间轴解释。
- 按本数据集重新计算或配置 normalization statistics，不沿用其他机械臂的数据统计。
- 若训练变换将绝对动作转为 delta，推理端需应用匹配的反变换。

StreamPI 的流式模型会基于多时刻视觉与语言输入预测动作序列/动作块；具体预测 horizon 由所选模型配置的 `action_horizon` 决定，不是转换数据中的固定值。
