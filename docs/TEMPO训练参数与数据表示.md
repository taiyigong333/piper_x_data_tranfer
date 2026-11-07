# TEMPO 训练参数与数据表示

## 核对依据

- [论文 TEMPO: Learning Temporal Context for Dynamic Robot Manipulation](https://arxiv.org/html/2609.16864)，arXiv:2609.16864v1，2026-09-15。论文 §4.2、§6、附录 B。
- [官方项目](https://github.com/tempo-robot/TEMPO)，本地 `../TEMPO` commit `dd8342a1234e7166b856ffec2d7917b51cce05e4`。
- 关键代码：`src/openpi/training/config.py` 的 `LeRobotYamDataConfig`，
  `policies/yam_dual_policy.py` 的 `YamInputs` / `LoadActionHistory`，
  `transforms.py` 的 `DeltaActions` / `AbsoluteActions`，`training/data_loader.py`。
- LeRobot 版本按 TEMPO pyproject 锁定为 commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`；本项目直接生成其 v2.1 格式。

README 文首另有旧标题“Closing the Representational Gap for VLAs in Dynamic Settings”和 CoRL 2026 表述。
本文按 arXiv 实际标题引用；未将 README 中的会议声明当作独立核实的发表信息。

## 模型究竟预测什么

论文 TEMPO 的主要改变在输入端：TEMPO-MOT 提供视觉运动上下文，TEMPO-ACT 提供历史机器人命令摘要。
标准 TEMPO 使用原有 flow matching 损失预测未来**关节位置动作块**，不是预测 TCP 的 xyz/旋转，也不是关节速度、力矩。
TEMPO+UVT 是论文另列的组合变体；官方 `_tempo_ut_*` 使用 64 维 $u_t$ 潜变量，再由解码器还原动作块。
本项目配置采用标准 TEMPO 的直接动作块模式，`predict_ut=False`。

### 磁盘上保存的 7D

$$
s_t=[q^1_t,q^2_t,q^3_t,q^4_t,q^5_t,q^6_t,g_t],\qquad
a_t=s_{\min(t+1,T-1)}.
$$

| 字段 | 内容 | 单位/类型 |
| --- | --- | --- |
| observation.state | 当前 6 个绝对关节角＋当前绝对夹爪开口 | rad、m，float32[7] |
| action | 下一 30Hz 时刻的同类绝对状态，末帧保持 | rad、m，float32[7] |
| timestamp | episode 内从 0 开始的 $t/30$ | 秒，float32 |
| frame_index / episode_index / index / task_index | 帧、episode、全局行、任务索引 | int64 |

状态读取 `puppet/arm_single_position_raw` 和 `puppet/end_effector_single_position_raw`。
不使用 `end_effector_single_pose_*` 的 TCP 位姿，不把夹爪二值化，也不做没有依据的弧度/角度转换。
关节弧度与夹爪米单位已在采集驱动 `devices/piper_x.py` 核对。

**标签限制：采集文件只有测量反馈，没有下发的目标命令。** 下一状态是行为克隆的监督代理，
不是恢复出的真实命令。表示空间、维度和 TEMPO 数据变换一致，但不能保证与论文原始命令标签完全等价。
论文 ACT 历史用命令，本数据 ACT 历史会用这些代理 action；若需严格复现命令语义，必须补充带时间戳的目标关节/夹爪命令记录。
未来真机验证必须核对控制器跟踪延迟、夹爪单位和目标约束；此项目没有执行任何机器人动作。

### 训练时变为相对当前状态的关节动作

`LeRobotYamDataConfig(use_delta_joint_actions=True)` 对动作块的前 6 维减去**块起点的当前状态**，夹爪保持绝对值：

$$
\widetilde a_{t,k}^{1:6}=a_{t+k}^{1:6}-s_t^{1:6},\qquad
\widetilde a_{t,k}^{7}=a_{t+k}^{7},\quad k=0,\ldots,15.
$$

不是每一帧减前一帧，也不是 TCP 位移。推理输出通过 `AbsoluteActions` 加回当前关节角，得到绝对关节目标。
磁盘不要预先存 delta，否则会重复相减。ACT 从 SAM2 缓存读取的是未经 delta/归一化的原始 action。
π0.5 的模型内部仍保留预训练的 32 维 action_dim；数据层实际 7D，自动补零到 32，输出裁回 7。
不能直接把预训练模型维度改为 7；应设 data.action_dim=7、model.action_history_dim=7。

## 图像与缺失相机

| 采集字段 | LeRobot 字段 | 模型槽位 | mask |
| --- | --- | --- | --- |
| camera_front | observation.images.head | base_0_rgb | True |
| camera_wrist_right | observation.images.left_wrist | left_wrist_0_rgb | True |
| 无 | 不写假视频 | right_wrist_0_rgb，YamInputs 自动全零 | False |

单臂保持 7D，不构造虚假的双臂 14D。腕部相机改名不代表对关节轴做镜像或改变顺序。
训练 data.cameras 必须设为 `("head", "left_wrist")`，默认三相机配置会尝试读取不存在的右腕文件。

JPEG 解码后转 RGB，保持比例 resize，再居中黑色填充到 224×224；640×480 得到 224×168，上下各补 28 像素。
输出 MP4 为 H.264 / yuv420p / 30fps / CRF18，LeRobot 解码后得到 RGB 图像。
采集 HDF5 的 channel 标签写作 rgb，但实际 RealSense 驱动请求 bgr8 并直接调用 OpenCV imencode；
因此按标准 JPEG 解码处理，不机械使用这个旧标签交换红蓝通道。已抽查原图与视频中的红色物体。
本相机没有应用 YAM 数据专用的鱼眼矫正或裁剪，`undistort_crop=False`。

## 30Hz 时间对齐

1. 检查相机与两个 raw 状态序列的时间戳严格递增、维度正确、数值有限。
2. 取三者时间覆盖的交集，$t_n=t_0+n/30$；不在覆盖范围外外推。
3. 各关节与夹爪分别按自身真实时间戳作线性插值。
4. 两路图像共用一个 camera_rig 索引，按时间最近邻匹配；需要时复用图像，不生成虚构中间帧。
5. 相机误差或机器人采样间隙超过配置阈值则报错；记录所有匹配索引、重复/未使用图像数及误差。

不能仅把原始帧重新编号为 30fps：采集时间戳有抖动；也不能简单按状态“隔一帧取一帧”。
本次输出 44,398 帧，107 个重复图像位置，最大匹配误差 0.039274s；未遇到超 100ms 的机器人采样间隙。
数据 loader 按 meta.fps 构建未来动作时间偏移；ACT history_fps 与真机控制频率也都必须为 30。

## 随代码提供的训练配置

`piperx_to_tempo_vla.tempo_entry` 在运行时注册配置，不修改 TEMPO 工作区。

| 配置 | MOT | ACT | 步数 | warmup / decay | 保存间隔 |
| --- | --- | --- | --- | --- | --- |
| pi05_piperx | 无 | 无 | 10,000 | 1,000 / 10,000 | 2,000 |
| pi05_piperx_tempo_mot | 有 | 无 | 5,000 | 500 / 5,000 | 1,000 |
| pi05_piperx_tempo | 有 | 有 | 5,000 | 500 / 5,000 | 1,000 |

这些是本地官方任务配置的默认值，并非声称已针对本数据调参或完成论文复现。
共同参数：π0.5、batch_size=128（全局）、num_workers=4、seed=42、AdamW，
betas=(0.9,0.95)、eps=1e-8、weight_decay=1e-10、梯度裁剪 1.0；学习率从 warmup 升至 5e-5，余弦降至 5e-6。
动作块 horizon=16，在 30Hz 下覆盖约 0.533s；本项目的首个目标是下一帧状态，末端无后续目标时按保持处理。
数据末尾不足完整动作块时由 LeRobot 边界处理。

MOT 配置保留官方 `obs_history=3`、`obs_history_stride_s=0.333`（30fps 时取整为间隔10帧），
`temporal_attn_period=4`、SAM2 token_dim=256、num_tokens=64；缓存原 latent 为 `(T,256,8,8)`。
ACT 使用过去 10s 的 300 帧动作，不含当前帧，10 个桶、每桶30帧求均值；episode 起始前补零，整桶越界时 pad=True。

训练脚本的实时进度含 loss、lr、step，入口把汇总日志间隔设为10步。
官方配置虽写 `ema_decay=0.999`，当前 PyTorch train_loop 未使用 EMA，不能据此声称训练会启用 EMA。
当前官方脚本没有自动早停或独立在线任务评测；建议额外留出按 episode 的验证集、观察验证动作误差和真机任务成功率。
本次不擅自改变官方训练逻辑或数据划分，输出全体属于 train。

## 后续训练执行方法（本次未执行训练）

先按 `../TEMPO/README.md` 安装其独立 uv 训练环境、SAM2、预训练 PyTorch π0.5 权重，
并把官方 transformers_replace 补丁复制进**实际训练 Python 版本**对应的 transformers 目录。
转换环境 `.venv` 不包含完整训练依赖。

以下从 `TEMPO/` 执行；PIPERX_BASE_WEIGHTS 必须包含适用于当前 TEMPO 实现的 PyTorch 基座权重。

```bash
cd /home/cv/gcj/project/data_transfer/to_tempo/TEMPO
export HF_LEROBOT_HOME=/home/cv/gcj/project/data_transfer/to_tempo/after_transfer
export LEROBOT_VIDEO_BACKEND=pyav
export PYTHONPATH=/home/cv/gcj/project/data_transfer/to_tempo/code:$PWD/src
export TEMPO_DATA_ROOT=$PWD/caches
export TEMPO_CKPT_ROOT=$PWD/checkpoints
export PIPERX_REPO_ID=piperx_rotating_red_square
export PIPERX_BASE_WEIGHTS=$PWD/checkpoints/pi05_base_pytorch
# 另行设置实际 SAM2_REPO 和 SAM2_CKPT，参见官方 README

# 生成 SAM2 缓存；end 必须覆盖本次所有393条，官方默认只到60
uv run python tools/precompute_sam2_tokens.py \
  --root "$HF_LEROBOT_HOME/$PIPERX_REPO_ID" \
  --cache "$TEMPO_DATA_ROOT/sam2/$PIPERX_REPO_ID" --start 0 --end 393

# 必须重新计算关节增量空间的归一化统计，不能复用 YAM 的统计
JAX_PLATFORMS=cpu uv run python -m piperx_to_tempo_vla.tempo_entry norm \
  --config-name pi05_piperx_tempo

# GPU数量按实际机器修改；global batch须能被进程数整除
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  --module piperx_to_tempo_vla.tempo_entry train pi05_piperx_tempo \
  --exp_name=piperx_rotating_square --batch_size=128
```

norm 输出 `TEMPO/assets/pi05_piperx_tempo/piperx_rotating_red_square/norm_stats.json`。
baseline / MOT 对照实验应选择各自配置名分别生成统计与训练。
训练入口启动前检查数据目录、30Hz、7D 与两相机字段。
本项目从 π0.5 基座开始，不使用官方其他机器人任务的 checkpoint；可通过 PIPERX_BASE_WEIGHTS 指定匹配的 warm start。
输入数据一旦重转或改变语言/episode 索引，必须使用新缓存目录并重新计算 SAM2/归一化统计，避免旧缓存错配。

## 关于 TEMPO+UVT

官方 README 的 `_tempo_ut_*` 还需要 MVAE、encode_ut_windows 和 train_ut_decoder。
这不是转图像/写 Parquet 这一步产生的标签。本项目没有生成伪造 $u_t$ 或把 7D action 写成潜变量。
当前官方 `train_ut_decoder.py` 对 raw action 直接归一化，而标准 YAM 数据变换先减当前 state；
若直接套用默认 delta 配置，必须先处理 decoder 目标空间的一致性。
因此本项目只提供已核对目标空间的直接动作块配置；UVT 变体需要另行验证后再启用。

## 验证边界

已执行输入审计、端到端小样例测试、全量输出与来源验证。
未安装/运行完整 TEMPO 训练依赖、真实 LeRobotDataset loader、SAM2 提取、归一化计算、GPU 训练或真机推理。
静态接口核对不能代替训练启动验证，后续需在训练环境执行上述 norm 和最小训练任务。
