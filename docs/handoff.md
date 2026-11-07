# 项目交接

## 目标与已交付结果

仅转换 blue_plate 的 batch_2/3/4 和 rotating_red_square 的 batch_5 下 forward 数据。
代码、uv 环境、Git 都位于 `/home/cv/gcj/project/data_transfer/to_tempo/code`。
分支 `piperx_to_tempo_vla`，远程 `git@github.com:taiyigong333/piper_x_data_tranfer.git` 已配置并成功 ls-remote。
未推送；本地提交管理代码，不提交原始或转换数据、.venv、日志。

输出：`/home/cv/gcj/project/data_transfer/to_tempo/after_transfer/piperx_rotating_red_square`。

| 来源 | episodes | frames | 输出 episode 索引 |
| --- | --- | --- | --- |
| rotating_red_square_blue_plate/batch_2 | 104 | 10,251 | 0–103 |
| rotating_red_square_blue_plate/batch_3 | 87 | 7,700 | 104–190 |
| rotating_red_square_blue_plate/batch_4 | 98 | 12,334 | 191–288 |
| rotating_red_square/batch_5 | 104 | 14,113 | 289–392 |
| 合计 | 393 | 44,398 | 0–392 |

合并 LeRobot v2.1、786 个视频、30Hz、224×224、单臂7D，语言统一为 `grab the rotating red square`。
没有写采集源、没有训练模型、没有修改 TEMPO 源码。

## 关键决策与限制

- 本地 TEMPO commit：`dd8342a1234e7166b856ffec2d7917b51cce05e4`；论文 arXiv:2609.16864v1 已在线核对。
- 6 个绝对关节弧度＋绝对夹爪开口米；训练时关节转相对块起点状态的 delta，夹爪绝对。
- **源数据没有 command，只能以 next observed state 作为 action 代理**；已向用户提示该差异。
  没有收到额外命令记录位置，按推荐代理方案完成转换；不能声称完全恢复论文命令标签。
- head ← camera_front；left_wrist ← camera_wrist_right；right_wrist 缺失，由 YamInputs mask=False。
- 真实时间戳取交集建立严格30Hz网格，raw 状态插值；不直接采用采集时 latest-state 的 align 字段。
- 图像最近邻匹配，107 个重复图像位置，最大误差约39.274ms；全部低于100ms阈值。
- 部分内部 batch_tag 与目录不符，以用户指定目录筛选，原信息保存在来源清单。
- 采集配置标签写 rgb，但驱动请求 bgr8 并用 OpenCV JPEG 编码；标准 JPEG 解码后转 RGB，已视觉抽查。
- 直接动作块训练 `predict_ut=False`；没有生成 SAM2、MVAE、UVT 缓存。
- 官方 UVT decoder 的 raw 动作归一化与默认 delta 训练空间存在一致性问题，未擅自改官方源码。

## 代码入口与依赖链

- `config/convert.json`：明确根路径/batch 白名单、输出、language、30Hz、224×224、阈值和并行数。
- `piperx_to_tempo_vla/convert.py`：发现 → 审计 → 对齐 → 并行视频/Parquet → 元数据 → 发布目录。
- `piperx_to_tempo_vla/validate.py`：回读所有来源与所有输出，写 validation.json。
- `piperx_to_tempo_vla/tempo_entry.py`：在 TEMPO 自身环境运行，动态注册 Piper-X baseline / MOT / MOT+ACT 配置并调用官方 norm/train。
- `tests/test_conversion.py`：时间插值、padding、batch/forward 筛选、坏时间戳、完整导出与视频颜色、动作、30Hz训练入口约束。
- `config/` 三份采集配置原样副本；其中早期 blue_plate 配置由 HDF5 source_config 追溯发现。
- `README.md`、`docs/使用方法.md`、`docs/TEMPO训练参数与数据表示.md`：运行说明与模型输入输出语义。

## 实际验证

- uv Python3.11 环境已创建于 code/.venv，依赖锁定在 uv.lock。
- 输入393条全量时间审计通过；5个pytest测试通过（包含完整合成数据导出和回读）。
- 全量44,398行状态/action、来源SHA256、指定batch、语言与索引通过。
- 全部786视频逐帧解码，帧数/尺寸/FPS/PTS通过。
- 源数据转换前后哈希一致；抽看两路图像与输出视频，比例、黑边及红色通道正确。
- 结果详情：输出 `meta/validation.json`、`meta/source_episodes.jsonl`；本机日志 `code/logs/conversion.log` 和 `validation.log`（不进 Git）。
- 未运行真实 LeRobotDataset loader、完整 TEMPO import、SAM2、归一化计算、训练或真机推理。

## 复跑与后续

```bash
cd /home/cv/gcj/project/data_transfer/to_tempo/code
UV_CACHE_DIR=/tmp/tempo-uv-cache uv run piperx-to-tempo --audit-only
UV_CACHE_DIR=/tmp/tempo-uv-cache uv run tempo-validate ../after_transfer/piperx_rotating_red_square
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/tempo-uv-cache uv run pytest -q
```

转换命令拒绝覆盖已有输出；新实验使用新的 output 路径。没有断点续跑。
训练前按训练说明安装 TEMPO 自身依赖、官方 transformers 补丁、π0.5 PyTorch 基座与 SAM2。
先跑 norm 最小验证，再做短训练确认两个相机、7D动作和mask；真机控制频率保持30Hz。
若用户要求原论文完全一致的命令监督，先取得目标命令记录及时间戳，再替换 action 构造；不要继续把反馈代理称为真命令。
