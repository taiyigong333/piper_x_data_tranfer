"""在 TEMPO 自身环境中注册 Piper-X 配置，再调用官方脚本。"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import runpy
import sys
from pathlib import Path


def verify_dataset_contract(root: Path) -> None:
    """训练前强制核对频率和模态，避免加载错误批次或把 60Hz 当成 30Hz。"""
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info["fps"] != 30:
        raise ValueError("Piper-X TEMPO 训练数据必须为 30Hz")
    for key in ("action", "observation.state"):
        if info["features"][key]["shape"] != [7]:
            raise ValueError(f"{key} 必须为单臂 7D")
    cameras = {k for k, v in info["features"].items() if v["dtype"] == "video"}
    if cameras != {"observation.images.head", "observation.images.left_wrist"}:
        raise ValueError("数据集必须包含 head 和 left_wrist 两路视频")


def register_configs() -> None:
    # 延迟导入：转换环境无须安装 GPU、JAX、torch 或整个训练依赖树。
    import openpi.training.config as config

    repo_id = os.environ.get("PIPERX_REPO_ID", "piperx_rotating_red_square")
    dataset_home = os.environ.get("HF_LEROBOT_HOME")
    if not dataset_home:
        raise ValueError("请先设置 HF_LEROBOT_HOME 为 after_transfer 的绝对路径")
    verify_dataset_contract(Path(dataset_home) / repo_id)
    weights = os.environ.get("PIPERX_BASE_WEIGHTS", config.pi05_base_ckpt())
    templates = {
        "pi05_piperx": "pi05_yam_ball_drop",
        "pi05_piperx_tempo_mot": "pi05_yam_tempo_mot_ball_catch_all",
        "pi05_piperx_tempo": "pi05_yam_tempo_ball_catch_all",
    }
    for name, template in templates.items():
        base = config.get_config(template)
        data = dataclasses.replace(
            base.data,
            repo_id=repo_id,
            action_dim=7,
            cameras=("head", "left_wrist"),
            default_prompt="grab the rotating red square",
            use_delta_joint_actions=True,
            sam2_cache_dir=config.sam2_cache(repo_id) if "tempo" in name else None,
            action_history_cache_dir=config.sam2_cache(repo_id) if name == "pi05_piperx_tempo" else None,
            action_history_fps=30,
            undistort_crop=False,
        )
        model = dataclasses.replace(base.model, predict_ut=False, action_history_dim=7)
        # 原 task 的 warm start 不适用于新机器人；从用户提供的 π0.5 基座初始化。
        custom = dataclasses.replace(
            base,
            name=name,
            model=model,
            data=data,
            pytorch_weight_path=weights,
            log_interval=10,
            policy_metadata={"control_hz": 30, "action_space": "joint_position", "gripper_unit": "m"},
        )
        config._CONFIGS_DICT[name] = custom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("norm", "train"))
    parser.add_argument("--tempo-root", type=Path, default=Path(__file__).resolve().parents[2] / "TEMPO")
    args, remaining = parser.parse_known_args()
    root = args.tempo_root.resolve()
    sys.path.insert(0, str(root / "src"))
    os.chdir(root)
    register_configs()
    script = root / "scripts" / ("compute_norm_stats.py" if args.mode == "norm" else "train_pytorch.py")
    sys.argv = [str(script), *remaining]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
