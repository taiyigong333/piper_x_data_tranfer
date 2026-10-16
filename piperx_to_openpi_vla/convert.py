"""将 Piper-X 正向采集轨迹转换为 OpenPI 可读取的 LeRobot 数据集。"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LOG = logging.getLogger(__name__)
CAMERAS = ("camera_front", "camera_wrist_right")
TARGET_SIZE = (224, 224)
STATE_NAMES = ("x", "y", "z", "rx", "ry", "rz", "gripper")


def _scalar(value):
    """统一处理 HDF5 中的 bytes、numpy 标量和零维数组。"""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _scalar(value.item())
    return value.item() if isinstance(value, np.generic) else value


def _dataset(group: h5py.Group, name: str) -> np.ndarray:
    if name not in group:
        raise KeyError(f"缺少 HDF5 数据集: {group.name}/{name}")
    return np.asarray(group[name][()])


def _decode_image(encoded: np.ndarray) -> np.ndarray:
    data = np.frombuffer(np.asarray(encoded, dtype=np.uint8).tobytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解码相机 JPEG 图像")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_with_padding(image: np.ndarray, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """保持宽高比缩放，再用黑边补齐，避免改变目标几何形状。"""
    target_w, target_h = size
    height, width = image.shape[:2]
    scale = min(target_w / width, target_h / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    left, top = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def _interpolate_state(raw_times: np.ndarray, raw_pose: np.ndarray, raw_gripper: np.ndarray, times: np.ndarray) -> np.ndarray:
    """将高频机器人状态按相机时间戳线性插值到 30Hz。"""
    order = np.argsort(raw_times)
    source_times = np.asarray(raw_times, dtype=np.float64)[order]
    pose = np.asarray(raw_pose, dtype=np.float64)[order]
    gripper = np.asarray(raw_gripper, dtype=np.float64)[order].reshape(-1, 1)
    unique, indices = np.unique(source_times, return_index=True)
    pose, gripper = pose[indices], gripper[indices]
    target_times = np.asarray(times, dtype=np.float64)
    state = np.column_stack([
        np.interp(target_times, unique, pose[:, column]) for column in range(pose.shape[1])
    ])
    return np.column_stack([state, np.interp(target_times, unique, gripper[:, 0])]).astype(np.float32)


def _state_from_hdf5(file: h5py.File, camera_times: np.ndarray) -> np.ndarray:
    puppet = file["puppet"]
    pose_name = "end_effector_single_pose_align"
    grip_name = "end_effector_single_position_align"
    if pose_name in puppet and grip_name in puppet:
        pose = _dataset(puppet[pose_name], "data")
        gripper = _dataset(puppet[grip_name], "data")
        align_times = _dataset(puppet[pose_name], "timestamp")
        if len(pose) == len(camera_times) and len(align_times) == len(camera_times):
            return np.column_stack([pose, gripper.reshape(-1, 1)]).astype(np.float32)

    pose = _dataset(puppet["end_effector_single_pose_raw"], "data")
    gripper = _dataset(puppet["end_effector_single_position_raw"], "data")
    raw_times = _dataset(puppet["end_effector_single_pose_raw"], "timestamp")
    return _interpolate_state(raw_times, pose, gripper, camera_times)


def _episode_files(input_root: Path) -> list[Path]:
    if input_root.is_file():
        if input_root.suffix not in {".h5", ".hdf5"}:
            raise ValueError(f"输入文件不是 HDF5: {input_root}")
        return [input_root]
    if not input_root.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {input_root}")
    paths = {
        path
        for pattern in ("forward/trajectory.hdf5", "*/forward/trajectory.hdf5", "forward/*_trajectory.hdf5", "*/forward/*_trajectory.hdf5")
        for path in input_root.glob(pattern)
    }
    return sorted(path for path in paths if ".staging" not in path.parts)


def discover_inputs(input_root: Path, batches: list[int] | None = None) -> list[tuple[str, list[Path]]]:
    """发现指定 batch 的正向轨迹；目录结构不符合预期时直接报错。"""
    if input_root.is_file():
        return [(input_root.stem, [input_root])]
    batch_dirs = sorted(path for path in input_root.glob("batch_*") if path.is_dir())
    if batch_dirs:
        wanted = {f"batch_{number}" for number in batches} if batches else None
        tasks = [(path.name, _episode_files(path)) for path in batch_dirs if wanted is None or path.name in wanted]
        tasks = [(name, paths) for name, paths in tasks if paths]
        if not tasks:
            raise FileNotFoundError(f"没有找到指定 batch 的 forward HDF5: {input_root}")
        return tasks
    episodes = _episode_files(input_root)
    if not episodes:
        raise FileNotFoundError(f"没有找到 forward HDF5: {input_root}")
    return [(input_root.name, episodes)]


def _write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, TARGET_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(values, dtype=np.float32)
    return {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(), "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(), "count": [int(len(values))]}


def _resolve_instruction(source_instruction: str, override: str | None) -> str:
    """选择输出指令；未指定覆盖值时保留采集文件中的原始指令。"""
    if override is None:
        return source_instruction
    instruction = override.strip()
    if not instruction:
        raise ValueError("language instruction 覆盖值不能为空")
    return instruction


def _convert_episode(
    source: Path,
    output: Path,
    episode_index: int,
    fps: int,
    language_instruction: str | None = None,
) -> dict:
    with h5py.File(source, "r") as file:
        camera = file["camera_observations"]
        camera_times = _dataset(camera, "timestamp").astype(np.float64)
        n = min(len(camera_times), int(_scalar(file["metadata/trajectory_length"][()])))
        camera_times = camera_times[:n]
        state = _state_from_hdf5(file, camera_times)[:n]
        action = np.vstack([state[1:], state[-1:]])
        frames = {}
        for name in CAMERAS:
            encoded = camera["color_images"][name]
            frames[name] = [resize_with_padding(_decode_image(item)) for item in encoded[:n]]
        instruction = _resolve_instruction(
            _scalar(file["metadata/language_instruction"][()]), language_instruction
        )

    chunk = episode_index // 1000
    data_dir = output / "data" / f"chunk-{chunk:03d}"
    video_dir = output / "videos" / f"chunk-{chunk:03d}"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, values in frames.items():
        _write_video(values, video_dir / f"observation.images.{name}" / f"episode_{episode_index:06d}.mp4", fps)
    frame_index = np.arange(n, dtype=np.int64)
    table = pa.Table.from_pandas(pd.DataFrame({
        "action": list(action),
        "observation.state": list(state),
        "timestamp": np.arange(n, dtype=np.float32) / fps,
        "frame_index": frame_index,
        "episode_index": np.full(n, episode_index, dtype=np.int64),
        "index": frame_index,
        "task_index": np.zeros(n, dtype=np.int64),
    }), preserve_index=False)
    pq.write_table(table, data_dir / f"episode_{episode_index:06d}.parquet")
    return {"episode_index": episode_index, "length": n, "instruction": instruction, "state": state, "action": action}


def convert_dataset(
    input_root: Path,
    output_root: Path,
    fps: int = 30,
    batches: list[int] | None = None,
    overwrite: bool = False,
    language_instruction: str | None = None,
) -> dict:
    tasks = discover_inputs(input_root, batches)
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"输出目录非空，请使用 --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records, total_frames, episode_index = [], 0, 0
    for batch_name, episodes in tasks:
        LOG.info("处理 %s：%d 条正向轨迹", batch_name, len(episodes))
        for source in episodes:
            LOG.info("[%d] 转换 %s", episode_index, source)
            record = _convert_episode(
                source,
                output_root,
                episode_index,
                fps,
                language_instruction=language_instruction,
            )
            record["batch"] = batch_name
            record["source"] = str(source)
            record["global_start"] = total_frames
            total_frames += record["length"]
            records.append(record)
            episode_index += 1
    task_indices = {
        instruction: index
        for index, instruction in enumerate(sorted({record["instruction"] for record in records}))
    }
    for record in records:
        path = output_root / "data" / f"chunk-{record['episode_index'] // 1000:03d}" / f"episode_{record['episode_index']:06d}.parquet"
        table = pq.read_table(path).to_pandas()
        table["index"] = np.arange(record["length"], dtype=np.int64) + record["global_start"]
        table["task_index"] = task_indices[record["instruction"]]
        pq.write_table(pa.Table.from_pandas(table, preserve_index=False), path)
    features = {
        "action": {"dtype": "float32", "shape": [7], "names": [list(STATE_NAMES)]},
        "observation.state": {"dtype": "float32", "shape": [7], "names": [list(STATE_NAMES)]},
    }
    for name in CAMERAS:
        features[f"observation.images.{name}"] = {"dtype": "video", "shape": [224, 224, 3], "names": ["height", "width", "channel"], "info": {"video.fps": fps, "video.codec": "mp4v", "video.height": 224, "video.width": 224, "video.channel": 3, "video.pix_fmt": "yuv420p"}}
    meta = output_root / "meta"
    meta.mkdir()
    info = {"codebase_version": "v2.1", "robot_type": "piper_x", "fps": fps, "total_episodes": len(records), "total_frames": total_frames, "total_tasks": len(task_indices), "total_videos": len(records) * len(CAMERAS), "total_chunks": math.ceil(len(records) / 1000), "chunks_size": 1000, "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet", "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4", "features": features, "splits": {"train": f"0:{len(records)}"}}
    (meta / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    all_state = np.concatenate([record["state"] for record in records])
    all_action = np.concatenate([record["action"] for record in records])
    (meta / "stats.json").write_text(json.dumps({"action": _stats(all_action), "observation.state": _stats(all_state)}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (meta / "tasks.jsonl").open("w", encoding="utf-8") as file:
        for instruction, task_index in sorted(task_indices.items(), key=lambda item: item[1]):
            file.write(json.dumps({"task_index": task_index, "task": instruction}, ensure_ascii=False) + "\n")
    with (meta / "episodes.jsonl").open("w", encoding="utf-8") as episode_file, (meta / "episodes_stats.jsonl").open("w", encoding="utf-8") as stats_file:
        for record in records:
            episode_file.write(json.dumps({"episode_index": record["episode_index"], "tasks": record["instruction"], "task_index": task_indices[record["instruction"]], "length": record["length"]}, ensure_ascii=False) + "\n")
            stats_file.write(json.dumps({"episode_index": record["episode_index"], "stats": {"action": _stats(record["action"]), "observation.state": _stats(record["state"])}}, ensure_ascii=False) + "\n")
    return {"episodes": len(records), "frames": total_frames, "output": str(output_root), "batches": [name for name, _ in tasks]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 3, 4], help="只转换这些 batch，默认 2 3 4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--language-instruction",
        default=None,
        help="覆盖输出数据集中的语言指令；不指定时沿用每条 HDF5 的原始指令",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.fps != 30:
        LOG.warning("OpenPI 训练约定为 30Hz，当前输出 fps=%d", args.fps)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")
    print(
        json.dumps(
            convert_dataset(
                args.input,
                args.output,
                args.fps,
                args.batches,
                args.overwrite,
                language_instruction=args.language_instruction,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
