"""将 Piper-X 正向 HDF5 轨迹转换为 StreamPI 可读取的 LeRobot v2.1 数据集。"""

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
STREAMPI_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
TARGET_SIZE = (224, 224)
STATE_NAMES = (
    "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5", "left_joint_6", "left_gripper",
    "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6", "right_gripper",
)
FPS = 30


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
    """保持宽高比缩放并补黑边，以免拉伸操作影响空间判断。"""
    target_w, target_h = size
    height, width = image.shape[:2]
    scale = min(target_w / width, target_h / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    left, top = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def _interpolate_state(raw_times: np.ndarray, pose: np.ndarray, gripper: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """按相机采样时刻插值 60Hz 机器人原始状态到 30Hz。"""
    order = np.argsort(raw_times)
    times = np.asarray(raw_times, dtype=np.float64)[order]
    pose = np.asarray(pose, dtype=np.float64)[order]
    gripper = np.asarray(gripper, dtype=np.float64).reshape(-1)[order]
    times, indices = np.unique(times, return_index=True)
    pose, gripper = pose[indices], gripper[indices]
    target = np.asarray(target_times, dtype=np.float64)
    state = np.column_stack([np.interp(target, times, pose[:, i]) for i in range(pose.shape[1])])
    return np.column_stack([state, np.interp(target, times, gripper)]).astype(np.float32)


def _state_from_hdf5(file: h5py.File, camera_times: np.ndarray) -> np.ndarray:
    """读取 6 关节+夹爪，并映射到 StreamPI 的双臂 14 维状态。"""
    puppet = file["puppet"]
    joints_align = puppet["arm_single_position_align"]
    grip = puppet["end_effector_single_position_raw"]
    joints = puppet["arm_single_position_raw"]
    grip_align = puppet["end_effector_single_position_align"]
    if len(joints_align["data"]) == len(camera_times) and len(grip_align["data"]) == len(camera_times):
        left = np.column_stack([_dataset(joints_align, "data"), _dataset(grip_align, "data").reshape(-1)])
    else:
        left = _interpolate_state(
            _dataset(joints, "timestamp"), _dataset(joints, "data"), _dataset(grip, "data"), camera_times
        )
    # 论文与 AgilexInputs 使用双臂 14 维接口；本采集是单臂，右臂槽位以零填充。
    state = np.zeros((len(left), 14), dtype=np.float32)
    state[:, :7] = left.astype(np.float32)
    return state


def _episode_files(root: Path) -> list[Path]:
    if root.is_file():
        if root.suffix not in {".h5", ".hdf5"}:
            raise ValueError(f"输入文件不是 HDF5: {root}")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {root}")
    patterns = ("forward/trajectory.hdf5", "*/forward/trajectory.hdf5")
    return sorted({p for pattern in patterns for p in root.glob(pattern) if ".staging" not in p.parts})


def discover_inputs(root: Path, batches: list[int] | None) -> list[tuple[str, list[Path]]]:
    """只发现指定 batch 的正式 forward 轨迹。"""
    batch_dirs = sorted(p for p in root.glob("batch_*") if p.is_dir())
    if batch_dirs:
        selected = {f"batch_{n}" for n in batches} if batches else None
        result = [(p.name, _episode_files(p)) for p in batch_dirs if selected is None or p.name in selected]
        result = [(name, paths) for name, paths in result if paths]
    else:
        result = [(root.name, _episode_files(root))]
    if not result or not any(paths for _, paths in result):
        raise FileNotFoundError(f"没有找到指定批次的 forward HDF5: {root}")
    return result


def _write_video(frames: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, TARGET_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    return {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(), "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(), "count": [int(len(values))]}


def _convert_episode(source: Path, output: Path, index: int, instruction: str) -> dict:
    with h5py.File(source, "r") as file:
        camera = file["camera_observations"]
        camera_times = _dataset(camera, "timestamp").astype(np.float64)
        length = min(len(camera_times), int(_scalar(file["metadata/trajectory_length"][()])))
        camera_times = camera_times[:length]
        state = _state_from_hdf5(file, camera_times)
        action = np.vstack([state[1:], state[-1:]])
        images = {}
        for source_name, target_name in zip(CAMERAS, STREAMPI_CAMERAS[:2], strict=True):
            images[target_name] = [resize_with_padding(_decode_image(item)) for item in camera["color_images"][source_name][:length]]
        # StreamPI 的三路输入接口固定；缺失的右腕相机用黑图占位。
        images["cam_right_wrist"] = [np.zeros((*TARGET_SIZE[::-1], 3), dtype=np.uint8) for _ in range(length)]

    chunk = index // 1000
    data_dir = output / "data" / f"chunk-{chunk:03d}"
    video_dir = output / "videos" / f"chunk-{chunk:03d}"
    data_dir.mkdir(parents=True, exist_ok=True)
    for camera_name, frames in images.items():
        _write_video(frames, video_dir / f"observation.images.{camera_name}" / f"episode_{index:06d}.mp4")
    frame_index = np.arange(length, dtype=np.int64)
    table = pd.DataFrame({
        "action": list(action), "observation.state": list(state),
        "timestamp": frame_index.astype(np.float32) / FPS, "frame_index": frame_index,
        "episode_index": np.full(length, index, dtype=np.int64), "index": frame_index,
        "task_index": np.zeros(length, dtype=np.int64),
    })
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False), data_dir / f"episode_{index:06d}.parquet")
    return {"episode_index": index, "length": length, "instruction": instruction, "state": state, "action": action}


def convert_dataset(input_roots: list[Path], output_root: Path, batches: list[int], language: str, overwrite: bool = False) -> dict:
    tasks = []
    for input_root in input_roots:
        tasks.extend(discover_inputs(input_root, batches))
    if not language.strip():
        raise ValueError("language 指令不能为空")
    paths = [path for _, episodes in tasks for path in episodes]
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"输出目录非空，请使用 --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, source in enumerate(paths):
        LOG.info("[%d/%d] 转换 %s", index + 1, len(paths), source)
        records.append(_convert_episode(source, output_root, index, language.strip()))
    global_start = 0
    for record in records:
        record["global_start"] = global_start
        global_start += record["length"]
        path = output_root / "data" / f"chunk-{record['episode_index'] // 1000:03d}" / f"episode_{record['episode_index']:06d}.parquet"
        frame = pq.read_table(path).to_pandas()
        frame["index"] = np.arange(record["length"], dtype=np.int64) + record["global_start"]
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    meta = output_root / "meta"
    meta.mkdir()
    features = {
        "action": {"dtype": "float32", "shape": [14], "names": [list(STATE_NAMES)]},
        "observation.state": {"dtype": "float32", "shape": [14], "names": [list(STATE_NAMES)]},
    }
    for name in STREAMPI_CAMERAS:
        features[f"observation.images.{name}"] = {"dtype": "video", "shape": [224, 224, 3], "names": ["height", "width", "channel"], "info": {"video.fps": FPS, "video.codec": "mp4v", "video.height": 224, "video.width": 224, "video.channel": 3, "video.pix_fmt": "yuv420p"}}
    total_frames = sum(record["length"] for record in records)
    info = {"codebase_version": "v2.1", "robot_type": "piper_x", "fps": FPS, "total_episodes": len(records), "total_frames": total_frames, "total_tasks": 1, "total_videos": len(records) * len(STREAMPI_CAMERAS), "total_chunks": math.ceil(len(records) / 1000), "chunks_size": 1000, "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet", "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4", "features": features, "splits": {"train": f"0:{len(records)}"}}
    (meta / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (meta / "stats.json").write_text(json.dumps({"action": _stats(np.concatenate([r["action"] for r in records])), "observation.state": _stats(np.concatenate([r["state"] for r in records]))}, ensure_ascii=False, indent=2), encoding="utf-8")
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": language.strip()}, ensure_ascii=False) + "\n", encoding="utf-8")
    with (meta / "episodes.jsonl").open("w", encoding="utf-8") as episode_file, (meta / "episodes_stats.jsonl").open("w", encoding="utf-8") as stats_file:
        for record in records:
            episode_file.write(json.dumps({"episode_index": record["episode_index"], "tasks": language.strip(), "task_index": 0, "length": record["length"]}, ensure_ascii=False) + "\n")
            stats_file.write(json.dumps({"episode_index": record["episode_index"], "stats": {"action": _stats(record["action"]), "observation.state": _stats(record["state"])}}, ensure_ascii=False) + "\n")
    return {"episodes": len(records), "frames": total_frames, "batches": [name for name, _ in tasks], "output": str(output_root)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--language", default="grab the rotating red square")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(convert_dataset(args.input, args.output, args.batches, args.language, args.overwrite), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
