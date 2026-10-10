"""将 Piper-X HDF5 正向轨迹转换为 DynamicVLA 使用的 LeRobot v2.1 数据集。"""

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
TARGET_SIZE = (384, 384)


def _scalar(value):
    """把 HDF5 字符串标量统一为 Python 类型，兼容 bytes 和 numpy 标量。"""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _scalar(value.item())
    return value.item() if isinstance(value, np.generic) else value


def _read_dataset(group: h5py.Group, name: str) -> np.ndarray:
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
    """保持宽高比缩放，并在四周补黑边，避免改变抓取目标几何形状。"""
    target_w, target_h = size
    height, width = image.shape[:2]
    scale = min(target_w / width, target_h / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    left, top = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def _state_from_hdf5(f: h5py.File, n: int) -> np.ndarray:
    puppet = f["puppet"]
    pose = _read_dataset(puppet, "end_effector_single_pose_align/data")[:n].astype(np.float32)
    gripper = _read_dataset(puppet, "end_effector_single_position_align/data")[:n].astype(np.float32)
    if pose.shape[1] not in (6, 7):
        raise ValueError(f"不支持的末端位姿维度: {pose.shape}")
    if pose.shape[1] == 7:
        # 采集配置是 xyz_rxryrz；若旧文件写入四元数，保留其七维兼容性。
        state = np.concatenate([pose, gripper[:, :1]], axis=1)
    else:
        state = np.concatenate([pose, gripper[:, :1]], axis=1)
    return state


def _is_publishable_episode(path: Path) -> bool:
    """只接受正式 forward 目录中的轨迹，排除采集器的 staging 文件。"""
    return path.is_file() and ".staging" not in path.parts and "forward" in path.parts


def _episode_files(input_root: Path) -> list[Path]:
    """发现单个输入目录中的新旧两种轨迹文件名。"""
    if input_root.is_file():
        if input_root.suffix not in {".h5", ".hdf5"}:
            raise ValueError(f"输入文件不是 HDF5: {input_root}")
        return [input_root]
    if not input_root.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {input_root}")

    patterns = ("*/forward/trajectory.hdf5", "*/forward/*_trajectory.hdf5")
    paths = {
        path
        for pattern in patterns
        for path in input_root.glob(pattern)
        if _is_publishable_episode(path)
    }
    return sorted(paths)


def discover_inputs(input_root: Path) -> list[tuple[str, Path, list[Path]]]:
    """将文件、单个 batch 或包含多个 batch 的数据根目录解析为转换任务。"""
    if input_root.is_file():
        # 向上回到 collection 目录，使 convert_dataset 能按普通 episode 目录处理。
        return [(input_root.stem, input_root.parent.parent, [input_root])]
    if not input_root.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {input_root}")

    batch_dirs = sorted(
        path for path in input_root.glob("batch_*") if path.is_dir()
    )
    if batch_dirs:
        tasks = []
        for batch_dir in batch_dirs:
            episodes = _episode_files(batch_dir)
            if episodes:
                tasks.append((batch_dir.name, batch_dir, episodes))
        if not tasks:
            raise FileNotFoundError(f"没有在任何 batch 中找到 forward HDF5: {input_root}")
        return tasks

    episodes = _episode_files(input_root)
    if not episodes:
        raise FileNotFoundError(f"没有找到 forward HDF5: {input_root}")
    return [(input_root.name, input_root, episodes)]


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
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(len(values))],
    }


def _convert_episode(source: Path, output: Path, episode_index: int, fps: int) -> dict:
    with h5py.File(source, "r") as f:
        camera = f["camera_observations"]
        timestamps = _read_dataset(camera, "timestamp").astype(np.float64)
        n = int(_scalar(f["metadata"]["trajectory_length"][()]))
        n = min(n, len(timestamps))
        state = _state_from_hdf5(f, n)
        # LeRobot 的 action 记录当前帧对应的控制目标；末帧重复自身以保持长度一致。
        action = np.vstack([state[1:], state[-1:]])
        cameras = ["camera_front", "camera_wrist_right"]
        frames = {name: [] for name in cameras}
        for name in cameras:
            encoded = camera["color_images"][name]
            for item in encoded[:n]:
                frames[name].append(resize_with_padding(_decode_image(item)))
        instruction = _scalar(f["metadata"]["language_instruction"][()])

    episode_dir = output / "data" / f"chunk-{episode_index // 1000:03d}"
    video_dir = output / "videos" / f"chunk-{episode_index // 1000:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    for name, values in frames.items():
        _write_video(values, video_dir / f"observation.images.{name}" / f"episode_{episode_index:06d}.mp4", fps)
    frame_index = np.arange(n, dtype=np.int64)
    table = pa.Table.from_pandas(pd.DataFrame({
        "action": list(action.astype(np.float32)),
        "observation.state": list(state.astype(np.float32)),
        # 绝对 Unix 时间不适合作为 episode 内时间轴，按首帧归零保留原始帧间隔。
        # 输出视频是固定帧率编码，元数据时间轴也固定为 30Hz，避免采集抖动造成错位。
        "timestamp": (np.arange(n, dtype=np.float32) / fps),
        "frame_index": frame_index,
        "episode_index": np.full(n, episode_index, dtype=np.int64),
        "index": frame_index + 0,  # 修正为全局索引由调用方更新
        "task_index": np.zeros(n, dtype=np.int64),
    }), preserve_index=False)
    pq.write_table(table, episode_dir / f"episode_{episode_index:06d}.parquet")
    return {"episode_index": episode_index, "length": n, "instruction": instruction, "state": state, "action": action}


def convert_dataset(
    input_root: Path,
    output_root: Path,
    fps: int = 30,
    overwrite: bool = False,
    episode_paths: list[Path] | None = None,
) -> dict:
    episodes = sorted(episode_paths) if episode_paths is not None else _episode_files(input_root)
    if not episodes:
        raise FileNotFoundError(f"没有找到正向轨迹: {input_root}/batch_*/**/forward/*_trajectory.hdf5")
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"输出目录非空，请使用 --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records, total_frames = [], 0
    for index, source in enumerate(episodes):
        LOG.info("[%d/%d] 转换 %s", index + 1, len(episodes), source)
        record = _convert_episode(source, output_root, index, fps)
        record["global_start"] = total_frames
        total_frames += record["length"]
        records.append(record)
    # 回填 parquet 的全局 index，DynamicVLA 的 LeRobot loader 依赖该字段。
    for record in records:
        path = output_root / "data" / f"chunk-{record['episode_index'] // 1000:03d}" / f"episode_{record['episode_index']:06d}.parquet"
        table = pq.read_table(path).to_pandas()
        table["index"] = np.arange(record["length"], dtype=np.int64) + record["global_start"]
        pq.write_table(pa.Table.from_pandas(table, preserve_index=False), path)
    features = {
        "action": {"dtype": "float32", "shape": [7], "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"]},
        "observation.state": {"dtype": "float32", "shape": [7], "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"]},
    }
    for name in ("camera_front", "camera_wrist_right"):
        features[f"observation.images.{name}"] = {"dtype": "video", "shape": [384, 384, 3], "names": ["height", "width", "channel"], "info": {"video.fps": fps, "video.codec": "mp4v", "video.height": 384, "video.width": 384, "video.channel": 3, "video.pix_fmt": "yuv420p"}}
    (output_root / "meta").mkdir()
    info = {"codebase_version": "v2.1", "robot_type": "piper_x", "fps": fps, "total_episodes": len(records), "total_frames": total_frames, "total_tasks": 1, "total_videos": len(records) * 2, "total_chunks": math.ceil(len(records) / 1000), "chunks_size": 1000, "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet", "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4", "features": features, "splits": {"train": f"0:{len(records)}"}}
    (output_root / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {"action": _stats(np.concatenate([r["action"] for r in records])), "observation.state": _stats(np.concatenate([r["state"] for r in records]))}
    (output_root / "meta" / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_root / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": records[0]["instruction"]}, ensure_ascii=False) + "\n")
    with (output_root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as episode_file, (output_root / "meta" / "episodes_stats.jsonl").open("w", encoding="utf-8") as stats_file:
        for r in records:
            ep_stats = {
                "action": _stats(r["action"]),
                "observation.state": _stats(r["state"]),
            }
            # DynamicVLA 写出的是 v2.1 的双清单格式，loader 分别读取 episode 和 episode stats。
            episode_file.write(json.dumps({"episode_index": r["episode_index"], "tasks": r["instruction"], "length": r["length"]}, ensure_ascii=False) + "\n")
            stats_file.write(json.dumps({"episode_index": r["episode_index"], "stats": ep_stats}, ensure_ascii=False) + "\n")
    return {"episodes": len(records), "frames": total_frames, "output": str(output_root), "source_files": [str(p) for p in episodes]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="单 batch 输出目录或多 batch 输出根目录")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")
    tasks = discover_inputs(args.input)
    input_path = args.input.resolve()
    multi_batch = args.input.is_dir() and any(
        batch_root.resolve() != input_path for _, batch_root, _ in tasks
    )
    results = []
    for name, batch_root, episodes in tasks:
        output = args.output / name if multi_batch else args.output
        results.append(convert_dataset(batch_root, output, args.fps, args.overwrite, episode_paths=episodes))
    print(json.dumps({"batches": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
