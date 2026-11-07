"""将指定批次的正向轨迹转换为 TEMPO 的 LeRobot v2.1 数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LOG = logging.getLogger(__name__)
CAMERAS = {"head": "camera_front", "left_wrist": "camera_wrist_right"}
NAMES = [f"joint_{i}" for i in range(1, 7)] + ["gripper_width"]
SERIES = ("arm_single_position_raw", "end_effector_single_position_raw")


def scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value.item() if isinstance(value, np.generic) else value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def source_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_config(path: Path, output: Path | None = None, language: str | None = None) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["output"] = str(output.resolve() if output else (path.resolve().parent.parent / cfg["output"]).resolve())
    for item in cfg["inputs"]:
        root = Path(item["root"])
        item["root"] = str((path.parent / root).resolve() if not root.is_absolute() else root.resolve())
    if language is not None:
        cfg["language"] = language
    if not cfg["language"].strip():
        raise ValueError("language 不得为空")
    cfg["language"] = cfg["language"].strip()
    if cfg["fps"] != 30 or cfg["image_size"] != [224, 224]:
        raise ValueError("本项目要求 30Hz、224×224，与已核对的 TEMPO 设置一致")
    if cfg["action_source"] != "next_observed_state":
        raise ValueError("当前采集数据只支持 next_observed_state 动作代理")
    if cfg["workers"] < 1 or min(cfg["max_image_error_s"], cfg["max_robot_gap_s"]) <= 0:
        raise ValueError("workers 和时间误差阈值必须大于零")
    return cfg


def discover_inputs(cfg: dict) -> list[Path]:
    """显式白名单避免意外纳入其他 batch、reverse 或临时轨迹。"""
    result = []
    for item in cfg["inputs"]:
        if not item["batches"]:
            raise ValueError("必须显式指定 batch 白名单")
        for number in item["batches"]:
            root = Path(item["root"]) / f"batch_{int(number)}"
            paths = sorted(set(root.glob("*/forward/trajectory.hdf5")) | set(root.glob("*/forward/*_trajectory.hdf5")))
            paths = [p.resolve() for p in paths if not any(part.startswith(".") for part in p.relative_to(root).parts)]
            if not paths:
                raise FileNotFoundError(f"指定 batch 没有 forward HDF5：{root}")
            result.extend(paths)
    if len(set(result)) != len(result):
        raise ValueError("输入重复，请检查 batches 配置")
    return result


def checked_times(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} 必须至少包含两个严格递增的有限时间戳")
    return values


def align_episode(file: h5py.File, cfg: dict) -> dict:
    """以真实秒建立均匀网格，禁止用重新编号掩盖原始时间抖动。"""
    camera_times = checked_times(file["camera_observations/timestamp"][:], "camera")
    n = len(camera_times)
    if int(file["metadata/trajectory_length"][()]) != n:
        raise ValueError("trajectory_length 与相机时间戳长度不一致")
    for source in CAMERAS.values():
        if len(file[f"camera_observations/color_images/{source}"]) != n:
            raise ValueError(f"{source} 图像数与时间戳数不一致")
    robot = []
    for name, width in zip(SERIES, (6, 1), strict=True):
        group = file[f"puppet/{name}"]
        ts = checked_times(group["timestamp"][:], name)
        values = np.asarray(group["data"][:], dtype=np.float64)
        if values.shape != (len(ts), width) or not np.isfinite(values).all():
            raise ValueError(f"{name} 数值形状错误或包含非有限值")
        if np.max(np.diff(ts)) > cfg["max_robot_gap_s"]:
            raise ValueError(f"{name} 存在超过阈值的机器人采样间隙")
        robot.append((ts, values))
    start = max(camera_times[0], *(ts[0] for ts, _ in robot))
    end = min(camera_times[-1], *(ts[-1] for ts, _ in robot))
    # 先减 Unix 时间原点，降低大数相减对插值和帧数计算的影响。
    length = math.floor((end - start) * cfg["fps"] + 1e-8) + 1
    if length < 2:
        raise ValueError("相机与机器人没有足够的共同时间范围")
    relative = np.arange(length, dtype=np.float64) / cfg["fps"]
    times = start + relative
    state = np.column_stack(
        [np.interp(relative, ts - start, values[:, i]) for ts, values in robot for i in range(values.shape[1])]
    ).astype(np.float32)
    # 使用最近的真实图像；同步复用同一相机组索引，不对像素做时间插值。
    camera_relative = camera_times - start
    right = np.searchsorted(camera_relative, relative).clip(0, n - 1)
    left = (right - 1).clip(0, n - 1)
    indices = np.where(abs(camera_relative[left] - relative) <= abs(camera_relative[right] - relative), left, right)
    errors = camera_relative[indices] - relative
    if np.max(abs(errors)) > cfg["max_image_error_s"]:
        raise ValueError(f"图像与网格误差超阈值：{np.max(abs(errors)):.6f}s")
    return {
        "state": state,
        "times": times,
        "indices": indices,
        "audit": {
            "source_camera_frames": n,
            "length": length,
            "start_unix_s": float(start),
            "end_unix_s": float(times[-1]),
            "source_duration_s": float(camera_times[-1] - camera_times[0]),
            "camera_max_abs_error_s": float(np.max(abs(errors))),
            "camera_mean_abs_error_s": float(np.mean(abs(errors))),
            "camera_repeated_frames": int(length - len(np.unique(indices))),
            "camera_unused_frames": int(n - len(np.unique(indices))),
            "camera_indices": indices.tolist(),
            "robot_raw_frames": [len(ts) for ts, _ in robot],
            "robot_max_gap_s": [float(np.max(np.diff(ts))) for ts, _ in robot],
        },
    }


def resize_with_padding(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """缩放尺寸取整与 TEMPO 一致，先等比缩放再在居中位置补黑边。"""
    target_h, target_w = size
    h, w = image.shape[:2]
    ratio = max(w / target_w, h / target_h)
    nh, nw = max(1, int(h / ratio)), max(1, int(w / ratio))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top, left = (target_h - nh) // 2, (target_w - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def numeric_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [len(values)],
    }


def write_video(encoded, indices: np.ndarray, path: Path, cfg: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    total, squares = np.zeros(3), np.zeros(3)
    low, high = np.ones(3), np.zeros(3)
    last_index, rgb = None, None
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=cfg["fps"])
        stream.width, stream.height = cfg["image_size"][1], cfg["image_size"][0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "fast", "threads": "1"}
        for index in indices:
            if index != last_index:
                bgr = cv2.imdecode(np.asarray(encoded[int(index)], dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"JPEG 解码失败：{index}")
                # 采集驱动请求 bgr8，并直接 imencode；JPEG 应按标准 BGR 解码再转 RGB。
                rgb = resize_with_padding(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), tuple(cfg["image_size"]))
                last_index = index
            mean, std = cv2.meanStdDev(rgb)
            mean, std = mean[:, 0] / 255, std[:, 0] / 255
            total += mean
            squares += std**2 + mean**2
            low = np.minimum(low, rgb.min(axis=(0, 1)) / 255)
            high = np.maximum(high, rgb.max(axis=(0, 1)) / 255)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    mean = total / len(indices)
    values = {"min": low, "max": high, "mean": mean, "std": np.sqrt(np.maximum(squares / len(indices) - mean**2, 0))}
    return {**{key: value.reshape(3, 1, 1).tolist() for key, value in values.items()}, "count": [len(indices)]}


def episode_paths(root: Path, index: int) -> tuple[Path, Path]:
    chunk = f"chunk-{index // 1000:03d}"
    return root / "data" / chunk / f"episode_{index:06d}.parquet", root / "videos" / chunk


def convert_episode(job: tuple) -> dict:
    source, root, ep, offset, cfg = job
    cv2.setNumThreads(1)  # 多进程并行时避免每个 JPEG 再创建一组 CPU 线程。
    data_path, video_root = episode_paths(root, ep)
    fingerprint = source_digest(source)
    with h5py.File(source, "r") as file:
        plan = align_episode(file, cfg)
        state, indices = plan["state"], plan["indices"]
        # 源文件没有 command：下一网格状态是显式标注的行为克隆代理，末帧保持。
        action = np.concatenate([state[1:], state[-1:]])
        n = len(state)
        columns = {
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 7)),
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 7)),
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / cfg["fps"]),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, ep, dtype=np.int64)),
            "index": pa.array(np.arange(n, dtype=np.int64) + offset),
            "task_index": pa.array(np.zeros(n, dtype=np.int64)),
        }
        stats = {key: numeric_stats(np.asarray(value.to_pylist())) for key, value in columns.items()}
        for dest, camera in CAMERAS.items():
            stats[f"observation.images.{dest}"] = write_video(
                file[f"camera_observations/color_images/{camera}"],
                indices,
                video_root / f"observation.images.{dest}" / f"episode_{ep:06d}.mp4",
                cfg,
            )
        original = {
            key: scalar(file[f"metadata/{key}"][()])
            for key in ("batch_tag", "language_instruction", "trajectory_length")
        }
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), data_path)
    if source_digest(source) != fingerprint:
        raise RuntimeError(f"转换期间源文件发生变化：{source}")
    return {
        "episode_index": ep,
        "length": n,
        "source": str(source),
        "batch": source.parent.parent.parent.name,
        "source_sha256": fingerprint,
        "original_metadata": original,
        "alignment": plan["audit"],
        "stats": stats,
    }


def aggregate_stats(records: list[dict]) -> dict:
    result = {}
    for key in records[0]["stats"]:
        stats = [r["stats"][key] for r in records]
        weights = np.array([s["count"][0] for s in stats], dtype=np.float64)
        means = np.array([s["mean"] for s in stats])
        mean = np.average(means, axis=0, weights=weights)
        var = np.average(np.array([s["std"] for s in stats]) ** 2 + (means - mean) ** 2, axis=0, weights=weights)
        result[key] = {
            "min": np.min([s["min"] for s in stats], axis=0).tolist(),
            "max": np.max([s["max"] for s in stats], axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "count": [int(weights.sum())],
        }
    return result


def dataset_info(records: list[dict], cfg: dict) -> dict:
    features = {key: {"dtype": "float32", "shape": [7], "names": NAMES} for key in ("action", "observation.state")}
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[key] = {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": [224, 224, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": cfg["fps"],
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.height": 224,
                "video.width": 224,
                "video.channels": 3,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return {
        "codebase_version": "v2.1",
        "robot_type": "piper_x",
        "fps": cfg["fps"],
        "total_episodes": len(records),
        "total_frames": sum(r["length"] for r in records),
        "total_tasks": 1,
        "total_videos": 2 * len(records),
        "total_chunks": math.ceil(len(records) / 1000),
        "chunks_size": 1000,
        "splits": {"train": f"0:{len(records)}"},
        "features": features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }


def audit_dataset(cfg: dict) -> list[dict]:
    result = []
    for source in discover_inputs(cfg):
        with h5py.File(source, "r") as file:
            plan = align_episode(file, cfg)
            result.append({"source": str(source), **plan["audit"]})
    return result


def convert_dataset(cfg: dict, plans: list[dict]) -> dict:
    output = Path(cfg["output"])
    staging = output.with_name(output.name + ".partial")
    if output.exists() or staging.exists():
        raise FileExistsError(f"输出或暂存目录已存在，不自动覆盖：{output} / {staging}")
    # 防止输出位置落入采集数据或 Git 仓库；转换只向独立输出目录写入。
    for item in cfg["inputs"]:
        if output.is_relative_to(Path(item["root"])):
            raise ValueError("输出不能位于采集数据目录内")
    staging.mkdir(parents=True)
    offsets = np.cumsum([0] + [p["length"] for p in plans[:-1]]).tolist()
    jobs = [
        (Path(p["source"]), staging, ep, offset, cfg) for ep, (p, offset) in enumerate(zip(plans, offsets, strict=True))
    ]
    records = []
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        for record in pool.map(convert_episode, jobs):
            records.append(record)
            LOG.info("已完成 %d/%d：%s，%d 帧", len(records), len(jobs), record["batch"], record["length"])
    meta = staging / "meta"
    meta.mkdir()
    write_json(meta / "info.json", dataset_info(records, cfg))
    write_json(meta / "stats.json", aggregate_stats(records))
    write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": cfg["language"]}])
    write_jsonl(
        meta / "episodes.jsonl",
        [{"episode_index": r["episode_index"], "tasks": [cfg["language"]], "length": r["length"]} for r in records],
    )
    write_jsonl(
        meta / "episodes_stats.jsonl", [{"episode_index": r["episode_index"], "stats": r["stats"]} for r in records]
    )
    write_jsonl(meta / "source_episodes.jsonl", [{k: v for k, v in r.items() if k != "stats"} for r in records])
    write_json(
        meta / "conversion.json",
        {
            "config": cfg,
            "action_semantics": "action[t] = state[min(t+1,T-1)]; observed feedback proxy, not commanded targets",
            "camera_mapping": CAMERAS,
            "right_wrist": "absent; YamInputs masks it False",
            "source_count": len(plans),
        },
    )
    # 元数据、视频、Parquet 全部成功后才发布目录，失败时保留 partial 便于定位。
    os.rename(staging, output)
    return {"episodes": len(records), "frames": sum(r["length"] for r in records), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/convert.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--language", help="覆盖全部输出轨迹的语言，不修改源数据")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, args.output, args.language)
    if args.workers is not None:
        if args.workers < 1:
            parser.error("workers 必须大于零")
        cfg["workers"] = args.workers
    plans = audit_dataset(cfg)
    summary = {
        "episodes": len(plans),
        "frames": sum(p["length"] for p in plans),
        "max_image_error_s": max(p["camera_max_abs_error_s"] for p in plans),
        "repeated_frames": sum(p["camera_repeated_frames"] for p in plans),
    }
    LOG.info("输入审计通过：%s", summary)
    print(json.dumps(summary if args.audit_only else convert_dataset(cfg, plans), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
