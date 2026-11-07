"""全量核对 LeRobot 元数据、数值、来源及视频帧时间。"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .convert import CAMERAS, align_episode, discover_inputs, episode_paths, write_json


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_episode(job: tuple) -> dict:
    root, record, episode, cfg, offset = job
    ep, n = record["episode_index"], record["length"]
    path, video_root = episode_paths(root, ep)
    table = pq.read_table(path)
    require(table.num_rows == n == episode["length"], f"episode {ep} 长度不符")
    state = np.array(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.array(table["action"].to_pylist(), dtype=np.float32)
    require(state.shape == action.shape == (n, 7), f"episode {ep} state/action 形状错误")
    require(np.isfinite(state).all() and np.isfinite(action).all(), "包含非有限值")
    for key in ("observation.state", "action"):
        dtype = table.schema.field(key).type
        require(
            pa.types.is_fixed_size_list(dtype) and dtype.value_type == pa.float32() and dtype.list_size == 7,
            f"{key} dtype 错误",
        )
    np.testing.assert_array_equal(action, np.concatenate([state[1:], state[-1:]]))
    for key, expected in {
        "frame_index": np.arange(n),
        "index": np.arange(n) + offset,
        "task_index": np.zeros(n),
        "episode_index": np.full(n, ep),
    }.items():
        require(table.schema.field(key).type == pa.int64(), f"{key} 必须 int64")
        np.testing.assert_array_equal(np.asarray(table[key]), expected)
    require(table.schema.field("timestamp").type == pa.float32(), "timestamp 必须 float32")
    np.testing.assert_array_equal(np.asarray(table["timestamp"]), np.arange(n, dtype=np.float32) / 30)
    source = Path(record["source"])
    with source.open("rb") as stream:
        require(hashlib.file_digest(stream, "sha256").hexdigest() == record["source_sha256"], f"源数据已改变：{source}")
    with h5py.File(source, "r") as file:
        plan = align_episode(file, cfg)
        np.testing.assert_array_equal(state, plan["state"])
        require(record["alignment"] == plan["audit"], "源图像匹配或时间审计不一致")
    require(episode["tasks"] == [cfg["language"]], "语言指令不一致")
    for camera in CAMERAS:
        video = video_root / f"observation.images.{camera}" / f"episode_{ep:06d}.mp4"
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            require(stream.average_rate == 30, f"{video} FPS 错误")
            require(stream.codec_context.name == "h264", f"{video} 编码错误")
            count = 0
            for count, frame in enumerate(container.decode(video=0), start=1):
                require((frame.height, frame.width) == (224, 224), f"{video} 图像尺寸错误")
                require(abs(float(frame.pts * frame.time_base) - (count - 1) / 30) < 1e-6, f"{video} PTS 错误")
            require(count == n, f"{video} 帧数错误 {count} != {n}")
    return {"episode_index": ep, "frames": n, "videos": len(CAMERAS)}


def validate_dataset(root: Path, workers: int = 12) -> dict:
    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text())
    conversion = json.loads((meta / "conversion.json").read_text())
    cfg = conversion["config"]
    records = read_jsonl(meta / "source_episodes.jsonl")
    episodes = read_jsonl(meta / "episodes.jsonl")
    require(info["fps"] == cfg["fps"] == 30, "FPS 必须 30")
    require(len(records) == len(episodes) == info["total_episodes"], "episode 数不一致")
    require([r["source"] for r in records] == [str(p) for p in discover_inputs(cfg)], "指定 batch 来源不一致")
    require([r["episode_index"] for r in records] == list(range(len(records))), "episode 编号不连续")
    expected_columns = {k for k, v in info["features"].items() if v["dtype"] != "video"}
    require(
        expected_columns
        == {"observation.state", "action", "timestamp", "frame_index", "index", "task_index", "episode_index"},
        "特征字段不符",
    )
    require(
        {k for k, v in info["features"].items() if v["dtype"] == "video"}
        == {f"observation.images.{c}" for c in CAMERAS},
        "相机特征错误",
    )
    require(read_jsonl(meta / "tasks.jsonl") == [{"task_index": 0, "task": cfg["language"]}], "tasks 不一致")
    require(len(list((root / "data").rglob("*.parquet"))) == len(records), "Parquet 数错误")
    require(len(list((root / "videos").rglob("*.mp4"))) == 2 * len(records), "视频数错误")
    require(len(read_jsonl(meta / "episodes_stats.jsonl")) == len(records), "逐 episode 统计不完整")
    offsets = np.cumsum([0] + [r["length"] for r in records[:-1]]).tolist()
    jobs = [(root, r, e, cfg, offset) for r, e, offset in zip(records, episodes, offsets, strict=True)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        checked = list(pool.map(validate_episode, jobs))
    frames = sum(r["frames"] for r in checked)
    require(frames == info["total_frames"], "总帧数错误")
    require(sum(r["videos"] for r in checked) == info["total_videos"], "总视频数错误")
    by_batch = {}
    for record in records:
        batch = by_batch.setdefault(record["batch"], {"episodes": 0, "frames": 0})
        batch["episodes"] += 1
        batch["frames"] += record["length"]
    result = {
        "validated": True,
        "episodes": len(records),
        "frames": frames,
        "videos": info["total_videos"],
        "fps": 30,
        "batches": by_batch,
        "checks": [
            "source_sha256",
            "source_state_interpolation",
            "next_state_action",
            "language",
            "indices",
            "all_video_frames_and_pts",
        ],
        "full_tempo_loader_tested": False,
        "training_tested": False,
    }
    write_json(meta / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers 必须大于零")
    print(json.dumps(validate_dataset(args.root.resolve(), args.workers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
