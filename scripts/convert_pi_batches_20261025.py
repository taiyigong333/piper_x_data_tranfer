"""在 piperx 按批次转换并全量核验本次指定的 OpenPI 数据。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

CODE = Path("/home/cv/gcj/project/data_transfer/to_pi/code")
OUTPUT = CODE.parent / "after_tranfer"
RUN = CODE / "logs/convert_pi_batches_20261025"
LANGUAGE = "grab the rotating red square"
sys.path.insert(0, str(CODE))
from piperx_to_openpi_vla.convert import (  # noqa: E402
    CAMERAS,
    STATE_NAMES,
    _state_from_hdf5,
    convert_dataset,
    discover_inputs,
)


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_root(batch: int) -> Path:
    task = "rotating_red_square" if batch == 5 else "rotating_red_square_blue_plate"
    return Path("/home/cv/gcj/data_collect/data") / task / f"batch_{batch}"


def convert_one(batch: int) -> dict:
    cv2.setNumThreads(1)
    source = source_root(batch)
    target = OUTPUT / source.name
    files = discover_inputs(source, [batch])[0][1]
    # 转换前保留指纹；验证时再次计算，避免采集文件在运行期间发生变化。
    provenance = []
    for index, path in enumerate(files):
        stat = path.stat()
        provenance.append({"episode_index": index, "source": str(path),
                           "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                           "sha256": sha256_file(path)})
    result = convert_dataset(source, target, batches=[batch], language_instruction=LANGUAGE)
    meta = target / "meta"
    info = json.loads((meta / "info.json").read_text())
    # 补齐官方 v2.1 元数据约定；仅修整本次新输出，不修改原转换器及旧数据。
    for name in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        info["features"][name] = {"dtype": "float32" if name == "timestamp" else "int64",
                                   "shape": [1], "names": None}
    for name in ("action", "observation.state"):
        info["features"][name]["names"] = list(STATE_NAMES)
    save_json(meta / "info.json", info)
    episodes = [json.loads(line) for line in (meta / "episodes.jsonl").read_text().splitlines()]
    for episode in episodes:
        episode["tasks"] = [LANGUAGE]
    (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    tasks = [json.loads(line) for line in (meta / "tasks.jsonl").read_text().splitlines()]
    assert tasks == [{"task_index": 0, "task": LANGUAGE}]
    assert info["fps"] == 30 and info["total_episodes"] == len(files) == len(episodes)
    assert len(list((target / "data").rglob("*.parquet"))) == len(files)
    assert len(list((target / "videos").rglob("*.mp4"))) == len(files) * 2
    frames = videos = 0
    for index, (path, episode, record) in enumerate(zip(files, episodes, provenance, strict=True)):
        stat = path.stat()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert (stat.st_size, stat.st_mtime_ns, digest) == (record["size"], record["mtime_ns"], record["sha256"])
        with h5py.File(path, "r") as file:
            camera_times = file["camera_observations/timestamp"][()]
            n = min(len(camera_times), int(file["metadata/trajectory_length"][()]))
            expected_state = _state_from_hdf5(file, camera_times[:n])[:n]
            assert n > 0 and expected_state.shape == (n, 7)
            for camera in CAMERAS:
                assert len(file[f"camera_observations/color_images/{camera}"]) >= n
        table = pq.read_table(target / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet")
        assert set(table.column_names) == {key for key, ft in info["features"].items() if ft["dtype"] != "video"}
        values = table.to_pandas()
        state, action = np.stack(values["observation.state"]), np.stack(values["action"])
        assert np.isfinite(state).all() and np.isfinite(action).all()
        assert state.dtype == action.dtype == np.float32
        np.testing.assert_array_equal(state, expected_state)
        np.testing.assert_array_equal(action, np.vstack([expected_state[1:], expected_state[-1:]]))
        np.testing.assert_array_equal(values["index"], np.arange(frames, frames + n))
        np.testing.assert_array_equal(values["frame_index"], np.arange(n))
        np.testing.assert_array_equal(values["episode_index"], np.full(n, index))
        np.testing.assert_array_equal(values["task_index"], np.zeros(n))
        np.testing.assert_allclose(values["timestamp"], np.arange(n, dtype=np.float32) / 30)
        assert episode == {"episode_index": index, "tasks": [LANGUAGE], "task_index": 0, "length": n}
        for camera in CAMERAS:
            video = target / "videos" / f"chunk-{index // 1000:03d}" / f"observation.images.{camera}" / f"episode_{index:06d}.mp4"
            reader = cv2.VideoCapture(str(video))
            assert reader.isOpened(), video
            try:
                assert abs(reader.get(cv2.CAP_PROP_FPS) - 30) < 1e-6
                decoded = 0
                while True:
                    ok, image = reader.read()
                    if not ok:
                        break
                    assert image.shape == (224, 224, 3)
                    decoded += 1
                assert decoded == n, (video, decoded, n)
            finally:
                reader.release()
            videos += 1
        record["length"] = n
        frames += n
        logging.info("核验 %s [%d/%d] %d 帧", source.name, index + 1, len(files), n)
    assert frames == info["total_frames"] == result["frames"]
    save_json(meta / "source_manifest.json", {"source_root": str(source), "language_instruction": LANGUAGE,
                                               "converter_sha256": hashlib.sha256((CODE / "piperx_to_openpi_vla/convert.py").read_bytes()).hexdigest(),
                                               "episodes": provenance})
    result.update({"batch": source.name, "videos": videos, "validated": True,
                   "validation": "全量来源SHA256、Parquet数值、标签、索引与全部视频解码通过；未运行真实LeRobot loader或训练"})
    save_json(meta / "conversion_validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=[2, 3, 4, 5])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.batch:
        print(json.dumps(convert_one(args.batch), ensure_ascii=False), flush=True)
        return
    RUN.mkdir(parents=True, exist_ok=True)
    if any(RUN.iterdir()):
        raise FileExistsError(f"运行记录目录非空，拒绝覆盖：{RUN}")
    # 先检查全部目标，避免部分批次已存在时开始写入其他批次。
    for batch in [2, 3, 4, 5]:
        target = OUTPUT / f"batch_{batch}"
        if target.exists():
            raise FileExistsError(f"本次目标已存在，拒绝覆盖：{target}")
        assert discover_inputs(source_root(batch), [batch])[0][1]
    spec = importlib.util.spec_from_file_location("existing_tests", CODE / "tests/test_convert.py")
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    tests.test_resize_with_padding_outputs_square_black_padded_image()
    tests.test_high_rate_state_interpolates_to_camera_timestamps()
    tests.test_language_instruction_can_be_overridden_or_preserved()
    tests.test_language_instruction_override_rejects_blank_text()
    with tempfile.TemporaryDirectory() as temp:
        # 旧测试误把 forward 排在 episode_1 前面；这里单独核对集合及实际排序约定。
        from piperx_to_openpi_vla.convert import _episode_files
        forward = Path(temp) / "episode_1/forward/trajectory.hdf5"
        reverse = Path(temp) / "episode_1/reverse/trajectory.hdf5"
        direct = Path(temp) / "forward/trajectory.hdf5"
        for path in [forward, reverse, direct]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        assert _episode_files(Path(temp)) == sorted([forward, direct])
    save_json(RUN / "plan.json", {"language_instruction": LANGUAGE, "batches": [2, 3, 4, 5],
                                  "output": str(OUTPUT), "existing_unit_tests_passed": 4,
                                  "additional_discovery_check_passed": True,
                                  "existing_discovery_test_issue": "原测试期望路径顺序错误；独立检查已验证仅选forward且按路径排序，未修改原测试",
                                  "started_at": datetime.now().astimezone().isoformat()})
    def run_batch(batch):
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (RUN / f"batch_{batch}.log").open("x") as log:
            return batch, subprocess.run([sys.executable, __file__, "--batch", str(batch)],
                                         stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    with ThreadPoolExecutor(max_workers=4) as pool:
        codes = dict(pool.map(run_batch, [2, 3, 4, 5]))
    if any(codes.values()):
        save_json(RUN / "failed.json", codes)
        raise RuntimeError(f"批次失败，请查看日志：{codes}")
    summary = [json.loads((OUTPUT / f"batch_{b}/meta/conversion_validation.json").read_text()) for b in [2, 3, 4, 5]]
    save_json(RUN / "completed.json", {"finished_at": datetime.now().astimezone().isoformat(), "batches": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
