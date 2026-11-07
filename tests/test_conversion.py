"""覆盖时间对齐、相机映射、动作代理和完整导出的关键约束。"""

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

from piperx_to_tempo_vla.convert import (
    align_episode,
    audit_dataset,
    convert_dataset,
    discover_inputs,
    resize_with_padding,
)
from piperx_to_tempo_vla.tempo_entry import verify_dataset_contract
from piperx_to_tempo_vla.validate import validate_dataset


@pytest.fixture
def sample(tmp_path):
    root = tmp_path / "input"
    source = root / "batch_2/session/forward/trajectory.hdf5"
    source.parent.mkdir(parents=True)
    camera_t = np.array([0, 0.031, 0.068, 0.102, 0.134])
    robot_t = np.arange(-0.02, 0.16, 0.01)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :, 2] = 240  # 用红色验证 BGR/RGB 往返没有交换通道。
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    with h5py.File(source, "w") as f:
        f["metadata/trajectory_length"] = len(camera_t)
        f["metadata/batch_tag"] = 999  # 故意与目录不同，筛选仍须以目录为准。
        f["metadata/language_instruction"] = "original instruction"
        f["camera_observations/timestamp"] = camera_t
        for name in ("camera_front", "camera_wrist_right"):
            d = f.create_dataset(
                f"camera_observations/color_images/{name}", (len(camera_t),), dtype=h5py.vlen_dtype(np.dtype("uint8"))
            )
            for i in range(len(camera_t)):
                d[i] = encoded
        for name, dim in (("arm_single_position_raw", 6), ("end_effector_single_position_raw", 1)):
            f[f"puppet/{name}/timestamp"] = robot_t
            f[f"puppet/{name}/data"] = np.column_stack([(i + 1) * robot_t for i in range(dim)])
    cfg = {
        "inputs": [{"root": str(root), "batches": [2]}],
        "output": str(tmp_path / "output"),
        "fps": 30,
        "image_size": [224, 224],
        "language": "grab the rotating red square",
        "max_image_error_s": 0.1,
        "max_robot_gap_s": 0.1,
        "workers": 1,
        "action_source": "next_observed_state",
    }
    return source, cfg


def test_grid_and_interpolation(sample):
    source, cfg = sample
    with h5py.File(source) as f:
        plan = align_episode(f, cfg)
    np.testing.assert_allclose(np.diff(plan["times"]), 1 / 30, rtol=1e-12)
    np.testing.assert_allclose(plan["state"][:, 0], np.arange(5) / 30, atol=1e-8)
    assert plan["indices"].tolist() == [0, 1, 2, 3, 4]


def test_padding():
    padded = resize_with_padding(np.full((480, 640, 3), 255, dtype=np.uint8))
    assert padded.shape == (224, 224, 3)
    assert np.all(padded[:28] == 0) and np.all(padded[-28:] == 0)
    assert np.all(padded[28:196] == 255)


def test_forward_allowlist_and_missing_batch(sample):
    source, cfg = sample
    reverse = source.parent.parent / "reverse/trajectory.hdf5"
    reverse.parent.mkdir()
    reverse.touch()
    assert discover_inputs(cfg) == [source]
    cfg["inputs"][0]["batches"] = [2, 3]
    with pytest.raises(FileNotFoundError):
        discover_inputs(cfg)


def test_bad_timestamps_and_large_gap(sample):
    source, cfg = sample
    with h5py.File(source, "r+") as f:
        f["camera_observations/timestamp"][1] = 0
        with pytest.raises(ValueError, match="严格递增"):
            align_episode(f, cfg)
        f["camera_observations/timestamp"][1] = 0.031
        cfg["max_robot_gap_s"] = 0.005
        with pytest.raises(ValueError, match="采样间隙"):
            align_episode(f, cfg)


def test_full_export(sample):
    source, cfg = sample
    output = Path(cfg["output"])
    result = convert_dataset(cfg, audit_dataset(cfg))
    assert result["episodes"] == 1 and result["frames"] == 5
    validated = validate_dataset(output, workers=1)
    assert validated["validated"] and validated["videos"] == 2
    table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    state, action = (np.array(table[k].to_pylist()) for k in ("observation.state", "action"))
    np.testing.assert_array_equal(action[:-1], state[1:])
    np.testing.assert_array_equal(action[-1], state[-1])
    info = json.loads((output / "meta/info.json").read_text())
    assert "observation.images.left_wrist" in info["features"]
    assert "observation.images.right_wrist" not in info["features"]
    verify_dataset_contract(output)
    invalid_info = dict(info, fps=60)
    (output / "meta/info.json").write_text(json.dumps(invalid_info))
    with pytest.raises(ValueError, match="30Hz"):
        verify_dataset_contract(output)
    (output / "meta/info.json").write_text(json.dumps(info))
    import av

    with av.open(str(output / "videos/chunk-000/observation.images.head/episode_000000.mp4")) as c:
        rgb = next(c.decode(video=0)).to_ndarray(format="rgb24")
    assert rgb[112, 112, 0] > 200 and rgb[112, 112, 2] < 20
    with h5py.File(source) as f:
        assert f["metadata/language_instruction"][()].decode() == "original instruction"
    with pytest.raises(FileExistsError):
        convert_dataset(cfg, audit_dataset(cfg))
