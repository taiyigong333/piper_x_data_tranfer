import numpy as np

from piperx_to_openpi_vla.convert import (
    _episode_files,
    _interpolate_state,
    _resolve_instruction,
    resize_with_padding,
)


def test_resize_with_padding_outputs_square_black_padded_image():
    image = np.full((100, 200, 3), 255, dtype=np.uint8)

    result = resize_with_padding(image)

    assert result.shape == (224, 224, 3)
    assert np.all(result[:56] == 0)
    assert np.all(result[168:] == 0)
    assert np.all(result[56:168] > 0)


def test_high_rate_state_interpolates_to_camera_timestamps():
    times = np.array([0.0, 0.01, 0.02])
    pose = np.column_stack([times * 10] * 6)
    gripper = np.array([[0.0], [1.0], [2.0]])

    result = _interpolate_state(times, pose, gripper, np.array([0.005, 0.015]))

    assert result.shape == (2, 7)
    np.testing.assert_allclose(result[:, 0], [0.05, 0.15])
    np.testing.assert_allclose(result[:, 6], [0.5, 1.5])


def test_episode_discovery_only_includes_forward_trajectories(tmp_path):
    forward = tmp_path / "episode_1" / "forward" / "trajectory.hdf5"
    reverse = tmp_path / "episode_1" / "reverse" / "trajectory.hdf5"
    batch_forward = tmp_path / "forward" / "trajectory.hdf5"
    forward.parent.mkdir(parents=True)
    reverse.parent.mkdir(parents=True)
    batch_forward.parent.mkdir(parents=True)
    forward.touch()
    reverse.touch()
    batch_forward.touch()

    assert _episode_files(tmp_path) == [batch_forward, forward]


def test_language_instruction_can_be_overridden_or_preserved():
    assert _resolve_instruction("grab the rotating red square", None) == "grab the rotating red square"
    assert _resolve_instruction("grab the rotating red square", "pick up the red block") == "pick up the red block"


def test_language_instruction_override_rejects_blank_text():
    try:
        _resolve_instruction("original", "   ")
    except ValueError as error:
        assert "不能为空" in str(error)
    else:
        raise AssertionError("空 language instruction 应该报错")
