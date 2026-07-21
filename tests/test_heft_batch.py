import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fk_compare.heft_batch import (
    angular_velocity_from_quat,
    expand_pos36,
    finite_difference,
    smooth_avg5,
)


def test_length_aware_difference_does_not_read_padding():
    value = torch.tensor([[[0.0], [1.0], [4.0], [999.0], [999.0]]])
    actual = finite_difference(value, torch.tensor([3]), fps=2.0)
    expected = torch.tensor([[[2.0], [4.0], [6.0], [0.0], [0.0]]])
    torch.testing.assert_close(actual, expected)


def test_heft_replicate_padded_avg5():
    value = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
    actual = smooth_avg5(value, torch.tensor([5]))
    expected = torch.tensor([0.6, 1.2, 2.0, 2.8, 3.4]).reshape(1, 5, 1)
    torch.testing.assert_close(actual, expected)


def test_constant_rotation_has_zero_angular_velocity():
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(2, 6, 4).clone()
    actual = angular_velocity_from_quat(quat, torch.tensor([6, 4]), fps=torch.tensor([50.0, 30.0]))
    torch.testing.assert_close(actual, torch.zeros_like(actual))


class _TwoBodyFakeFK:
    def body_pose(self, joint_pos):
        prefix = joint_pos.shape[:-1]
        pos = torch.zeros(prefix + (2, 3), dtype=joint_pos.dtype, device=joint_pos.device)
        pos[..., 1, 0] = 1.0
        quat = torch.zeros(prefix + (2, 4), dtype=joint_pos.dtype, device=joint_pos.device)
        quat[..., 0] = 1.0
        return pos, quat


def test_expand_pos36_uses_only_pose_and_preserves_root_link():
    frames = 7
    pos36 = torch.zeros((1, frames, 36), dtype=torch.float32)
    pos36[0, :, 0] = torch.arange(frames, dtype=torch.float32) * 0.02
    pos36[..., 3] = 2.0
    processed, raw = expand_pos36(
        pos36,
        lengths=torch.tensor([frames]),
        fps=50.0,
        fk_helper=_TwoBodyFakeFK(),
    )

    np.testing.assert_allclose(processed["body_pos_w"][0, :, 0].numpy(), pos36[0, :, :3].numpy())
    np.testing.assert_allclose(processed["body_pos_w"][0, :, 1, 0].numpy(), pos36[0, :, 0].numpy() + 1.0)
    torch.testing.assert_close(
        processed["body_quat_w"][0, :, 0],
        torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(frames, 4),
    )
    torch.testing.assert_close(processed["body_lin_vel_w"], raw["body_lin_vel_w"], atol=1e-6, rtol=1e-6)


def test_expand_pos36_preserves_quaternion_sign_but_angvel_is_continuous():
    pos36 = torch.zeros((1, 4, 36), dtype=torch.float32)
    pos36[0, :, 3] = torch.tensor([1.0, -1.0, 1.0, -1.0])
    processed, _ = expand_pos36(
        pos36,
        lengths=torch.tensor([4]),
        fps=50.0,
        fk_helper=_TwoBodyFakeFK(),
    )

    torch.testing.assert_close(processed["body_quat_w"][0, :, 0, 0], pos36[0, :, 3])
    torch.testing.assert_close(processed["root_ang_vel_w"], torch.zeros((1, 4, 3)))
