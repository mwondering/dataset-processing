import numpy as np

from fk_compare.motion import angular_velocity_world, make_quat_continuous, slerp_pair


def test_quaternion_sign_continuity():
    q = np.asarray([[1.0, 0, 0, 0], [-1.0, 0, 0, 0]])
    np.testing.assert_allclose(make_quat_continuous(q), [[1, 0, 0, 0], [1, 0, 0, 0]])


def test_slerp_half_turn():
    q0 = np.asarray([[1.0, 0, 0, 0]])
    q1 = np.asarray([[0.0, 0, 0, 1.0]])
    actual = slerp_pair(q0, q1, np.asarray([0.5]))
    np.testing.assert_allclose(actual, [[2**-0.5, 0, 0, 2**-0.5]], atol=1e-7)


def test_world_angular_velocity():
    angle = np.asarray([0.0, 0.1, 0.2, 0.3])
    q = np.stack((np.cos(angle / 2), np.zeros(4), np.zeros(4), np.sin(angle / 2)), axis=-1)
    expected = np.tile([0.0, 0.0, 1.0], (4, 1))
    np.testing.assert_allclose(angular_velocity_world(q, 0.1), expected, atol=1e-7)
