from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fk_compare.g1_fk import G1PureTorchFK
from scripts.process_dataset_multigpu import balanced_shards, merge_global_differences


def test_embedded_g1_fk_zero_pose_contract():
    helper = G1PureTorchFK("cpu")
    body_pos, body_quat = helper.body_pose(torch.zeros((2, 29), dtype=torch.float32))

    assert body_pos.shape == (2, 30, 3)
    assert body_quat.shape == (2, 30, 4)
    torch.testing.assert_close(body_pos[:, 0], torch.zeros((2, 3)))
    torch.testing.assert_close(
        body_pos[:, 1],
        torch.tensor([0.0, 0.064452, -0.1027]).expand(2, 3),
        atol=1e-7,
        rtol=1e-7,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(body_quat, dim=-1),
        torch.ones((2, 30)),
        atol=1e-6,
        rtol=1e-6,
    )


def test_balanced_shards_are_deterministic_and_cover_every_motion():
    specs = [SimpleNamespace(path=f"motion_{index}", length=length) for index, length in enumerate((9, 8, 7, 6))]
    shards, loads = balanced_shards(specs, 2)

    assert sorted(item.path for shard in shards for item in shard) == [f"motion_{index}" for index in range(4)]
    assert loads == [15, 15]


def test_merge_global_differences_uses_weighted_moments():
    summaries = [
        {"global_differences": {"x": {"term": {"unit": "m", "count": 2, "mean": 1.0, "rmse": 1.0, "max": 1.0}}}},
        {"global_differences": {"x": {"term": {"unit": "m", "count": 1, "mean": 2.0, "rmse": 2.0, "max": 2.0}}}},
    ]
    result = merge_global_differences(summaries)["x"]["term"]

    assert result["count"] == 3
    assert result["mean"] == pytest.approx(4.0 / 3.0)
    assert result["rmse"] == pytest.approx(2.0**0.5)
    assert result["max"] == 2.0
