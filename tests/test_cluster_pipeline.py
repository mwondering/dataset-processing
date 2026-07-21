from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fk_compare.g1_fk import G1PureTorchFK
from scripts.process_dataset_multigpu import (
    balanced_shards,
    build_progress_lines,
    merge_global_differences,
)


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


def test_global_progress_aggregates_worker_frames_and_motions():
    plan = {"motion_count": 20, "frame_count": 2_000}
    workers = [
        {"gpu_id": 0, "motion_count": 10},
        {"gpu_id": 1, "motion_count": 10},
    ]
    states = [
        {"completed_motion_count": 4, "completed_frame_count": 400},
        {"completed_motion_count": 6, "completed_frame_count": 500},
    ]

    lines = build_progress_lines(plan, workers, states, elapsed=10.0)

    assert lines[0].startswith("overall   45.0%")
    assert lines[1] == "motions  10 / 20"
    assert lines[2] == "frames   900 / 2,000"
    assert lines[3] == "speed    90 frame/s"
    assert lines[4] == "ETA      00:00:12"
    assert lines[5] == "GPU0     4 / 10 motions"
    assert lines[6] == "GPU1     6 / 10 motions"
