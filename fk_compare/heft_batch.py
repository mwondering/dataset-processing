"""GPU-batched HEFT expansion of a 36D G1 pose representation.

The minimal pose layout is::

    root position (3) + root quaternion wxyz (4) + joint position (29)

This is HEFT's native float32 minimal-motion representation.  All quaternions
in this module use IsaacLab/MuJoCo ``wxyz`` ordering.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


ISAACLAB_G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

ISAACLAB_G1_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

DATA10K_TERMS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

TERM_UNITS = {
    "joint_pos": "rad",
    "joint_vel": "rad/s",
    "root_lin_vel_w": "m/s",
    "root_ang_vel_w": "rad/s",
    "body_pos_w": "m",
    "body_quat_w": "rad_geodesic",
    "body_lin_vel_w": "m/s",
    "body_ang_vel_w": "rad/s",
}


def normalize(value: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(eps)


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    out = quat.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = torch.unbind(left, dim=-1)
    rw, rx, ry, rz = torch.unbind(right, dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]
    w = quat[..., :1]
    cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + w * cross + torch.cross(xyz, cross, dim=-1)


def quat_apply_inverse(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]
    w = quat[..., :1]
    cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector - w * cross + torch.cross(xyz, cross, dim=-1)


def _time_gather(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2 or indices.ndim != 2 or value.shape[:2] != indices.shape:
        raise ValueError(f"Invalid time gather shapes: value={tuple(value.shape)}, indices={tuple(indices.shape)}")
    index = indices.reshape(indices.shape + (1,) * (value.ndim - 2)).expand_as(value)
    return torch.gather(value, dim=1, index=index)


def _fps_tensor(fps: float | torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(fps, dtype=torch.float32, device=device).reshape(-1)
    if value.numel() == 1:
        value = value.expand(batch)
    if value.numel() != batch:
        raise ValueError(f"Expected one FPS per sequence, got {value.numel()} for batch={batch}")
    if bool((value <= 0.0).any()):
        raise ValueError("FPS must be positive")
    return value


def finite_difference(
    value: torch.Tensor,
    lengths: torch.Tensor,
    fps: float | torch.Tensor,
) -> torch.Tensor:
    """Length-aware HEFT central difference for padded ``[B,T,...]`` data."""

    if value.ndim < 2:
        raise ValueError(f"Expected [B,T,...], got {tuple(value.shape)}")
    batch, steps = value.shape[:2]
    lengths = lengths.to(device=value.device, dtype=torch.long).reshape(-1)
    if lengths.numel() != batch or bool((lengths < 1).any()) or bool((lengths > steps).any()):
        raise ValueError(f"Invalid lengths {lengths.tolist()} for shape {tuple(value.shape)}")
    fps_b = _fps_tensor(fps, batch, value.device)
    time = torch.arange(steps, device=value.device).unsqueeze(0).expand(batch, -1)
    last = lengths.unsqueeze(1) - 1
    previous = torch.maximum(time - 1, torch.zeros_like(time))
    following = torch.minimum(time + 1, last)
    delta = _time_gather(value, following) - _time_gather(value, previous)
    endpoint = (time == 0) | (time == last)
    scale = torch.where(endpoint, fps_b.unsqueeze(1), fps_b.unsqueeze(1) * 0.5)
    scale = scale.reshape(scale.shape + (1,) * (value.ndim - 2))
    valid = (time < lengths.unsqueeze(1)).reshape((batch, steps) + (1,) * (value.ndim - 2))
    return torch.where(valid, delta * scale, torch.zeros_like(delta))


def make_quat_continuous(quat: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if quat.ndim < 3 or quat.shape[-1] != 4:
        raise ValueError(f"Expected [B,T,...,4], got {tuple(quat.shape)}")
    batch, steps = quat.shape[:2]
    lengths = lengths.to(device=quat.device, dtype=torch.long).reshape(-1)
    flat = normalize(quat).reshape(batch, steps, -1, 4)
    if steps == 1:
        return flat.reshape_as(quat)
    dots = (flat[:, 1:] * flat[:, :-1]).sum(dim=-1)
    pair_time = torch.arange(1, steps, device=quat.device).view(1, -1, 1)
    pair_valid = pair_time < lengths.view(-1, 1, 1)
    signs = torch.where((dots < 0.0) & pair_valid, -torch.ones_like(dots), torch.ones_like(dots))
    signs = torch.cat((torch.ones_like(signs[:, :1]), signs), dim=1)
    flat = flat * torch.cumprod(signs, dim=1).unsqueeze(-1)
    return flat.reshape_as(quat)


def angular_velocity_from_quat(
    quat: torch.Tensor,
    lengths: torch.Tensor,
    fps: float | torch.Tensor,
) -> torch.Tensor:
    continuous = make_quat_continuous(quat, lengths)
    qdot = finite_difference(continuous, lengths, fps)
    return 2.0 * quat_mul(qdot, quat_conjugate(continuous))[..., 1:]


def smooth_avg5(value: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Replicate-padded five-point average without crossing sequence ends."""

    batch, steps = value.shape[:2]
    lengths = lengths.to(device=value.device, dtype=torch.long).reshape(-1)
    time = torch.arange(steps, device=value.device).unsqueeze(0).expand(batch, -1)
    last = lengths.unsqueeze(1) - 1
    total = torch.zeros_like(value)
    for offset in (-2, -1, 0, 1, 2):
        indices = torch.minimum(torch.maximum(time + offset, torch.zeros_like(time)), last)
        total = total + _time_gather(value, indices)
    valid = (time < lengths.unsqueeze(1)).reshape((batch, steps) + (1,) * (value.ndim - 2))
    return torch.where(valid, total * 0.2, torch.zeros_like(total))


def expand_pos36(
    pos36: torch.Tensor,
    lengths: torch.Tensor,
    fps: float | torch.Tensor,
    fk_helper: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Run HEFT FK and smoothing from batched native 36D poses.

    Returns ``(processed, raw_velocity)``.  The processed dictionary includes
    the exact six arrays written to a Data10k ``motion.npz`` plus root
    velocities used for diagnostics.  Despite the legacy key name,
    ``body_lin_vel_w`` is the world velocity of each link origin, matching
    HEFT and mjlab semantics.
    """

    if pos36.ndim != 3 or pos36.shape[-1] != 36:
        raise ValueError(f"Expected [B,T,36] pose input, got {tuple(pos36.shape)}")
    if pos36.dtype != torch.float32:
        raise ValueError(f"Expected float32 poses, got {pos36.dtype}")
    batch, steps = pos36.shape[:2]
    lengths = lengths.to(device=pos36.device, dtype=torch.long).reshape(-1)
    if lengths.numel() != batch or bool((lengths < 1).any()) or bool((lengths > steps).any()):
        raise ValueError(f"Invalid lengths {lengths.tolist()} for shape {tuple(pos36.shape)}")
    root_pos_w = pos36[..., :3]
    root_quat_input = pos36[..., 3:7]
    root_quat_norm = torch.linalg.vector_norm(root_quat_input, dim=-1)
    valid = torch.arange(steps, device=pos36.device).unsqueeze(0) < lengths.unsqueeze(1)
    zero_valid_quat = (root_quat_norm < 1.0e-6) & valid
    if bool(zero_valid_quat.any()):
        examples = zero_valid_quat.nonzero(as_tuple=False)[:5].detach().cpu().tolist()
        raise ValueError(f"Zero root quaternions at batch/time indices {examples}")
    # HEFT normalizes the input but preserves its q/-q sign in output poses.
    # Sign continuity is applied only inside angular-velocity calculation.
    root_quat_w = normalize(root_quat_input)
    joint_pos = pos36[..., 7:]
    if joint_pos.shape[-1] != len(ISAACLAB_G1_JOINT_NAMES):
        raise ValueError(f"Expected 29 G1 joints, got {joint_pos.shape[-1]}")

    root_lin_raw = finite_difference(root_pos_w, lengths, fps)
    root_ang_raw = angular_velocity_from_quat(root_quat_w, lengths, fps)
    joint_vel_raw = finite_difference(joint_pos, lengths, fps)
    root_lin_smooth = smooth_avg5(root_lin_raw, lengths)
    root_ang_smooth = smooth_avg5(root_ang_raw, lengths)
    joint_vel_smooth = smooth_avg5(joint_vel_raw, lengths)

    body_pos_b, body_quat_b = fk_helper.body_pose(joint_pos)
    body_quat_b = normalize(body_quat_b)
    body_pos_w = quat_apply(root_quat_w.unsqueeze(2), body_pos_b) + root_pos_w.unsqueeze(2)
    body_quat_w = normalize(quat_mul(root_quat_w.unsqueeze(2), body_quat_b))

    body_pos_b_derivative = finite_difference(body_pos_b, lengths, fps)
    root_ang_b_raw = quat_apply_inverse(root_quat_w, root_ang_raw)
    root_ang_b_smooth = quat_apply_inverse(root_quat_w, root_ang_smooth)
    # HEFT smooths root angular velocity before using it in the root-relative
    # body velocity decomposition, then smooths the resulting body term again.
    body_lin_b_before_final_smooth = body_pos_b_derivative + torch.cross(
        root_ang_b_smooth.unsqueeze(2), body_pos_b, dim=-1
    )
    # Keep a fully unsmoothed counterpart only for the diagnostic report.
    body_lin_b_fully_raw = body_pos_b_derivative + torch.cross(
        root_ang_b_raw.unsqueeze(2), body_pos_b, dim=-1
    )
    body_ang_b_raw = angular_velocity_from_quat(body_quat_b, lengths, fps)
    body_lin_b_smooth = smooth_avg5(body_lin_b_before_final_smooth, lengths)
    body_ang_b_smooth = smooth_avg5(body_ang_b_raw, lengths)

    body_lin_w_raw = quat_apply(root_quat_w.unsqueeze(2), body_lin_b_fully_raw) + root_lin_raw.unsqueeze(2)
    body_ang_w_raw = quat_apply(root_quat_w.unsqueeze(2), body_ang_b_raw) + root_ang_raw.unsqueeze(2)
    body_lin_w_smooth = quat_apply(root_quat_w.unsqueeze(2), body_lin_b_smooth) + root_lin_smooth.unsqueeze(2)
    body_ang_w_smooth = quat_apply(root_quat_w.unsqueeze(2), body_ang_b_smooth) + root_ang_smooth.unsqueeze(2)

    processed = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel_smooth,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_w_smooth,
        "body_ang_vel_w": body_ang_w_smooth,
        "root_lin_vel_w": root_lin_smooth,
        "root_ang_vel_w": root_ang_smooth,
    }
    raw_velocity = {
        "joint_vel": joint_vel_raw,
        "body_lin_vel_w": body_lin_w_raw,
        "body_ang_vel_w": body_ang_w_raw,
        "root_lin_vel_w": root_lin_raw,
        "root_ang_vel_w": root_ang_raw,
    }
    return processed, raw_velocity


def term_error(term: str, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Return scalar, vector-L2, or sign-invariant quaternion error."""

    before64 = np.asarray(before, dtype=np.float64)
    after64 = np.asarray(after, dtype=np.float64)
    if before64.shape != after64.shape:
        raise ValueError(f"Shape mismatch for {term}: {before64.shape} != {after64.shape}")
    if term == "body_quat_w":
        before64 /= np.maximum(np.linalg.norm(before64, axis=-1, keepdims=True), 1.0e-12)
        after64 /= np.maximum(np.linalg.norm(after64, axis=-1, keepdims=True), 1.0e-12)
        dot = np.abs(np.sum(before64 * after64, axis=-1))
        return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
    difference = before64 - after64
    if term.startswith("joint_"):
        return np.abs(difference)
    return np.linalg.norm(difference, axis=-1)


def error_stats(error: np.ndarray, *, unit: str) -> dict[str, float | int | str]:
    flat = np.asarray(error, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise ValueError("Cannot summarize an empty error array")
    return {
        "unit": unit,
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "rmse": float(np.sqrt(np.mean(flat * flat))),
        "p95": float(np.percentile(flat, 95)),
        "max": float(flat.max()),
    }


def compare_terms(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    terms: tuple[str, ...] | list[str],
) -> tuple[dict[str, dict[str, float | int | str]], dict[str, np.ndarray]]:
    summary: dict[str, dict[str, float | int | str]] = {}
    errors: dict[str, np.ndarray] = {}
    for term in terms:
        error = term_error(term, before[term], after[term])
        errors[term] = error
        summary[term] = error_stats(error, unit=TERM_UNITS[term])
    return summary, errors
