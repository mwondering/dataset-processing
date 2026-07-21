from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Found a zero quaternion")
    return q / norm


def make_quat_continuous(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q.copy())
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0.0:
            q[i] *= -1.0
    return q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ), axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    q = np.where(q[..., :1] < 0.0, -q, q)
    xyz = q[..., 1:]
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, np.clip(q[..., 0], -1.0, 1.0))
    scale = np.where(sin_half > 1e-8, angle / np.maximum(sin_half, 1e-8), 2.0)
    return xyz * scale[..., None]


def slerp_pair(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1)
    q1 = np.where((dot < 0.0)[:, None], -q1, q1)
    dot = np.abs(dot)
    linear = dot > 0.9995
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    safe = np.where(np.abs(sin_theta) < 1e-8, 1.0, sin_theta)
    w0 = np.sin((1.0 - alpha) * theta) / safe
    w1 = np.sin(alpha * theta) / safe
    out = w0[:, None] * q0 + w1[:, None] * q1
    lerped = (1.0 - alpha[:, None]) * q0 + alpha[:, None] * q1
    out = np.where(linear[:, None], lerped, out)
    return normalize_quat(out)


def angular_velocity_world(q: np.ndarray, dt: float) -> np.ndarray:
    q = make_quat_continuous(q)
    if len(q) < 3:
        raise ValueError("At least three frames are required")
    rel = quat_mul(q[2:], quat_conjugate(q[:-2]))
    middle = quat_to_rotvec(rel) / (2.0 * dt)
    return np.concatenate((middle[:1], middle, middle[-1:]), axis=0)


def differentiate(x: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(x, dt, axis=0, edge_order=1)


def load_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    with path.open(newline="") as stream:
        header = next(csv.reader(stream))
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 36 or len(header) != 36:
        raise ValueError(f"Expected a 36-column CSV, got {data.shape}")
    expected_quat = ["root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z"]
    if header[3:7] != expected_quat:
        raise ValueError(f"Expected wxyz quaternion columns, got {header[3:7]}")
    expected_dofs = ["dof_" + n.replace("_joint", "_link") + "(rad)" for n in JOINT_NAMES]
    # The waist pitch DOF is historically named torso_link in this dataset.
    expected_dofs[14] = "dof_torso_link(rad)"
    if header[7:] != expected_dofs:
        mismatches = [(a, b) for a, b in zip(header[7:], expected_dofs) if a != b]
        raise ValueError(f"Unexpected DOF columns: {mismatches}")
    if not np.isfinite(data).all():
        raise ValueError("CSV contains NaN or Inf")
    return data, header


def prepare_motion(csv_path: Path, output_path: Path, input_fps: float, output_fps: float) -> None:
    raw, header = load_csv(csv_path)
    duration = (len(raw) - 1) / input_fps
    times = np.arange(0.0, duration, 1.0 / output_fps, dtype=np.float64)
    sample = times * input_fps
    i0 = np.floor(sample).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(raw) - 1)
    alpha = sample - i0

    root_pos = (1.0 - alpha[:, None]) * raw[i0, :3] + alpha[:, None] * raw[i1, :3]
    q_raw = make_quat_continuous(raw[:, 3:7])
    root_quat = slerp_pair(q_raw[i0], q_raw[i1], alpha)
    joint_pos = (1.0 - alpha[:, None]) * raw[i0, 7:] + alpha[:, None] * raw[i1, 7:]
    dt = 1.0 / output_fps
    root_lin_vel = differentiate(root_pos, dt)
    root_ang_vel = angular_velocity_world(root_quat, dt)
    joint_vel = differentiate(joint_pos, dt)
    metadata = {
        "source_csv": str(csv_path.resolve()),
        "source_frames": len(raw),
        "input_fps": input_fps,
        "output_fps": output_fps,
        "duration_seconds": duration,
        "quaternion_order": "wxyz",
        "velocity_frame": "world",
        "csv_columns": header,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        fps=np.asarray([output_fps], dtype=np.float64),
        time=times,
        root_pos=root_pos.astype(np.float32),
        root_quat=root_quat.astype(np.float32),
        root_lin_vel=root_lin_vel.astype(np.float32),
        root_ang_vel=root_ang_vel.astype(np.float32),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        joint_names=np.asarray(JOINT_NAMES),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def body_velocities_from_pose(pos: np.ndarray, quat: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    lin = differentiate(pos, 1.0 / fps)
    angular = np.empty_like(pos)
    for body in range(pos.shape[1]):
        angular[:, body] = angular_velocity_world(quat[:, body], 1.0 / fps)
    return lin, angular
