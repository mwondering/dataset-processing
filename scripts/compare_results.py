#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fk_compare.motion import body_velocities_from_pose, normalize_quat


def name_reindex(source: np.ndarray, target: list[str], kind: str) -> np.ndarray:
    lookup = {str(name): i for i, name in enumerate(source.tolist())}
    missing = [name for name in target if name not in lookup]
    if missing:
        raise ValueError(f"Missing {kind} names: {missing}")
    return np.asarray([lookup[name] for name in target])


def vector_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=-1)


def quaternion_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize_quat(a.astype(np.float64))
    b = normalize_quat(b.astype(np.float64))
    dot = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def scalar_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.float64) - b.astype(np.float64))


def stats(error: np.ndarray) -> dict[str, float | int]:
    flat = error.reshape(-1)
    argmax = int(np.argmax(flat))
    index = np.unravel_index(argmax, error.shape)
    return {
        "mean": float(np.mean(flat)), "rmse": float(np.sqrt(np.mean(flat**2))),
        "p95": float(np.percentile(flat, 95)), "p99": float(np.percentile(flat, 99)),
        "max": float(flat[argmax]), "max_frame": int(index[0]),
        "max_item": int(index[1]) if len(index) > 1 else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac", type=Path, required=True)
    parser.add_argument("--mjwarp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    isaac = np.load(args.isaac)
    mj = np.load(args.mjwarp)
    if len(isaac["joint_pos"]) != len(mj["joint_pos"]):
        raise ValueError("Frame counts differ")

    joint_names = [str(x) for x in isaac["joint_names"]]
    body_names = [str(x) for x in isaac["body_names"]]
    mj_joint = name_reindex(mj["joint_names"], joint_names, "joint")
    mj_body = name_reindex(mj["body_names"], body_names, "body")
    errors = {
        "joint_pos_rad": scalar_error(isaac["joint_pos"], mj["joint_pos"][:, mj_joint]),
        "joint_vel_rad_s": scalar_error(isaac["joint_vel"], mj["joint_vel"][:, mj_joint]),
        "body_pos_m": vector_error(isaac["body_pos_w"], mj["body_pos_w"][:, mj_body]),
        "body_quat_rad": quaternion_error(isaac["body_quat_w"], mj["body_quat_w"][:, mj_body]),
        "body_lin_vel_m_s": vector_error(isaac["body_lin_vel_w"], mj["body_lin_vel_w"][:, mj_body]),
        "body_ang_vel_rad_s": vector_error(isaac["body_ang_vel_w"], mj["body_ang_vel_w"][:, mj_body]),
    }
    semantic_errors = {}
    explicit = ("body_link_lin_vel_w", "body_com_pos_w", "body_com_lin_vel_w")
    if all(key in isaac.files and key in mj.files for key in explicit):
        semantic_errors = {
            "body_link_lin_vel_link_to_link_m_s": vector_error(
                isaac["body_link_lin_vel_w"], mj["body_link_lin_vel_w"][:, mj_body]
            ),
            "body_com_pos_com_to_com_m": vector_error(
                isaac["body_com_pos_w"], mj["body_com_pos_w"][:, mj_body]
            ),
            "body_com_lin_vel_com_to_com_m_s": vector_error(
                isaac["body_com_lin_vel_w"], mj["body_com_lin_vel_w"][:, mj_body]
            ),
        }
    fps = float(isaac["fps"][0])
    isaac_fd_lin, isaac_fd_ang = body_velocities_from_pose(isaac["body_pos_w"], isaac["body_quat_w"], fps)
    mj_fd_lin, mj_fd_ang = body_velocities_from_pose(mj["body_pos_w"][:, mj_body], mj["body_quat_w"][:, mj_body], fps)
    diagnostics = {
        "isaac_engine_vs_pose_fd_lin_m_s": vector_error(isaac["body_lin_vel_w"], isaac_fd_lin),
        "isaac_engine_vs_pose_fd_ang_rad_s": vector_error(isaac["body_ang_vel_w"], isaac_fd_ang),
        "mjwarp_engine_vs_pose_fd_lin_m_s": vector_error(mj["body_lin_vel_w"][:, mj_body], mj_fd_lin),
        "mjwarp_engine_vs_pose_fd_ang_rad_s": vector_error(mj["body_ang_vel_w"][:, mj_body], mj_fd_ang),
        "pose_fd_backend_lin_m_s": vector_error(isaac_fd_lin, mj_fd_lin),
        "pose_fd_backend_ang_rad_s": vector_error(isaac_fd_ang, mj_fd_ang),
    }
    summary = {
        "fps": fps, "frames": len(isaac["joint_pos"]),
        "isaac_metadata": json.loads(str(isaac["metadata_json"])),
        "mjwarp_metadata": json.loads(str(mj["metadata_json"])),
        "backend_errors": {key: stats(value) for key, value in errors.items()},
        "semantic_aligned_errors": {key: stats(value) for key, value in semantic_errors.items()},
        "velocity_diagnostics": {key: stats(value) for key, value in diagnostics.items()},
    }
    summary["backend_errors"]["body_quat_deg"] = {
        key: value * 180.0 / np.pi if key not in ("max_frame", "max_item") else value
        for key, value in summary["backend_errors"]["body_quat_rad"].items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    with (args.output_dir / "per_item.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("term", "name", "mean", "rmse", "p95", "max", "max_frame"))
        for term, error in {**errors, **semantic_errors, **diagnostics}.items():
            names = joint_names if term.startswith("joint_") else body_names
            for index, name in enumerate(names):
                item = stats(error[:, index])
                writer.writerow((term, name, item["mean"], item["rmse"], item["p95"], item["max"], item["max_frame"]))

    rows = []
    for term, item in summary["backend_errors"].items():
        rows.append(f"<tr><td>{term}</td><td>{item['mean']:.8g}</td><td>{item['rmse']:.8g}</td><td>{item['p95']:.8g}</td><td>{item['max']:.8g}</td><td>{item['max_frame']}</td><td>{body_names[item['max_item']] if term.startswith('body_') else joint_names[item['max_item']]}</td></tr>")
    semantic_rows = []
    for term, item in summary["semantic_aligned_errors"].items():
        semantic_rows.append(f"<tr><td>{term}</td><td>{item['mean']:.8g}</td><td>{item['rmse']:.8g}</td><td>{item['p95']:.8g}</td><td>{item['max']:.8g}</td><td>{item['max_frame']}</td><td>{body_names[item['max_item']]}</td></tr>")
    html = """<!doctype html><meta charset='utf-8'><title>FK backend comparison</title>
<style>body{{font-family:sans-serif;max-width:1200px;margin:2rem auto}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:.4rem;text-align:right}}td:first-child,td:last-child{{text-align:left}}</style>
<h1>IsaacLab vs mjlab/mujoco_warp</h1><p>{frames} frames at {fps:g} FPS.</p>
<h2>Legacy NPZ term names</h2>
<table><tr><th>term</th><th>mean</th><th>RMSE</th><th>P95</th><th>max</th><th>frame</th><th>item</th></tr>{rows}</table>
<h2>Explicitly aligned frame semantics</h2>
<table><tr><th>term</th><th>mean</th><th>RMSE</th><th>P95</th><th>max</th><th>frame</th><th>item</th></tr>{semantic_rows}</table>
<p>See <code>summary.json</code> for finite-difference velocity diagnostics and <code>per_item.csv</code> for every joint/body.</p>""".format(
        frames=summary["frames"], fps=fps, rows="".join(rows), semantic_rows="".join(semantic_rows)
    )
    (args.output_dir / "report.html").write_text(html)
    print(json.dumps(summary["backend_errors"], indent=2))
    print(f"Wrote report to {args.output_dir}")


if __name__ == "__main__":
    main()
