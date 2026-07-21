#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fk_compare.motion import normalize_quat


def reindex(names: np.ndarray, target: list[str]) -> np.ndarray:
    lookup = {str(name): i for i, name in enumerate(names.tolist())}
    missing = [name for name in target if name not in lookup]
    if missing:
        raise ValueError(f"Missing names: {missing}")
    return np.asarray([lookup[name] for name in target])


def vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=-1)


def scalar(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.float64) - b.astype(np.float64))


def quat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = normalize_quat(a.astype(np.float64)), normalize_quat(b.astype(np.float64))
    return 2.0 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0))


def stats(e: np.ndarray) -> dict[str, float | int]:
    flat = e.ravel()
    index = np.unravel_index(int(np.argmax(flat)), e.shape)
    return {
        "mean": float(flat.mean()), "rmse": float(np.sqrt(np.mean(flat**2))),
        "p95": float(np.percentile(flat, 95)), "max": float(flat.max()),
        "max_frame": int(index[0]), "max_item": int(index[1]) if len(index) > 1 else 0,
    }


def avg5(x: np.ndarray) -> np.ndarray:
    padding = [(2, 2)] + [(0, 0)] * (x.ndim - 1)
    padded = np.pad(x, padding, mode="edge")
    return sum(padded[i : i + len(x)] for i in range(5)) / 5.0


def pair_errors(heft, other, body_lin_key: str) -> dict[str, np.ndarray]:
    joints = [str(x) for x in heft["joint_names"]]
    bodies = [str(x) for x in heft["body_names"]]
    ji, bi = reindex(other["joint_names"], joints), reindex(other["body_names"], bodies)
    other_joint_vel = other["joint_vel"][:, ji]
    other_body_lin_vel = other[body_lin_key][:, bi]
    other_body_ang_vel = other["body_ang_vel_w"][:, bi]
    return {
        "joint_pos_rad": scalar(heft["joint_pos"], other["joint_pos"][:, ji]),
        "joint_vel_rad_s": scalar(heft["joint_vel"], other_joint_vel),
        "joint_vel_vs_avg5_engine_rad_s": scalar(heft["joint_vel"], avg5(other_joint_vel)),
        "body_pos_m": vec(heft["body_pos_w"], other["body_pos_w"][:, bi]),
        "body_quat_rad": quat(heft["body_quat_w"], other["body_quat_w"][:, bi]),
        "body_link_lin_vel_m_s": vec(heft["body_lin_vel_w"], other_body_lin_vel),
        "body_link_lin_vel_vs_avg5_engine_m_s": vec(heft["body_lin_vel_w"], avg5(other_body_lin_vel)),
        "body_ang_vel_rad_s": vec(heft["body_ang_vel_w"], other_body_ang_vel),
        "body_ang_vel_vs_avg5_engine_rad_s": vec(heft["body_ang_vel_w"], avg5(other_body_ang_vel)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heft", type=Path, required=True)
    parser.add_argument("--isaac", type=Path, required=True)
    parser.add_argument("--mjwarp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    heft, isaac, mj = np.load(args.heft), np.load(args.isaac), np.load(args.mjwarp)
    errors = {
        "heft_vs_isaac_aligned": pair_errors(heft, isaac, "body_link_lin_vel_w"),
        "heft_vs_mjwarp": pair_errors(heft, mj, "body_link_lin_vel_w"),
    }
    summary = {
        "frames": len(heft["joint_pos"]), "fps": float(heft["fps"][0]),
        "heft_metadata": json.loads(str(heft["metadata_json"])),
        "pairs": {pair: {term: stats(value) for term, value in terms.items()} for pair, terms in errors.items()},
    }
    for pair in summary["pairs"].values():
        q = pair["body_quat_rad"]
        pair["body_quat_deg"] = {k: v * 180 / np.pi if k not in ("max_frame", "max_item") else v for k, v in q.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    joint_names, body_names = heft["joint_names"].tolist(), heft["body_names"].tolist()
    with (args.output_dir / "per_item.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("pair", "term", "name", "mean", "rmse", "p95", "max", "max_frame"))
        for pair, terms in errors.items():
            for term, error in terms.items():
                names = joint_names if term.startswith("joint_") else body_names
                for i, name in enumerate(names):
                    item = stats(error[:, i])
                    writer.writerow((pair, term, name, item["mean"], item["rmse"], item["p95"], item["max"], item["max_frame"]))
    rows = []
    for pair, terms in summary["pairs"].items():
        for term, item in terms.items():
            rows.append(f"<tr><td>{pair}</td><td>{term}</td><td>{item['mean']:.8g}</td><td>{item['rmse']:.8g}</td><td>{item['p95']:.8g}</td><td>{item['max']:.8g}</td></tr>")
    html = """<!doctype html><meta charset='utf-8'><title>HEFT FK baseline</title>
<style>body{font-family:sans-serif;max-width:1200px;margin:2rem auto}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.4rem;text-align:right}td:first-child,td:nth-child(2){text-align:left}</style>
<h1>HEFT pure-Torch FK baseline</h1><p>{frames} frames at {fps:g} FPS.</p>
<table><tr><th>pair</th><th>term</th><th>mean</th><th>RMSE</th><th>P95</th><th>max</th></tr>{rows}</table>""".replace("{frames}", str(summary["frames"])).replace("{fps:g}", f"{summary['fps']:g}").replace("{rows}", "".join(rows))
    (args.output_dir / "report.html").write_text(html)
    print(json.dumps(summary["pairs"], indent=2))


if __name__ == "__main__":
    main()
