#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    motion = np.load(args.input)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    cfg = unitree_g1_flat_tracking_env_cfg().scene
    scene = Scene(cfg, device=device)
    model = scene.compile()
    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / float(motion["fps"][0])
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    scene.reset()
    robot: Entity = scene["robot"]
    canonical_joint_names = motion["joint_names"].tolist()
    joint_ids = robot.find_joints(canonical_joint_names, preserve_order=True)[0]
    if len(joint_ids) != len(canonical_joint_names):
        raise RuntimeError("Could not map all canonical joints")

    keys = (
        "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
        "body_lin_vel_w", "body_ang_vel_w", "body_link_lin_vel_w",
        "body_com_pos_w", "body_com_lin_vel_w",
    )
    log = {key: [] for key in keys}
    for frame in range(len(motion["joint_pos"])):
        root = robot.data.default_root_state.clone()
        root[:, :3] = torch.as_tensor(motion["root_pos"][frame], device=device)
        root[:, 3:7] = torch.as_tensor(motion["root_quat"][frame], device=device)
        root[:, 7:10] = torch.as_tensor(motion["root_lin_vel"][frame], device=device)
        root[:, 10:13] = torch.as_tensor(motion["root_ang_vel"][frame], device=device)
        robot.write_root_state_to_sim(root)
        q = robot.data.default_joint_pos.clone()
        qd = robot.data.default_joint_vel.clone()
        q[:, joint_ids] = torch.as_tensor(motion["joint_pos"][frame], device=device)
        qd[:, joint_ids] = torch.as_tensor(motion["joint_vel"][frame], device=device)
        robot.write_joint_state_to_sim(q, qd)
        sim.forward()
        scene.update(sim.mj_model.opt.timestep)
        values = (
            robot.data.joint_pos, robot.data.joint_vel,
            robot.data.body_link_pos_w, robot.data.body_link_quat_w,
            robot.data.body_link_lin_vel_w, robot.data.body_link_ang_vel_w,
            robot.data.body_link_lin_vel_w, robot.data.body_com_pos_w,
            robot.data.body_com_lin_vel_w,
        )
        for key, value in zip(keys, values):
            log[key].append(value[0].detach().cpu().numpy().copy())

    xml_path = Path(__file__).resolve().parents[2] / "UNICTL/mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
    metadata = {
        "backend": "mjlab/mujoco_warp", "device": device,
        "asset_path": str(xml_path),
        "asset_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest() if xml_path.exists() else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output, fps=motion["fps"],
        joint_names=np.asarray(robot.joint_names), body_names=np.asarray(robot.body_names),
        metadata_json=np.asarray(json.dumps(metadata)),
        **{key: np.stack(value).astype(np.float32) for key, value in log.items()},
    )
    print(f"Wrote {args.output} ({len(log['joint_pos'])} frames)")


if __name__ == "__main__":
    main()
