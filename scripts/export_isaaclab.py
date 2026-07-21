#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--beyond-mimic", type=Path, default=Path("/home/lenovo/workspace/BeyondMimic_sjy"))
parser.add_argument(
    "--root-velocity-semantics", choices=("link", "legacy_com"), default="link",
    help="Use explicit link-origin velocity, or reproduce BeyondMimic's legacy root-state write.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

sys.path.insert(0, str(args.beyond_mimic / "source/whole_body_tracking"))
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run() -> None:
    motion = np.load(args.input)
    sim_cfg = sim_utils.SimulationCfg(device=args.device, dt=1.0 / float(motion["fps"][0]))
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]
    canonical_joint_names = motion["joint_names"].tolist()
    joint_ids = robot.find_joints(canonical_joint_names, preserve_order=True)[0]
    keys = (
        "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
        "body_lin_vel_w", "body_ang_vel_w", "body_link_lin_vel_w",
        "body_com_pos_w", "body_com_lin_vel_w",
    )
    log = {key: [] for key in keys}
    for frame in range(len(motion["joint_pos"])):
        root = robot.data.default_root_state.clone()
        root[:, :3] = torch.as_tensor(motion["root_pos"][frame], device=sim.device)
        root[:, 3:7] = torch.as_tensor(motion["root_quat"][frame], device=sim.device)
        root[:, 7:10] = torch.as_tensor(motion["root_lin_vel"][frame], device=sim.device)
        root[:, 10:13] = torch.as_tensor(motion["root_ang_vel"][frame], device=sim.device)
        # Canonical root position and its derivative describe the link origin.
        # Avoid write_root_state_to_sim: in IsaacLab 2.2 its velocity portion is
        # COM velocity, unlike mjlab's link-origin root-state contract.
        if args.root_velocity_semantics == "link":
            robot.write_root_link_pose_to_sim(root[:, :7])
            robot.write_root_link_velocity_to_sim(root[:, 7:13])
        else:
            robot.write_root_state_to_sim(root)
        q = robot.data.default_joint_pos.clone()
        qd = robot.data.default_joint_vel.clone()
        q[:, joint_ids] = torch.as_tensor(motion["joint_pos"][frame], device=sim.device)
        qd[:, joint_ids] = torch.as_tensor(motion["joint_vel"][frame], device=sim.device)
        robot.write_joint_state_to_sim(q, qd)
        sim.render()
        scene.update(sim.get_physics_dt())
        # IsaacLab's legacy body_lin_vel_w alias is COM velocity, while body_pos_w
        # is link-frame position. Log both explicit semantics for diagnosis.
        values = (
            robot.data.joint_pos, robot.data.joint_vel, robot.data.body_pos_w,
            robot.data.body_quat_w, robot.data.body_lin_vel_w, robot.data.body_ang_vel_w,
            robot.data.body_link_lin_vel_w, robot.data.body_com_pos_w,
            robot.data.body_com_lin_vel_w,
        )
        for key, value in zip(keys, values):
            log[key].append(value[0].detach().cpu().numpy().copy())

    urdf = args.beyond_mimic / "source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf"
    metadata = {"backend": "IsaacLab", "device": str(sim.device),
                "root_velocity_semantics": args.root_velocity_semantics, "asset_path": str(urdf),
                "asset_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, fps=motion["fps"], joint_names=np.asarray(robot.joint_names),
             body_names=np.asarray(robot.body_names), metadata_json=np.asarray(json.dumps(metadata)),
             **{key: np.stack(value).astype(np.float32) for key, value in log.items()})
    print(f"Wrote {args.output} ({len(log['joint_pos'])} frames)")


try:
    run()
finally:
    simulation_app.close()
