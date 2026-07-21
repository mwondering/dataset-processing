#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from active_adaptation.utils.fk_helper import MotionFKHelper
from active_adaptation.utils.motion import MotionMinimalData


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HEFT's native pure-Torch FK baseline.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heft-root", type=Path, default=Path("/home/lenovo/workspace/UNICTL/heft"))
    parser.add_argument("--xml", type=Path, help="Optional alternate MJCF; the HEFT FK algorithm is unchanged.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    source = np.load(args.input, allow_pickle=False)
    joint_names = [str(x) for x in source["joint_names"]]
    # Use the same 30-body contract as the IsaacLab/mjlab outputs, in mjlab order.
    body_names = [
        "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
        "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
        "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
        "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
        "waist_yaw_link", "waist_roll_link", "torso_link",
        "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
        "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
        "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
        "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
    ]
    xml_path = args.xml or (args.heft_root / "active_adaptation/assets/G1/g1.xml")
    helper = MotionFKHelper.from_mjcf_path(
        xml_path=xml_path,
        dataset_joint_names=joint_names,
        output_body_names=body_names,
        base_body_name="pelvis",
        device=args.device,
    )
    minimal = MotionMinimalData(
        root_pos_w=torch.as_tensor(source["root_pos"], dtype=torch.float32, device=args.device).unsqueeze(0),
        root_quat_w=torch.as_tensor(source["root_quat"], dtype=torch.float32, device=args.device).unsqueeze(0),
        joint_pos=torch.as_tensor(source["joint_pos"], dtype=torch.float32, device=args.device).unsqueeze(0),
    )
    with torch.inference_mode():
        motion = helper.expand_minimal_motion(minimal, fps=float(source["fps"][0]))

    def array(value: torch.Tensor) -> np.ndarray:
        return value[0].detach().cpu().numpy().astype(np.float32, copy=False)

    metadata = {
        "backend": "HEFT MotionFKHelper",
        "device": args.device,
        "asset_path": str(xml_path),
        "asset_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
        "pose_method": "pure_torch_recursive_tree_fk",
        "velocity_method": "central_finite_difference_then_replicate_padded_avg5",
        "body_velocity_semantics": "link_origin_world",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        fps=source["fps"],
        joint_names=np.asarray(joint_names),
        body_names=np.asarray(body_names),
        metadata_json=np.asarray(json.dumps(metadata)),
        joint_pos=array(motion.joint_pos),
        joint_vel=array(motion.joint_vel),
        body_pos_w=array(motion.body_pos_w),
        body_quat_w=array(motion.body_quat_w),
        body_lin_vel_w=array(motion.body_vel_w),
        body_ang_vel_w=array(motion.body_angvel_w),
    )
    print(f"Wrote {args.output} ({motion.joint_pos.shape[1]} frames)")


if __name__ == "__main__":
    main()
