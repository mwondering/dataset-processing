"""Standalone pure-Torch FK for the mjlab Unitree G1 29-DoF asset.

The constants were exported from HEFT ``MotionFKHelper`` at commit b070dab
using mjlab's ``g1_29dof_rev_1_0`` XML.  Runtime dependencies are only Torch;
MuJoCo, mjlab, IsaacLab, and HEFT are not imported.
"""

from __future__ import annotations

import math

import torch

from fk_compare.heft_batch import normalize, quat_apply, quat_mul


ASSET_NAME = "mjlab_unitree_g1_29dof_rev_1_0"

_PARENT_LOCAL_IDX = (
    -1, 0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 15, 23, 24, 25, 26, 27, 28,
)

_BODY_POS0 = (
    (0.0, 0.0, 0.0),
    (0.0, 0.06445199996232986, -0.10270000249147415),
    (0.0, 0.052000001072883606, -0.03046499937772751),
    (0.025001000612974167, 0.0, -0.12411999702453613),
    (-0.0782729983329773, 0.0021488999482244253, -0.17734000086784363),
    (0.0, -9.444500028621405e-05, -0.30000999569892883),
    (0.0, 0.0, -0.01755799911916256),
    (0.0, -0.06445199996232986, -0.10270000249147415),
    (0.0, -0.052000001072883606, -0.03046499937772751),
    (0.025001000612974167, 0.0, -0.12411999702453613),
    (-0.0782729983329773, -0.0021488999482244253, -0.17734000086784363),
    (0.0, 9.444500028621405e-05, -0.30000999569892883),
    (0.0, 0.0, -0.01755799911916256),
    (0.0, 0.0, 0.0),
    (-0.003963499795645475, 0.0, 0.04399999976158142),
    (0.0, 0.0, 0.0),
    (0.00395630020648241, 0.10022000223398209, 0.2477799952030182),
    (0.0, 0.03799999877810478, -0.013830999843776226),
    (0.0, 0.006240000016987324, -0.10320000350475311),
    (0.015783000737428665, 0.0, -0.08051799982786179),
    (0.10000000149011612, 0.001887909951619804, -0.009999999776482582),
    (0.03799999877810478, 0.0, 0.0),
    (0.04600000008940697, 0.0, 0.0),
    (0.00395630020648241, -0.10021000355482101, 0.2477799952030182),
    (0.0, -0.03799999877810478, -0.013830999843776226),
    (0.0, -0.006240000016987324, -0.10320000350475311),
    (0.015783000737428665, 0.0, -0.08051799982786179),
    (0.10000000149011612, -0.001887909951619804, -0.009999999776482582),
    (0.03799999877810478, 0.0, 0.0),
    (0.04600000008940697, 0.0, 0.0),
)

_BODY_QUAT0 = (
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.9961786866188049, 0.0, -0.08733857423067093, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.9961786866188049, 0.0, 0.08733857423067093, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.9961786866188049, 0.0, -0.08733857423067093, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.9961786866188049, 0.0, 0.08733857423067093, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.9902641773223877, 0.13920103013515472, 1.387220254400745e-05, -9.868681809166446e-05),
    (0.9902682304382324, -0.1391720324754715, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.9902641773223877, -0.13920103013515472, 1.387220254400745e-05, 9.868681809166446e-05),
    (0.9902682304382324, 0.1391720324754715, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
)

_JOINT_AXIS_LOCAL = (
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
)

_JOINT_DATASET_IDX = (
    -1, 0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
)

_OUTPUT_LOCAL_IDX = (
    0, 1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 16, 23, 5, 11, 17, 24, 6, 12, 18, 25, 19, 26, 20, 27, 21, 28, 22, 29,
)

_DEPTH_GROUPS = (
    (1, 7, 13),
    (2, 8, 14),
    (3, 9, 15),
    (4, 10, 16, 23),
    (5, 11, 17, 24),
    (6, 12, 18, 25),
    (19, 26),
    (20, 27),
    (21, 28),
    (22, 29),
)


def _quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    half = angle * 0.5
    return torch.cat((torch.cos(half).unsqueeze(-1), axis * torch.sin(half).unsqueeze(-1)), dim=-1)


class G1PureTorchFK:
    """HEFT-compatible FK helper with a fixed, embedded mjlab G1 tree."""

    asset_name = ASSET_NAME

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self.parent = torch.tensor(_PARENT_LOCAL_IDX, dtype=torch.long, device=self.device)
        self.pos0 = torch.tensor(_BODY_POS0, dtype=torch.float32, device=self.device)
        self.quat0 = torch.tensor(_BODY_QUAT0, dtype=torch.float32, device=self.device)
        self.axis = torch.tensor(_JOINT_AXIS_LOCAL, dtype=torch.float32, device=self.device)
        self.joint_idx = torch.tensor(_JOINT_DATASET_IDX, dtype=torch.long, device=self.device)
        self.output_idx = torch.tensor(_OUTPUT_LOCAL_IDX, dtype=torch.long, device=self.device)
        self.depth_groups = tuple(
            torch.tensor(group, dtype=torch.long, device=self.device) for group in _DEPTH_GROUPS
        )

    def body_pose(self, joint_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if joint_pos.dtype != torch.float32:
            raise RuntimeError("G1 FK expects float32 joint positions")
        if joint_pos.shape[-1] != 29:
            raise ValueError(f"Expected 29 G1 joints, got {joint_pos.shape[-1]}")
        if joint_pos.device != self.device:
            raise ValueError(f"FK is on {self.device}, input is on {joint_pos.device}")

        prefix = joint_pos.shape[:-1]
        flat_count = math.prod(prefix) if prefix else 1
        joint_pos_f = joint_pos.reshape(flat_count, 29)
        tree_pos = torch.zeros((flat_count, 30, 3), dtype=torch.float32, device=self.device)
        tree_quat = torch.zeros((flat_count, 30, 4), dtype=torch.float32, device=self.device)
        tree_quat[:, 0, 0] = 1.0

        for local_idx in self.depth_groups:
            parent_idx = self.parent.index_select(0, local_idx)
            parent_pos = tree_pos.index_select(1, parent_idx)
            parent_quat = tree_quat.index_select(1, parent_idx)
            quat0 = self.quat0.index_select(0, local_idx).unsqueeze(0)
            pos0 = self.pos0.index_select(0, local_idx).unsqueeze(0)
            axis = self.axis.index_select(0, local_idx).unsqueeze(0)
            dataset_idx = self.joint_idx.index_select(0, local_idx)
            angle = joint_pos_f.index_select(1, dataset_idx)
            rel_quat = quat_mul(quat0, _quat_from_angle_axis(angle, axis))
            tree_quat.index_copy_(1, local_idx, normalize(quat_mul(parent_quat, rel_quat)))
            tree_pos.index_copy_(1, local_idx, parent_pos + quat_apply(parent_quat, pos0))

        body_pos = tree_pos.index_select(1, self.output_idx).reshape(prefix + (30, 3))
        body_quat = tree_quat.index_select(1, self.output_idx).reshape(prefix + (30, 4))
        return body_pos, body_quat
