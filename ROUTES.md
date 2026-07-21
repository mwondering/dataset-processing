# 三条 FK 路线的计算逻辑

## 共同输入

三条路线都使用 `canonical_input.npz` 中完全相同的 50 Hz、float32 root pose 和
29 维 joint position。原始 CSV 的四元数按表头直接解释为 `wxyz`，120 Hz 输入通过
位置线性插值和四元数 SLERP 重采样到 50 Hz。

canonical velocity 使用未平滑差分：位置和关节为中心差分、端点单边差分；root
角速度通过跨两帧相对旋转的 log map 计算。IsaacLab 和 mjwarp 接收这些预先计算的
广义速度，HEFT 则忽略它们并从 pose sequence 重新计算速度。

## IsaacLab / PhysX

1. 将 root pose、root velocity、joint position 和 joint velocity写入 PhysX articulation。
2. 触发 articulation kinematic update。
3. link pose 来自 PhysX link transform。
4. body angular velocity和 COM linear velocity来自 PhysX link velocity缓存。
5. link-origin linear velocity由 COM velocity和平移关系转换：
   `v_link = v_com + omega × (p_link - p_com)`。

IsaacLab 2.2 的兼容别名存在两个易混点：

- `body_pos_w` 是 link-origin position，但 `body_lin_vel_w` 是 COM velocity。
- `write_root_state_to_sim` 中的 velocity 是 root COM velocity。

本项目的 aligned 路线改用显式 `write_root_link_pose_to_sim`、
`write_root_link_velocity_to_sim` 和 `body_link_lin_vel_w`。

## mjlab / mujoco_warp

1. root state 和 joint state写入 MuJoCo `qpos/qvel`。
2. `sim.forward()` 调用 mujoco_warp forward，完成树状 FK、速度传播和其他派生量。
3. link pose 读取 `xpos/xquat`。
4. angular velocity读取 MuJoCo spatial `cvel`。
5. mjlab 根据 spatial velocity 的参考点，将线速度换算到各 link origin。

mjlab 的 `write_root_state_to_sim` 将 root linear velocity解释为 link-origin velocity；
其导出脚本保存的是 `body_link_lin_vel_w`。因此它与 IsaacLab 旧兼容别名并不同义，
但与本项目的 Isaac aligned 路线同义。

## HEFT / MotionFKHelper

HEFT 不执行 MuJoCo forward。它只用 MuJoCo 加载一次 XML，然后提取：

- `body_parentid`、静态 `body_pos/body_quat`
- joint type、joint anchor、joint axis
- dataset joint name 到树节点的映射

随后按树深分组，用纯 Torch 批量递归：

- hinge rotation：`q_rel = q0 * axis_angle(axis, joint_pos)`
- hinge translation：考虑非零 joint anchor 的旋转补偿
- world pose：先得到 pelvis frame 下的 body pose，再与 root pose 复合

HEFT 的速度不是引擎雅可比传播值，而是整段序列差分：

- joint/root position：中心差分，端点单边差分
- quaternion：先做符号连续化，再由 `omega = 2 * vec(q_dot * conjugate(q))` 求角速度
- 每个速度 term 最后使用 replicate padding 的 5 点均值滤波
- body velocity先在 root frame 中计算
  `d(body_pos_b)/dt + root_ang_vel_b × body_pos_b`，平滑后再旋转到 world 并加 root velocity
- body angular velocity同样先计算相对 root 的局部角速度，平滑后再与 root angular velocity合成

因此 HEFT 的 body velocity 不等于简单地对引擎 world velocity做一次 5 点均值。滤波、
随时间变化的坐标旋转和 root/body 分解并不交换。

## 资产差异

HEFT 原生 `g1.xml` 有 35 个 robot bodies，包括 toe、head、hand mimic fixed bodies；
本项目只输出三方共有的 30 个 bodies。它还将左右 `wrist_yaw_link` 的局部 x offset
设为 `0.051 m`，而 mjlab/Isaac 对应资产为 `0.046 m`，所以原生 HEFT 在两个
`wrist_yaw_link` 上稳定出现 5 mm position difference。

`heft_shared_asset_fk.npz` 保持 HEFT 计算逻辑不变，但改用 mjlab XML，用于单独验证
算法差异；此时 HEFT pose 与 mjwarp pose 的平均位置误差为 `9.59e-8 m`。
