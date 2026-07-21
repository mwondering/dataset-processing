# Take 045 result

Input: `Take_045_Skeleton0_main.csv`, 3390 frames at 120 FPS, resampled to 1413
frames at 50 FPS. The header declares root quaternions as `wxyz`.

## Existing pipeline semantics

The BeyondMimic exporter reads IsaacLab's legacy `body_lin_vel_w`, which is an alias
of COM linear velocity. The mjlab exporter writes `body_link_lin_vel_w`, which is
the velocity of the link origin. In addition, IsaacLab 2.2 interprets the velocity
portion of `write_root_state_to_sim` as root COM velocity, while mjlab interprets it
as root link-origin velocity.

Measured IsaacLab legacy vs current mjlab NPZ terms:

| term | mean | RMSE | P95 | max |
|---|---:|---:|---:|---:|
| joint position (rad) | 0 | 0 | 0 | 0 |
| joint velocity (rad/s) | 0 | 0 | 0 | 0 |
| body position (m) | 2.30e-7 | 2.81e-7 | 5.46e-7 | 1.11e-6 |
| body orientation (deg) | 4.87e-5 | 5.81e-5 | 1.06e-4 | 1.62e-4 |
| legacy body linear velocity (m/s) | 3.239e-2 | 5.997e-2 | 1.184e-1 | 8.050e-1 |
| body angular velocity (rad/s) | 5.35e-7 | 1.49e-6 | 2.03e-6 | 3.42e-5 |

The maximum legacy linear-velocity mismatch is on `left_hip_yaw_link`, frame 949
(18.98 s).

## Same semantic quantities

After writing canonical root velocity explicitly as root link-origin velocity and
comparing like with like:

| comparison | mean | RMSE | P95 | max |
|---|---:|---:|---:|---:|
| link linear velocity ↔ link linear velocity (m/s) | 1.51e-7 | 4.09e-7 | 7.05e-7 | 1.15e-5 |
| COM linear velocity ↔ COM linear velocity (m/s) | 4.29e-7 | 1.31e-6 | 1.99e-6 | 1.34e-5 |
| COM position ↔ COM position (m) | 1.88e-6 | 6.46e-6 | 1.99e-5 | 2.94e-5 |

Conclusion: for this motion the two backends' FK and velocity propagation agree to
approximately float32 precision once root velocity and link/COM semantics are aligned.
The large `body_lin_vel_w` discrepancy comes from API/data-contract mismatches, not
from materially different forward kinematics.

## Runtime versions

- BeyondMimic commit `a159e4f`, IsaacLab 2.2.1 checkout, Python package `isaaclab 0.45.9`
- mjlab commit `6c0b6bf0`, `mjlab 1.5.0`, MuJoCo `3.10.0`, mujoco_warp `3.10.0.1`
- Both exporters used `num_envs=1`; mjlab ran on CPU and IsaacLab ran headless.

## HEFT baseline

HEFT commit `b070dab` was run on CPU using its native `MotionFKHelper`. It performs
pure-Torch recursive FK and recomputes all velocities with central finite differences
followed by a replicate-padded 5-point moving average.

HEFT native asset vs mjwarp:

| term | mean | RMSE | P95 | max |
|---|---:|---:|---:|---:|
| body position (m) | 3.33e-4 | 1.29e-3 | 5.00e-3 | 5.00e-3 |
| body orientation (deg) | 8.34e-6 | 1.02e-5 | 1.95e-5 | 4.66e-5 |
| joint velocity (rad/s) | 5.48e-2 | 3.29e-1 | 1.39e-1 | 20.72 |
| link linear velocity (m/s) | 8.60e-3 | 1.68e-2 | 2.30e-2 | 0.508 |
| body angular velocity (rad/s) | 7.18e-2 | 3.30e-1 | 1.90e-1 | 20.89 |

The position error is asset-driven: HEFT uses `0.051 m` wrist-yaw offsets while the
mjlab asset uses `0.046 m`. With the same mjlab XML, HEFT pure-Torch FK vs mjwarp is:

| term | mean | RMSE | P95 | max |
|---|---:|---:|---:|---:|
| body position (m) | 9.59e-8 | 1.24e-7 | 2.38e-7 | 4.95e-7 |
| body orientation (deg) | 8.34e-6 | 1.02e-5 | 1.95e-5 | 4.66e-5 |

HEFT joint velocity matches a 5-point average of the engine joint velocity to
`1.03e-7 rad/s` mean error. Body velocity retains larger differences after the same
simple world-frame average because HEFT differentiates and smooths root-relative pose
before composing it into the world frame. See [ROUTES.md](ROUTES.md) for the formulas.
