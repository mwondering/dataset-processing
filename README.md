# G1 FK backend comparison

This project feeds one canonical G1 motion state into IsaacLab and
mjlab/mujoco_warp, then compares the seven terms used by tracking motion NPZ files.

The Take 045 CSV is headered and stores root quaternions as `wxyz`. Its 3390 frames
are interpreted as 120 FPS and resampled to 50 FPS. Both backends consume the same
precomputed positions and velocities; neither backend performs its own interpolation.

## Reproduce

```bash
cd /home/lenovo/workspace/fk_backend_compare
python scripts/prepare_motion.py \
  --input /home/lenovo/DATASETS/LAFAN1_Retargeting_Dataset/g1/20260104_mocap/D0106_XY_JinJiDuLi_2_2_4_001_SK00_045/Retargeting/Take_045_Skeleton0_main.csv \
  --input-fps 120 --output-fps 50 --output outputs/take045/canonical_input.npz

cd /home/lenovo/workspace/UNICTL/mjlab
uv run python /home/lenovo/workspace/fk_backend_compare/scripts/export_mjwarp.py \
  --input /home/lenovo/workspace/fk_backend_compare/outputs/take045/canonical_input.npz \
  --output /home/lenovo/workspace/fk_backend_compare/outputs/take045/mjwarp_fk.npz --device cpu

cd /home/lenovo/workspace/BeyondMimic_sjy
conda run -n bydmimic python /home/lenovo/workspace/fk_backend_compare/scripts/export_isaaclab.py \
  --input /home/lenovo/workspace/fk_backend_compare/outputs/take045/canonical_input.npz \
  --output /home/lenovo/workspace/fk_backend_compare/outputs/take045/isaaclab_fk_aligned.npz \
  --root-velocity-semantics link --headless

cd /home/lenovo/workspace/fk_backend_compare
python scripts/compare_results.py --isaac outputs/take045/isaaclab_fk_aligned.npz \
  --mjwarp outputs/take045/mjwarp_fk.npz --output-dir outputs/take045/report_aligned
```

The comparison aligns joints and bodies by name. Quaternion error is sign-invariant
geodesic distance. Body velocities are also independently reconstructed from each
backend's pose sequence to separate FK differences from velocity convention differences.

All exporters use exactly one environment. The mjlab command above runs on CPU; the
IsaacLab command is headless and performs no physics stepping or rendering.

To reproduce the original BeyondMimic root-state write contract, replace the Isaac
output name and pass `--root-velocity-semantics legacy_com`.

See [RESULTS.md](RESULTS.md) for the measured Take 045 result.

## HEFT baseline

```bash
cd /home/lenovo/workspace/UNICTL/heft
.venv/bin/python /home/lenovo/workspace/fk_backend_compare/scripts/export_heft.py \
  --input /home/lenovo/workspace/fk_backend_compare/outputs/take045/canonical_input.npz \
  --output /home/lenovo/workspace/fk_backend_compare/outputs/take045/heft_fk.npz --device cpu

cd /home/lenovo/workspace/fk_backend_compare
python scripts/compare_heft.py --heft outputs/take045/heft_fk.npz \
  --isaac outputs/take045/isaaclab_fk_aligned.npz --mjwarp outputs/take045/mjwarp_fk.npz \
  --output-dir outputs/take045/report_heft_native
```

To isolate HEFT's algorithm from its native XML, pass mjlab's G1 XML through
`export_heft.py --xml .../mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml`.
The detailed computation routes are documented in [ROUTES.md](ROUTES.md).

## GPU batch rebuild from HEFT-native 36D poses

`scripts/process_isaaclab_pos36.py` rebuilds an IsaacLab-style Data10k tree
with HEFT's pure-Torch FK and velocity smoothing.  For every source
`motion.npz`, the FK stage consumes only this float32 36D tensor:

```text
root position [3]
+ root quaternion wxyz [4]
+ joint position in IsaacLab G1 order [29]
```

This is HEFT's native minimal-motion representation.  There is no float16
quantization and no quaternion/rotation-6D round trip.  Source `body_quat_w`
values are read and written as `wxyz` quaternions.  The target NPZ contains
exactly the original seven Data10k fields and preserves the IsaacLab
joint/body order.

Run one motion on CPU for validation:

```bash
/home/lenovo/workspace/UNICTL/heft/.venv/bin/python \
  scripts/process_isaaclab_pos36.py \
  --input /home/lenovo/DATASETS/Data10k/dance1_subject2_0_3945/motion.npz \
  --output-dir outputs/data10k_heft_smoothed \
  --device cpu
```

The production dataset path has only two runtime dependencies.  It does not
import or install IsaacLab, mjlab, MuJoCo, HEFT, Warp, or NCCL distributed
communication:

```bash
python -m venv .venv-cluster
source .venv-cluster/bin/activate
python -m pip install -r requirements-cluster.txt
```

Process a complete dataset on one GPU:

```bash
python \
  scripts/process_isaaclab_pos36.py \
  --input /path/to/Data10k \
  --output-dir /path/to/Data10k_heft_smoothed \
  --device cuda:0 --batch-frames 262144 --batch-motions 32 --io-workers 8 \
  --skip-existing
```

Process one dataset on every visible GPU, or select local GPU IDs explicitly:

```bash
python scripts/process_dataset_multigpu.py \
  --input /path/to/Data10k \
  --output-dir /path/to/Data10k_heft_smoothed \
  --gpus 0,1,2,3 \
  --batch-frames 262144 \
  --batch-motions 32 \
  --io-workers-per-gpu 4 \
  --scan-workers 16 \
  --skip-existing
```

`--batch-frames` limits padded frames resident in one GPU batch; reduce it if
CUDA memory is insufficient.  Variable-length motions are padded only for FK,
and the length-aware difference/filter functions do not cross sequence
boundaries.  Motions are sorted by length to reduce padding, and CPU workers
prefetch the next NPZ batch while the GPU processes the current batch.  The
multi-GPU launcher scans the dataset once, balances motions by frame count,
writes one manifest and log per GPU under `_cluster/`, and merges worker
statistics into the root `summary.json`.  Each output file is written
atomically, so `--skip-existing` safely resumes an interrupted job.  Use
`--dry-run` to inspect the sharding plan without starting workers.

Each rebuilt `motion.npz` has a `motion.diff.json` sidecar containing exact
per-file statistics for:

- original IsaacLab terms versus rebuilt terms;
- raw finite-difference velocities versus HEFT's replicate-padded five-point
  smoothed velocities.

The output key `body_lin_vel_w` keeps the Data10k filename contract but stores
the world velocity of the **link origin**, matching HEFT and mjlab.  It is not
IsaacLab 2.2's legacy COM-velocity alias.  This semantic change is also written
to every difference report.
