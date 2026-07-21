#!/usr/bin/env python3
"""Rebuild IsaacLab-style Data10k motions from HEFT-native 36D poses.

For an IsaacLab ``motion.npz``, the processor first keeps only
``pelvis position (3) + pelvis quaternion wxyz (4) + joint position (29)``.
It then runs an embedded, HEFT-validated pure-Torch G1 FK and HEFT-compatible
temporal smoothing in padded GPU batches.  Output NPZ files contain exactly the
seven fields used by the original Data10k loader; detailed before/after metrics
are sidecar JSON.

The legacy ``body_lin_vel_w`` output key intentionally contains link-origin
world velocity so that its values follow HEFT/mjlab semantics.  It is not the
IsaacLab 2.2 compatibility alias for COM velocity.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fk_compare.heft_batch import (  # noqa: E402
    DATA10K_TERMS,
    ISAACLAB_G1_BODY_NAMES,
    ISAACLAB_G1_JOINT_NAMES,
    TERM_UNITS,
    compare_terms,
    expand_pos36,
)
from fk_compare.g1_fk import G1PureTorchFK  # noqa: E402


DATA10K_FIELDS = frozenset(DATA10K_TERMS)
DEFAULT_FPS = 50.0
SPEC_MANIFEST_VERSION = 2


@dataclass(frozen=True)
class MotionSpec:
    path: Path
    length: int
    fps: float
    kind: str
    fps_was_defaulted: bool = False


@dataclass
class SourceMotion:
    spec: MotionSpec
    pos36: np.ndarray
    reference: dict[str, np.ndarray] | None


class RunningStats:
    """Constant-memory aggregate for large datasets."""

    def __init__(self, unit: str):
        self.unit = unit
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.maximum = 0.0

    def update(self, error: np.ndarray) -> None:
        flat = np.asarray(error, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            return
        self.count += int(flat.size)
        self.total += float(flat.sum())
        self.total_square += float(np.dot(flat, flat))
        self.maximum = max(self.maximum, float(flat.max()))

    def result(self) -> dict[str, float | int | str]:
        if self.count == 0:
            return {"unit": self.unit, "count": 0, "mean": 0.0, "rmse": 0.0, "max": 0.0}
        return {
            "unit": self.unit,
            "count": self.count,
            "mean": self.total / self.count,
            "rmse": (self.total_square / self.count) ** 0.5,
            "max": self.maximum,
        }


def _scalar_fps(value: np.ndarray, path: Path) -> float:
    values = np.asarray(value).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]) or float(values[0]) <= 0.0:
        raise ValueError(f"Expected one positive FPS value in {path}, got {values.tolist()}")
    return float(values[0])


def _resolve_fps(
    data: Any,
    path: Path,
    fallback_fps: float | None,
) -> tuple[float, bool]:
    if "fps" in data.files and np.asarray(data["fps"]).size > 0:
        return _scalar_fps(data["fps"], path), False
    if fallback_fps is None or not np.isfinite(fallback_fps) or fallback_fps <= 0.0:
        raise ValueError(f"FPS is missing or empty in {path}; provide one positive --fps fallback")
    return float(fallback_fps), True


def inspect_motion(path: Path, *, input_key: str, fallback_fps: float | None) -> MotionSpec:
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != 36:
            raise ValueError(f"Expected [T,36] in {path}, got {array.shape}")
        if fallback_fps is None or not np.isfinite(fallback_fps) or fallback_fps <= 0.0:
            raise ValueError(f"One positive --fps fallback is required for raw .npy input: {path}")
        return MotionSpec(
            path=path,
            length=int(array.shape[0]),
            fps=float(fallback_fps),
            kind="pos36",
            fps_was_defaulted=True,
        )

    with np.load(path, allow_pickle=False) as data:
        if DATA10K_FIELDS.issubset(data.files):
            length = int(data["joint_pos"].shape[0])
            fps, fps_was_defaulted = _resolve_fps(data, path, fallback_fps)
            return MotionSpec(
                path=path,
                length=length,
                fps=fps,
                kind="data10k",
                fps_was_defaulted=fps_was_defaulted,
            )
        if input_key not in data.files:
            raise ValueError(f"Neither Data10k fields nor key '{input_key}' were found in {path}")
        array = data[input_key]
        if array.ndim != 2 or array.shape[1] != 36:
            raise ValueError(f"Expected {input_key}=[T,36] in {path}, got {array.shape}")
        fps, fps_was_defaulted = _resolve_fps(data, path, fallback_fps)
        return MotionSpec(
            path=path,
            length=int(array.shape[0]),
            fps=fps,
            kind="pos36",
            fps_was_defaulted=fps_was_defaulted,
        )


def _validate_data10k_arrays(path: Path, data: Any) -> None:
    length = int(data["joint_pos"].shape[0])
    expected = {
        "joint_pos": (length, len(ISAACLAB_G1_JOINT_NAMES)),
        "joint_vel": (length, len(ISAACLAB_G1_JOINT_NAMES)),
        "body_pos_w": (length, len(ISAACLAB_G1_BODY_NAMES), 3),
        "body_quat_w": (length, len(ISAACLAB_G1_BODY_NAMES), 4),
        "body_lin_vel_w": (length, len(ISAACLAB_G1_BODY_NAMES), 3),
        "body_ang_vel_w": (length, len(ISAACLAB_G1_BODY_NAMES), 3),
    }
    for term, shape in expected.items():
        if data[term].shape != shape:
            raise ValueError(f"Unexpected {term} shape in {path}: {data[term].shape} != {shape}")
        if not np.isfinite(data[term]).all():
            raise ValueError(f"Non-finite values in {term}: {path}")
    quat_norm = np.linalg.norm(data["body_quat_w"][:, 0], axis=-1)
    if not np.allclose(quat_norm, 1.0, atol=1.0e-3):
        raise ValueError(f"Root body_quat_w is not normalized in {path}")


def load_motion(spec: MotionSpec, *, input_key: str) -> SourceMotion:
    path = spec.path
    if path.suffix == ".npy":
        pos36 = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        return SourceMotion(spec=spec, pos36=pos36, reference=None)

    with np.load(path, allow_pickle=False) as data:
        if spec.kind == "data10k":
            _validate_data10k_arrays(path, data)
            reference = {term: np.asarray(data[term], dtype=np.float32) for term in DATA10K_TERMS}
            root_pos = reference["body_pos_w"][:, 0]
            root_quat = reference["body_quat_w"][:, 0]
            pos36 = np.concatenate((root_pos, root_quat, reference["joint_pos"]), axis=-1)
            # The FK stage receives only this minimal tensor.  Reference arrays
            # remain on CPU and are used exclusively after reconstruction.
            return SourceMotion(spec=spec, pos36=pos36.astype(np.float32, copy=False), reference=reference)
        pos36 = np.asarray(data[input_key], dtype=np.float32)
        return SourceMotion(spec=spec, pos36=pos36, reference=None)


def discover_inputs(root: Path) -> list[Path]:
    if root.is_file():
        if root.suffix not in (".npz", ".npy"):
            raise ValueError(f"Only .npz and .npy inputs are supported: {root}")
        return [root]
    motion_paths = sorted(root.rglob("motion.npz"))
    if motion_paths:
        return motion_paths
    paths = sorted((*root.rglob("*.npz"), *root.rglob("*.npy")))
    if not paths:
        raise RuntimeError(f"No .npz or .npy motions found under {root}")
    return paths


def read_manifest(path: Path) -> list[Path]:
    manifest = path.expanduser().resolve()
    paths: list[Path] = []
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        item = Path(line).expanduser()
        if not item.is_absolute():
            item = manifest.parent / item
        item = item.absolute()
        if item.suffix not in (".npz", ".npy") or not item.exists():
            raise ValueError(f"Invalid manifest entry at {manifest}:{line_number}: {item}")
        paths.append(item)
    if not paths:
        raise ValueError(f"Manifest contains no input files: {manifest}")
    if len(set(paths)) != len(paths):
        raise ValueError(f"Manifest contains duplicate paths: {manifest}")
    return paths


def read_spec_manifest(path: Path) -> list[MotionSpec]:
    """Read a launcher-generated manifest without reopening every source NPZ."""

    manifest = path.expanduser().resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != SPEC_MANIFEST_VERSION:
        raise ValueError(f"Unsupported spec manifest version: {manifest}")
    raw_specs = payload.get("motions")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError(f"Spec manifest contains no motions: {manifest}")

    specs: list[MotionSpec] = []
    for index, raw in enumerate(raw_specs):
        try:
            item = Path(raw["path"]).expanduser()
            if not item.is_absolute():
                item = manifest.parent / item
            length = int(raw["length"])
            fps = float(raw["fps"])
            kind = str(raw["kind"])
            fps_was_defaulted = bool(raw.get("fps_was_defaulted", False))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid motion entry {index} in {manifest}") from error
        if item.suffix not in (".npz", ".npy") or length < 1 or not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"Invalid motion entry {index} in {manifest}: {raw!r}")
        if kind not in ("data10k", "pos36"):
            raise ValueError(f"Invalid motion kind in {manifest}: {kind!r}")
        specs.append(
            MotionSpec(
                path=item.absolute(),
                length=length,
                fps=fps,
                kind=kind,
                fps_was_defaulted=fps_was_defaulted,
            )
        )
    if len({spec.path for spec in specs}) != len(specs):
        raise ValueError(f"Spec manifest contains duplicate paths: {manifest}")
    return specs


def make_batches(specs: list[MotionSpec], *, max_padded_frames: int, max_motions: int) -> list[list[MotionSpec]]:
    if max_padded_frames < 1 or max_motions < 1:
        raise ValueError("Batch limits must be positive")
    batches: list[list[MotionSpec]] = []
    current: list[MotionSpec] = []
    max_length = 0
    for spec in sorted(specs, key=lambda item: (item.length, str(item.path))):
        candidate_max = max(max_length, spec.length)
        candidate_padded = candidate_max * (len(current) + 1)
        if current and (candidate_padded > max_padded_frames or len(current) >= max_motions):
            batches.append(current)
            current = []
            max_length = 0
        current.append(spec)
        max_length = max(max_length, spec.length)
    if current:
        batches.append(current)
    return batches


def output_path_for(input_root: Path, output_root: Path, source: Path) -> Path:
    if input_root.is_file():
        relative = Path("motion.npz") if source.name == "motion.npz" else Path(f"{source.stem}.motion.npz")
    else:
        relative = source.relative_to(input_root)
        if source.name != "motion.npz":
            relative = relative.with_name(f"{source.stem}.motion.npz")
    return output_root / relative


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_worker_progress(
    path: Path | None,
    *,
    device: torch.device,
    status: str,
    completed_motions: int,
    completed_frames: int,
    total_motions: int,
    total_frames: int,
    start: float,
) -> None:
    if path is None:
        return
    elapsed = time.perf_counter() - start
    _atomic_write_json(
        path,
        {
            "pid": os.getpid(),
            "device": str(device),
            "status": status,
            "completed_motion_count": completed_motions,
            "completed_frame_count": completed_frames,
            "total_motion_count": total_motions,
            "total_frame_count": total_frames,
            "elapsed_seconds": elapsed,
            "frames_per_second": completed_frames / elapsed if elapsed > 0.0 else None,
            "updated_at_unix": time.time(),
        },
    )


def pack_batch(sources: list[SourceMotion], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(sources)
    max_length = max(source.spec.length for source in sources)
    pin_memory = device.type == "cuda"
    cpu = torch.empty((batch, max_length, 36), dtype=torch.float32, pin_memory=pin_memory)
    lengths = torch.tensor([source.spec.length for source in sources], dtype=torch.long)
    fps = torch.tensor([source.spec.fps for source in sources], dtype=torch.float32)
    for index, source in enumerate(sources):
        length = source.spec.length
        cpu[index, :length].copy_(torch.from_numpy(source.pos36))
        if length < max_length:
            cpu[index, length:].copy_(cpu[index, length - 1].expand(max_length - length, -1))
    non_blocking = device.type == "cuda"
    return (
        cpu.to(device=device, non_blocking=non_blocking),
        lengths.to(device=device, non_blocking=non_blocking),
        fps.to(device=device, non_blocking=non_blocking),
    )


def _record_global(
    aggregates: dict[str, dict[str, RunningStats]],
    category: str,
    errors: dict[str, np.ndarray],
) -> None:
    category_stats = aggregates.setdefault(category, {})
    for term, error in errors.items():
        stats = category_stats.setdefault(term, RunningStats(TERM_UNITS[term]))
        stats.update(error)


def _global_result(aggregates: dict[str, dict[str, RunningStats]]) -> dict[str, Any]:
    return {
        category: {term: stats.result() for term, stats in terms.items()}
        for category, terms in aggregates.items()
    }


def process_batch(
    sources: list[SourceMotion],
    *,
    input_root: Path,
    output_root: Path,
    device: torch.device,
    fk_helper: Any,
    overwrite: bool,
    aggregates: dict[str, dict[str, RunningStats]],
) -> None:
    pos36, lengths, fps = pack_batch(sources, device)
    with torch.inference_mode():
        processed, raw = expand_pos36(pos36, lengths, fps, fk_helper)
    processed_cpu = {term: value.detach().cpu().numpy() for term, value in processed.items()}
    raw_cpu = {term: value.detach().cpu().numpy() for term, value in raw.items()}

    velocity_terms = ("root_lin_vel_w", "root_ang_vel_w", "joint_vel", "body_lin_vel_w", "body_ang_vel_w")
    for batch_index, source in enumerate(sources):
        length = source.spec.length
        output_path = output_path_for(input_root, output_root, source.spec.path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_path}")

        output_arrays = {
            "fps": np.asarray([source.spec.fps], dtype=np.int64)
            if source.spec.fps.is_integer()
            else np.asarray([source.spec.fps], dtype=np.float32),
        }
        output_arrays.update(
            {
                term: np.asarray(processed_cpu[term][batch_index, :length], dtype=np.float32)
                for term in DATA10K_TERMS
            }
        )
        _atomic_save_npz(output_path, output_arrays)

        processed_one = {term: output_arrays[term] for term in DATA10K_TERMS}
        raw_one = {term: raw_cpu[term][batch_index, :length] for term in velocity_terms}
        smooth_one = {term: processed_cpu[term][batch_index, :length] for term in velocity_terms}
        raw_summary, raw_errors = compare_terms(raw_one, smooth_one, list(velocity_terms))
        _record_global(aggregates, "raw_vs_smoothed", raw_errors)

        reference_summary = None
        if source.reference is not None:
            reference_summary, reference_errors = compare_terms(
                source.reference, processed_one, list(DATA10K_TERMS)
            )
            _record_global(aggregates, "reference_vs_processed", reference_errors)

        report = {
            "source": str(source.spec.path),
            "output": str(output_path),
            "frames": length,
            "fps": source.spec.fps,
            "fps_was_defaulted": source.spec.fps_was_defaulted,
            "minimal_input": {
                "shape": [length, 36],
                "layout": "root_pos[3] + root_quat_wxyz[4] + joint_pos_isaaclab_order[29]",
                "dtype": "float32",
                "fk_consumed_only_this_tensor": True,
            },
            "output_contract": {
                "npz_fields": ["fps", *DATA10K_TERMS],
                "joint_order": "IsaacLab Data10k G1 order",
                "body_order": "IsaacLab Entity.body_names order",
                "quaternion_order": "wxyz",
                "body_position_semantics": "link_origin_world",
                "body_lin_vel_w_semantics": "link_origin_world (HEFT/mjlab; not IsaacLab 2.2 COM alias)",
                "body_ang_vel_w_semantics": "world",
            },
            "reference_contract": {
                "available": source.reference is not None,
                "body_lin_vel_w_semantics": "IsaacLab exporter value; expected COM for the legacy Data10k pipeline",
            },
            "reference_vs_processed": reference_summary,
            "raw_vs_smoothed": raw_summary,
        }
        _atomic_write_json(output_path.with_suffix(".diff.json"), report)


def build_fk_helper(device: torch.device) -> G1PureTorchFK:
    return G1PureTorchFK(device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One motion or a directory tree.")
    parser.add_argument("--output-dir", type=Path, required=True)
    input_subset = parser.add_mutually_exclusive_group()
    input_subset.add_argument("--manifest", type=Path, help="Optional newline-delimited subset of input files.")
    input_subset.add_argument(
        "--spec-manifest",
        type=Path,
        help="Launcher-generated JSON manifest containing validated length/FPS metadata.",
    )
    parser.add_argument("--summary-name", type=Path, default=Path("summary.json"))
    parser.add_argument("--progress-path", type=Path, help="Optional atomic worker-progress JSON path.")
    parser.add_argument("--device", default="cuda", help="Torch device, normally cuda or cuda:0.")
    parser.add_argument("--input-key", default="pos", help="36D key for a non-Data10k NPZ.")
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Fallback when FPS is missing/empty (default: {DEFAULT_FPS:g} Hz).",
    )
    parser.add_argument("--batch-frames", type=int, default=32768, help="Maximum padded frames per GPU batch.")
    parser.add_argument("--batch-motions", type=int, default=32)
    parser.add_argument("--io-workers", type=int, default=4)
    write_mode = parser.add_mutually_exclusive_group()
    write_mode.add_argument("--overwrite", action="store_true")
    write_mode.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    input_root = args.input.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    if args.io_workers < 1:
        raise ValueError("--io-workers must be positive")
    if args.summary_name.is_absolute() or ".." in args.summary_name.parts:
        raise ValueError("--summary-name must stay inside --output-dir")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu only for validation")

    if args.spec_manifest:
        specs = read_spec_manifest(args.spec_manifest)
    else:
        paths = read_manifest(args.manifest) if args.manifest else discover_inputs(input_root)
        with ThreadPoolExecutor(max_workers=args.io_workers) as executor:
            specs = list(
                executor.map(
                    lambda path: inspect_motion(path, input_key=args.input_key, fallback_fps=args.fps), paths
                )
            )
    for spec in specs:
        try:
            spec.path.relative_to(input_root if input_root.is_dir() else input_root.parent)
        except ValueError as error:
            raise ValueError(f"Manifest input is outside --input: {spec.path}") from error
    if any(spec.length < 1 for spec in specs):
        raise ValueError("Empty motions are not supported")
    discovered_motion_count = len(specs)
    expected_outputs = [output_path_for(input_root, output_root, spec.path) for spec in specs]
    if args.skip_existing:
        specs = [spec for spec, output in zip(specs, expected_outputs) if not output.exists()]
        expected_outputs = [output_path_for(input_root, output_root, spec.path) for spec in specs]
    skipped_motion_count = discovered_motion_count - len(specs)
    batches = make_batches(specs, max_padded_frames=args.batch_frames, max_motions=args.batch_motions) if specs else []
    conflicts = [path for path in expected_outputs if path.exists()]
    if conflicts and not args.overwrite:
        preview = "\n".join(str(path) for path in conflicts[:10])
        raise FileExistsError(f"{len(conflicts)} outputs already exist; pass --overwrite. First paths:\n{preview}")

    total_frames = sum(spec.length for spec in specs)
    progress_path = args.progress_path.expanduser().resolve() if args.progress_path else None
    progress_start = time.perf_counter()
    completed_motions = 0
    completed_frames = 0
    _write_worker_progress(
        progress_path,
        device=device,
        status="starting",
        completed_motions=0,
        completed_frames=0,
        total_motions=len(specs),
        total_frames=total_frames,
        start=progress_start,
    )

    fk_helper = build_fk_helper(device)
    aggregates: dict[str, dict[str, RunningStats]] = {}
    start = time.perf_counter()
    _write_worker_progress(
        progress_path,
        device=device,
        status="running",
        completed_motions=0,
        completed_frames=0,
        total_motions=len(specs),
        total_frames=total_frames,
        start=progress_start,
    )
    with ThreadPoolExecutor(max_workers=args.io_workers) as executor:
        def submit_loads(batch_specs: list[MotionSpec]):
            return [executor.submit(load_motion, spec, input_key=args.input_key) for spec in batch_specs]

        if batches:
            pending_loads = submit_loads(batches[0])
            for batch_index, batch_specs in enumerate(batches, start=1):
                sources = [future.result() for future in pending_loads]
                # Load and decode the next NPZ batch while the current batch runs
                # FK on the GPU and writes its output files.
                pending_loads = submit_loads(batches[batch_index]) if batch_index < len(batches) else []
                process_batch(
                    sources,
                    input_root=input_root,
                    output_root=output_root,
                    device=device,
                    fk_helper=fk_helper,
                    overwrite=args.overwrite,
                    aggregates=aggregates,
                )
                completed_motions += len(sources)
                completed_frames += sum(source.spec.length for source in sources)
                _write_worker_progress(
                    progress_path,
                    device=device,
                    status="running",
                    completed_motions=completed_motions,
                    completed_frames=completed_frames,
                    total_motions=len(specs),
                    total_frames=total_frames,
                    start=progress_start,
                )
                print(
                    f"batch {batch_index}/{len(batches)}: "
                    f"{len(sources)} motions, {sum(source.spec.length for source in sources)} frames"
                )

    elapsed = time.perf_counter() - start
    summary = {
        "input": str(input_root),
        "output_dir": str(output_root),
        "fk_asset": fk_helper.asset_name,
        "runtime_dependencies": ["numpy", "torch"],
        "device": str(device),
        "manifest": str((args.spec_manifest or args.manifest).expanduser().resolve())
        if args.spec_manifest or args.manifest
        else None,
        "discovered_motion_count": discovered_motion_count,
        "skipped_existing_motion_count": skipped_motion_count,
        "motion_count": len(specs),
        "frame_count": total_frames,
        "defaulted_fps_motion_count": sum(spec.fps_was_defaulted for spec in specs),
        "fallback_fps": args.fps,
        "elapsed_seconds": elapsed,
        "frames_per_second": total_frames / elapsed if elapsed > 0.0 else None,
        "batch_frames_limit": args.batch_frames,
        "batch_motions_limit": args.batch_motions,
        "global_differences": _global_result(aggregates),
        "notes": [
            "Per-file p95 is exact; global aggregation is constant-memory and reports mean/RMSE/max.",
            "body_lin_vel_w in outputs is link-origin world velocity for HEFT/mjlab compatibility.",
        ],
    }
    summary_path = output_root / args.summary_name
    _atomic_write_json(summary_path, summary)
    _write_worker_progress(
        progress_path,
        device=device,
        status="complete",
        completed_motions=completed_motions,
        completed_frames=completed_frames,
        total_motions=len(specs),
        total_frames=total_frames,
        start=progress_start,
    )
    print(f"processed {len(specs)} motions / {total_frames} frames in {elapsed:.3f}s")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
