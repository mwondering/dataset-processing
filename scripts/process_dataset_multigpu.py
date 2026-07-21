#!/usr/bin/env python3
"""Launch independent, balanced 36D dataset workers across local CUDA GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.process_isaaclab_pos36 import (  # noqa: E402
    DEFAULT_FPS,
    MotionSpec,
    SPEC_MANIFEST_VERSION,
    _atomic_write_json,
    inspect_motion,
    output_path_for,
    validate_unique_output_paths,
)


DATASET_INDEX_VERSION = 3


def parse_gpu_ids(value: str | None) -> list[int]:
    if value is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment")
        ids = list(range(torch.cuda.device_count()))
    else:
        try:
            ids = [int(item.strip()) for item in value.split(",") if item.strip()]
        except ValueError as error:
            raise ValueError(f"Invalid --gpus value: {value!r}") from error
    if not ids or len(set(ids)) != len(ids) or any(item < 0 for item in ids):
        raise ValueError(f"GPU IDs must be unique non-negative integers, got {ids}")
    visible = torch.cuda.device_count()
    if any(item >= visible for item in ids):
        raise ValueError(f"Requested GPU IDs {ids}, but this process sees {visible} CUDA devices")
    return ids


def balanced_shards(specs: list[Any], shard_count: int) -> tuple[list[list[Any]], list[int]]:
    """Greedy longest-first partitioning by frame count."""

    shards: list[list[Any]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for spec in sorted(specs, key=lambda item: (-item.length, str(item.path))):
        shard_index = min(range(shard_count), key=lambda index: (loads[index], index))
        shards[shard_index].append(spec)
        loads[shard_index] += spec.length
    for shard in shards:
        shard.sort(key=lambda item: str(item.path))
    return shards, loads


def merge_global_differences(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    accumulators: dict[str, dict[str, dict[str, Any]]] = {}
    for summary in summaries:
        for category, terms in summary.get("global_differences", {}).items():
            for term, stats in terms.items():
                count = int(stats["count"])
                target = accumulators.setdefault(category, {}).setdefault(
                    term,
                    {"unit": stats["unit"], "count": 0, "total": 0.0, "square": 0.0, "max": 0.0},
                )
                target["count"] += count
                target["total"] += float(stats["mean"]) * count
                target["square"] += float(stats["rmse"]) ** 2 * count
                target["max"] = max(target["max"], float(stats["max"]))

    merged: dict[str, Any] = {}
    for category, terms in accumulators.items():
        merged[category] = {}
        for term, value in terms.items():
            count = value["count"]
            merged[category][term] = {
                "unit": value["unit"],
                "count": count,
                "mean": value["total"] / count if count else 0.0,
                "rmse": (value["square"] / count) ** 0.5 if count else 0.0,
                "max": value["max"],
            }
    return merged


def tail(path: Path, lines: int = 40) -> str:
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _all_motion_candidates(paths: list[Path]) -> list[Path]:
    if not paths:
        raise RuntimeError("No .npz or .npy motions found")
    return sorted(paths)


def _collect_motion_files_os_walk(root: Path) -> tuple[list[Path], int, int]:
    paths: list[Path] = []
    directory_count = 0
    file_count = 0
    for directory, _, filenames in os.walk(root):
        directory_count += 1
        file_count += len(filenames)
        for filename in filenames:
            if filename.endswith((".npz", ".npy")):
                paths.append(Path(directory) / filename)
    return paths, directory_count, file_count


def _scan_inputs_python(root: Path, *, workers: int, log_interval: float) -> list[Path]:
    start = time.perf_counter()
    print(f"path scan start: backend=python workers={workers} root={root}", flush=True)
    with os.scandir(root) as entries:
        root_files: list[Path] = []
        child_dirs: list[Path] = []
        root_file_count = 0
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                child_dirs.append(Path(entry.path))
            else:
                root_file_count += 1
                if entry.name.endswith((".npz", ".npy")):
                    root_files.append(Path(entry.path))

    paths = list(root_files)
    directory_count = 1
    file_count = root_file_count
    last_log = start
    if workers > 1 and len(child_dirs) > 1:
        worker_count = min(workers, len(child_dirs))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_collect_motion_files_os_walk, child) for child in child_dirs]
            for completed, future in enumerate(as_completed(futures), start=1):
                child_paths, child_directory_count, child_file_count = future.result()
                paths.extend(child_paths)
                directory_count += child_directory_count
                file_count += child_file_count
                now = time.perf_counter()
                if log_interval > 0.0 and now - last_log >= log_interval:
                    print(
                        "path scan progress: "
                        f"dirs={directory_count:,} files={file_count:,} motions={len(paths):,} "
                        f"roots={completed}/{len(child_dirs)} elapsed={now - start:.1f}s",
                        flush=True,
                    )
                    last_log = now
    else:
        child_paths, child_directory_count, child_file_count = _collect_motion_files_os_walk(root)
        paths = child_paths
        directory_count = child_directory_count
        file_count = child_file_count

    selected = _all_motion_candidates(paths)
    print(
        f"path scan done: backend=python dirs={directory_count:,} files={file_count:,} "
        f"motions={len(selected):,} elapsed={time.perf_counter() - start:.2f}s",
        flush=True,
    )
    return selected


def _find_fd(executable: str | None) -> str | None:
    candidates = [executable] if executable else ["fd", "fdfind"]
    for candidate in candidates:
        if candidate:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
    return None


def _scan_inputs_fd(
    root: Path,
    *,
    executable: str,
    workers: int,
    log_interval: float,
) -> list[Path]:
    start = time.perf_counter()
    command = [
        executable,
        "--hidden",
        "--no-ignore",
        "--type",
        "f",
        "--color",
        "never",
    ]
    if workers > 0:
        command.extend(("--threads", str(workers)))
    command.extend((r"\.(npz|npy)$", str(root)))
    print(f"path scan start: backend=fd workers={workers} root={root}", flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    paths: list[Path] = []
    last_log = start
    assert process.stdout is not None
    for line in process.stdout:
        value = line.strip()
        if value:
            paths.append(Path(value).absolute())
        now = time.perf_counter()
        if log_interval > 0.0 and now - last_log >= log_interval:
            print(
                f"path scan progress: backend=fd motions={len(paths):,} elapsed={now - start:.1f}s",
                flush=True,
            )
            last_log = now
    stderr = process.stderr.read() if process.stderr is not None else ""
    code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command, stderr=stderr)
    selected = _all_motion_candidates(paths)
    print(
        f"path scan done: backend=fd motions={len(selected):,} "
        f"elapsed={time.perf_counter() - start:.2f}s",
        flush=True,
    )
    return selected


def discover_inputs_fast(
    root: Path,
    *,
    backend: str,
    workers: int,
    fd_executable: str | None,
    log_interval: float,
) -> list[Path]:
    if root.is_file():
        if root.suffix not in (".npz", ".npy"):
            raise ValueError(f"Only .npz and .npy inputs are supported: {root}")
        return [root]
    if backend not in ("auto", "fd", "python"):
        raise ValueError(f"Unsupported scan backend: {backend}")
    if backend in ("auto", "fd"):
        fd_path = _find_fd(fd_executable)
        if fd_path:
            try:
                return _scan_inputs_fd(
                    root,
                    executable=fd_path,
                    workers=workers,
                    log_interval=log_interval,
                )
            except (OSError, subprocess.SubprocessError) as error:
                if backend == "fd":
                    raise RuntimeError(f"fd input scan failed: {error}") from error
                print(f"fd scan failed; falling back to Python: {error}", flush=True)
        elif backend == "fd":
            requested = fd_executable or "fd/fdfind"
            raise FileNotFoundError(f"scan backend 'fd' requested but executable was not found: {requested}")
    return _scan_inputs_python(root, workers=workers, log_interval=log_interval)


def _inspect_motion_job(
    job: tuple[int, str, str, float | None],
) -> tuple[int, MotionSpec]:
    index, path, input_key, fallback_fps = job
    return index, inspect_motion(Path(path), input_key=input_key, fallback_fps=fallback_fps)


def read_motion_specs(
    paths: list[Path],
    *,
    input_key: str,
    fallback_fps: float | None,
    backend: str,
    workers: int,
    chunksize: int,
    log_interval: float,
) -> list[MotionSpec]:
    total = len(paths)
    worker_count = min(max(workers, 1), total)
    resolved_backend = "serial" if backend == "serial" or worker_count <= 1 else backend
    print(
        f"metadata read start: count={total:,} backend={resolved_backend} "
        f"workers={worker_count} chunksize={chunksize}",
        flush=True,
    )
    start = time.perf_counter()
    last_log = start
    specs: list[MotionSpec | None] = [None] * total
    completed = 0

    def record(index: int, spec: MotionSpec) -> None:
        nonlocal completed, last_log
        specs[index] = spec
        completed += 1
        now = time.perf_counter()
        if completed == total or (log_interval > 0.0 and now - last_log >= log_interval):
            rate = completed / (now - start) if now > start else 0.0
            print(
                f"metadata progress: {completed:,}/{total:,} files "
                f"({completed / total * 100:.1f}%) {rate:,.0f} file/s "
                f"elapsed={now - start:.1f}s",
                flush=True,
            )
            last_log = now

    jobs = ((index, str(path), input_key, fallback_fps) for index, path in enumerate(paths))
    if resolved_backend == "serial":
        for job in jobs:
            index, spec = _inspect_motion_job(job)
            record(index, spec)
    elif resolved_backend == "process":
        with mp.Pool(processes=worker_count) as pool:
            for index, spec in pool.imap_unordered(_inspect_motion_job, jobs, chunksize=chunksize):
                record(index, spec)
    elif resolved_backend == "thread":
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_inspect_motion_job, job) for job in jobs]
            for future in as_completed(futures):
                index, spec = future.result()
                record(index, spec)
    else:
        raise ValueError(f"Unsupported metadata backend: {backend}")

    if any(spec is None for spec in specs):
        raise RuntimeError("Metadata reader returned an incomplete result")
    print(f"metadata read done: {total:,} files in {time.perf_counter() - start:.2f}s", flush=True)
    return [spec for spec in specs if spec is not None]


def load_dataset_index(
    path: Path,
    *,
    input_root: Path,
    input_key: str,
    fallback_fps: float | None,
) -> list[MotionSpec] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload["version"]) != DATASET_INDEX_VERSION:
            return None
        if payload["input"] != str(input_root) or payload["input_key"] != input_key:
            return None
        if payload.get("fallback_fps") != fallback_fps:
            return None
        specs = [
            MotionSpec(
                path=Path(raw["path"]),
                length=int(raw["length"]),
                fps=float(raw["fps"]),
                kind=str(raw["kind"]),
                fps_was_defaulted=bool(raw.get("fps_was_defaulted", False)),
            )
            for raw in payload["motions"]
        ]
        if not specs or any(spec.length < 1 or spec.kind not in ("data10k", "pos36") for spec in specs):
            return None
        return specs
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"dataset index ignored: file={path} error={error}", flush=True)
        return None


def write_dataset_index(
    path: Path,
    *,
    input_root: Path,
    input_key: str,
    fallback_fps: float | None,
    specs: list[MotionSpec],
) -> None:
    _atomic_write_json(
        path,
        {
            "version": DATASET_INDEX_VERSION,
            "input": str(input_root),
            "input_key": input_key,
            "fallback_fps": fallback_fps,
            "motion_count": len(specs),
            "frame_count": sum(spec.length for spec in specs),
            "defaulted_fps_motion_count": sum(spec.fps_was_defaulted for spec in specs),
            "motions": [
                {
                    "path": str(spec.path),
                    "length": spec.length,
                    "fps": spec.fps,
                    "kind": spec.kind,
                    "fps_was_defaulted": spec.fps_was_defaulted,
                }
                for spec in specs
            ],
        },
    )


def write_spec_manifest(path: Path, specs: list[MotionSpec]) -> None:
    _atomic_write_json(
        path,
        {
            "version": SPEC_MANIFEST_VERSION,
            "motions": [
                {
                    "path": str(spec.path),
                    "length": spec.length,
                    "fps": spec.fps,
                    "kind": spec.kind,
                    "fps_was_defaulted": spec.fps_was_defaulted,
                }
                for spec in specs
            ],
        },
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0.0:
        return "--:--:--"
    total = int(seconds + 0.5)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_progress_lines(
    plan: dict[str, Any],
    workers: list[dict[str, Any]],
    states: list[dict[str, Any]],
    *,
    elapsed: float,
    bar_width: int = 30,
) -> list[str]:
    completed_motions = sum(int(state.get("completed_motion_count", 0)) for state in states)
    completed_frames = sum(int(state.get("completed_frame_count", 0)) for state in states)
    total_motions = int(plan["motion_count"])
    total_frames = int(plan["frame_count"])
    completed_motions = min(completed_motions, total_motions)
    completed_frames = min(completed_frames, total_frames)
    ratio = completed_frames / total_frames if total_frames else 1.0
    filled = min(bar_width, int(ratio * bar_width + 0.5))
    speed = completed_frames / elapsed if elapsed > 0.0 else 0.0
    eta = (total_frames - completed_frames) / speed if speed > 0.0 else None

    lines = [
        f"overall  {ratio * 100:5.1f}% │{'█' * filled}{'─' * (bar_width - filled)}│",
        f"motions  {completed_motions:,} / {total_motions:,}",
        f"frames   {completed_frames:,} / {total_frames:,}",
        f"speed    {speed:,.0f} frame/s",
        f"ETA      {_format_duration(eta)}",
    ]
    for worker, state in zip(workers, states):
        worker_total = int(worker["motion_count"])
        worker_done = min(int(state.get("completed_motion_count", 0)), worker_total)
        lines.append(f"GPU{worker['gpu_id']:<3}   {worker_done:,} / {worker_total:,} motions")
    return lines


def read_progress_states(workers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for worker in workers:
        try:
            states.append(json.loads(worker["progress"].read_text(encoding="utf-8")))
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
            states.append({})
    return states


class ProgressRenderer:
    def __init__(self, stream: Any = sys.stdout):
        self.stream = stream
        self.is_tty = bool(stream.isatty())
        self.previous_line_count = 0

    def render(self, lines: list[str]) -> None:
        if self.is_tty:
            if self.previous_line_count:
                self.stream.write(f"\x1b[{self.previous_line_count}F")
            for line in lines:
                self.stream.write(f"\x1b[2K{line}\n")
            for _ in range(max(0, self.previous_line_count - len(lines))):
                self.stream.write("\x1b[2K\n")
            self.previous_line_count = len(lines)
        else:
            self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", help="Comma-separated local CUDA IDs; default uses every visible GPU.")
    parser.add_argument("--input-key", default="pos")
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Fallback when FPS is missing/empty (default: {DEFAULT_FPS:g} Hz).",
    )
    parser.add_argument("--batch-frames", type=int, default=262144)
    parser.add_argument("--batch-motions", type=int, default=32)
    parser.add_argument("--io-workers-per-gpu", type=int, default=4)
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=16,
        help="Workers for filesystem enumeration, metadata reads, and existing-output checks.",
    )
    parser.add_argument("--scan-backend", choices=("auto", "fd", "python"), default="auto")
    parser.add_argument(
        "--scan-fd-executable",
        help="fd executable name/path; auto mode searches both fd and fdfind.",
    )
    parser.add_argument(
        "--metadata-read-backend",
        choices=("process", "thread", "serial"),
        default="process",
    )
    parser.add_argument("--metadata-read-chunksize", type=int, default=128)
    parser.add_argument("--scan-log-interval", type=float, default=10.0)
    parser.add_argument(
        "--index-cache",
        type=Path,
        help="Persistent dataset index; default is OUTPUT/_cluster/input_index.json.",
    )
    parser.add_argument("--no-index-cache", action="store_true")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Ignore and replace the cached input snapshot after the source dataset changes.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        help="Refresh seconds; default is 1 in a terminal and 10 in redirected/Slurm logs.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable the aggregate progress display.")
    write_mode = parser.add_mutually_exclusive_group()
    write_mode.add_argument("--overwrite", action="store_true")
    write_mode.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write balanced manifests without starting workers.")
    args = parser.parse_args()

    if args.batch_frames < 1 or args.batch_motions < 1:
        raise ValueError("Batch limits must be positive")
    if args.io_workers_per_gpu < 1 or args.scan_workers < 1:
        raise ValueError("Worker counts must be positive")
    if args.metadata_read_chunksize < 1:
        raise ValueError("--metadata-read-chunksize must be positive")
    if args.scan_log_interval < 0.0:
        raise ValueError("--scan-log-interval must be non-negative")
    if args.no_index_cache and args.index_cache:
        raise ValueError("--index-cache and --no-index-cache cannot be combined")
    if args.no_index_cache and args.rebuild_index:
        raise ValueError("--rebuild-index has no effect with --no-index-cache")
    if args.progress_interval is not None and args.progress_interval <= 0.0:
        raise ValueError("--progress-interval must be positive")

    input_root = args.input.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(input_root)

    output_preexisting = output_root.exists()
    cluster_dir = output_root / "_cluster"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    index_path = (
        args.index_cache.expanduser().resolve()
        if args.index_cache
        else cluster_dir / "input_index.json"
    )
    specs = None
    index_hit = False
    if not args.no_index_cache and not args.rebuild_index:
        specs = load_dataset_index(
            index_path,
            input_root=input_root,
            input_key=args.input_key,
            fallback_fps=args.fps,
        )
        index_hit = specs is not None
    if specs is not None:
        print(
            f"dataset index hit: {len(specs):,} motions / "
            f"{sum(spec.length for spec in specs):,} frames <- {index_path}",
            flush=True,
        )
        print("dataset index is a snapshot; use --rebuild-index after changing input files", flush=True)
    else:
        print(f"dataset index build start: cache={index_path if not args.no_index_cache else 'disabled'}", flush=True)
        paths = discover_inputs_fast(
            input_root,
            backend=args.scan_backend,
            workers=args.scan_workers,
            fd_executable=args.scan_fd_executable,
            log_interval=args.scan_log_interval,
        )
        specs = read_motion_specs(
            paths,
            input_key=args.input_key,
            fallback_fps=args.fps,
            backend=args.metadata_read_backend,
            workers=args.scan_workers,
            chunksize=args.metadata_read_chunksize,
            log_interval=args.scan_log_interval,
        )
        defaulted_fps_count = sum(spec.fps_was_defaulted for spec in specs)
        if defaulted_fps_count:
            print(
                f"metadata FPS fallback: {defaulted_fps_count:,}/{len(specs):,} motions "
                f"used {args.fps:g} Hz because fps was missing or empty",
                flush=True,
            )
        if not args.no_index_cache:
            write_dataset_index(
                index_path,
                input_root=input_root,
                input_key=args.input_key,
                fallback_fps=args.fps,
                specs=specs,
            )
            print(f"dataset index wrote: {index_path}", flush=True)

    validate_unique_output_paths(input_root, output_root, specs)

    skipped_existing = 0
    if args.skip_existing:
        if output_preexisting:
            start = time.perf_counter()
            print(f"existing-output check start: {len(specs):,} motions", flush=True)
            expected_outputs = [output_path_for(input_root, output_root, spec.path) for spec in specs]
            with ThreadPoolExecutor(max_workers=args.scan_workers) as executor:
                existing = list(executor.map(Path.exists, expected_outputs))
            pending_specs = [spec for spec, output_exists in zip(specs, existing) if not output_exists]
            skipped_existing = len(specs) - len(pending_specs)
            specs = pending_specs
            print(
                f"existing-output check done: skipped={skipped_existing:,} "
                f"pending={len(specs):,} elapsed={time.perf_counter() - start:.2f}s",
                flush=True,
            )
    if not specs:
        print(f"nothing to process; skipped {skipped_existing} existing motions")
        return

    # CUDA is queried only after the optional process-based metadata pool exits.
    # This avoids forking worker processes after CUDA runtime initialization.
    gpu_ids = parse_gpu_ids(args.gpus)
    shards, frame_loads = balanced_shards(specs, len(gpu_ids))

    worker_script = Path(__file__).with_name("process_isaaclab_pos36.py").resolve()
    worker_specs: list[dict[str, Any]] = []
    for gpu_id, shard, frame_count in zip(gpu_ids, shards, frame_loads):
        if not shard:
            continue
        manifest = cluster_dir / f"manifest.gpu{gpu_id}.json"
        write_spec_manifest(manifest, shard)
        log_path = cluster_dir / f"worker.gpu{gpu_id}.log"
        progress_path = cluster_dir / f"progress.gpu{gpu_id}.json"
        summary_name = Path("_cluster") / f"summary.gpu{gpu_id}.json"
        command = [
            sys.executable,
            str(worker_script),
            "--input", str(input_root),
            "--spec-manifest", str(manifest),
            "--output-dir", str(output_root),
            "--summary-name", str(summary_name),
            "--progress-path", str(progress_path),
            "--device", f"cuda:{gpu_id}",
            "--input-key", args.input_key,
            "--batch-frames", str(args.batch_frames),
            "--batch-motions", str(args.batch_motions),
            "--io-workers", str(args.io_workers_per_gpu),
        ]
        if args.fps is not None:
            command.extend(("--fps", str(args.fps)))
        if args.overwrite:
            command.append("--overwrite")
        elif args.skip_existing:
            command.append("--skip-existing")
        worker_specs.append(
            {
                "gpu_id": gpu_id,
                "motion_count": len(shard),
                "frame_count": frame_count,
                "manifest": manifest,
                "log": log_path,
                "progress": progress_path,
                "summary": output_root / summary_name,
                "command": command,
            }
        )

    plan = {
        "input": str(input_root),
        "output_dir": str(output_root),
        "python": sys.executable,
        "gpus": gpu_ids,
        "motion_count": len(specs),
        "frame_count": sum(spec.length for spec in specs),
        "skipped_existing_motion_count": skipped_existing,
        "defaulted_fps_motion_count": sum(spec.fps_was_defaulted for spec in specs),
        "fallback_fps": args.fps,
        "dataset_index": None if args.no_index_cache else str(index_path),
        "dataset_index_hit": index_hit,
        "scan_backend": args.scan_backend,
        "metadata_read_backend": args.metadata_read_backend,
        "workers": [
            {key: str(value) if isinstance(value, Path) else value for key, value in item.items() if key != "command"}
            for item in worker_specs
        ],
    }
    _atomic_write_json(cluster_dir / "plan.json", plan)
    for item in worker_specs:
        print(
            f"gpu {item['gpu_id']}: {item['motion_count']} motions / "
            f"{item['frame_count']} frames -> {item['log']}"
        )
    if args.dry_run:
        print(f"dry run complete: {cluster_dir / 'plan.json'}")
        return

    start = time.perf_counter()
    processes: list[tuple[dict[str, Any], subprocess.Popen[Any], Any]] = []
    child_env = os.environ.copy()
    child_env.setdefault("OMP_NUM_THREADS", "1")
    for item in worker_specs:
        item["progress"].unlink(missing_ok=True)
        log_stream = item["log"].open("w", encoding="utf-8")
        process = subprocess.Popen(
            item["command"],
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            env=child_env,
        )
        processes.append((item, process, log_stream))

    failed: tuple[dict[str, Any], int] | None = None
    active = list(processes)
    renderer = ProgressRenderer()
    progress_interval = args.progress_interval
    if progress_interval is None:
        progress_interval = 1.0 if renderer.is_tty else 10.0
    last_render = float("-inf")
    while active:
        for entry in list(active):
            item, process, log_stream = entry
            code = process.poll()
            if code is None:
                continue
            log_stream.close()
            active.remove(entry)
            if code != 0 and failed is None:
                failed = (item, code)
                for _, other, _ in active:
                    other.terminate()
        now = time.perf_counter()
        if not args.no_progress and now - last_render >= progress_interval:
            renderer.render(
                build_progress_lines(
                    plan,
                    worker_specs,
                    read_progress_states(worker_specs),
                    elapsed=now - start,
                )
            )
            last_render = now
        if active:
            time.sleep(0.2)
    for _, process, log_stream in processes:
        process.wait()
        if not log_stream.closed:
            log_stream.close()
    if not args.no_progress:
        renderer.render(
            build_progress_lines(
                plan,
                worker_specs,
                read_progress_states(worker_specs),
                elapsed=time.perf_counter() - start,
            )
        )
    if failed is not None:
        item, code = failed
        raise RuntimeError(
            f"GPU {item['gpu_id']} worker exited with code {code}. Last log lines:\n{tail(item['log'])}"
        )

    elapsed = time.perf_counter() - start
    worker_summaries = [json.loads(item["summary"].read_text(encoding="utf-8")) for item in worker_specs]
    summary = {
        **plan,
        "elapsed_seconds": elapsed,
        "frames_per_second": plan["frame_count"] / elapsed if elapsed > 0.0 else None,
        "batch_frames_limit_per_gpu": args.batch_frames,
        "batch_motions_limit_per_gpu": args.batch_motions,
        "global_differences": merge_global_differences(worker_summaries),
        "worker_summaries": [str(item["summary"]) for item in worker_specs],
    }
    _atomic_write_json(output_root / "summary.json", summary)
    print(
        f"completed {plan['motion_count']} motions / {plan['frame_count']} frames on "
        f"{len(worker_specs)} GPUs in {elapsed:.3f}s"
    )
    print(f"summary: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
