#!/usr/bin/env python3
"""Launch independent, balanced 36D dataset workers across local CUDA GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.process_isaaclab_pos36 import (  # noqa: E402
    _atomic_write_json,
    discover_inputs,
    inspect_motion,
    output_path_for,
)


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
    parser.add_argument("--fps", type=float)
    parser.add_argument("--batch-frames", type=int, default=262144)
    parser.add_argument("--batch-motions", type=int, default=32)
    parser.add_argument("--io-workers-per-gpu", type=int, default=4)
    parser.add_argument("--scan-workers", type=int, default=16)
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
    if args.progress_interval is not None and args.progress_interval <= 0.0:
        raise ValueError("--progress-interval must be positive")

    input_root = args.input.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    gpu_ids = parse_gpu_ids(args.gpus)
    paths = discover_inputs(input_root)

    skipped_existing = 0
    if args.skip_existing:
        pending = []
        for path in paths:
            if output_path_for(input_root, output_root, path).exists():
                skipped_existing += 1
            else:
                pending.append(path)
        paths = pending
    if not paths:
        print(f"nothing to process; skipped {skipped_existing} existing motions")
        return

    with ThreadPoolExecutor(max_workers=args.scan_workers) as executor:
        specs = list(
            executor.map(
                lambda path: inspect_motion(path, input_key=args.input_key, fallback_fps=args.fps),
                paths,
            )
        )
    shards, frame_loads = balanced_shards(specs, len(gpu_ids))

    cluster_dir = output_root / "_cluster"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    worker_script = Path(__file__).with_name("process_isaaclab_pos36.py").resolve()
    worker_specs: list[dict[str, Any]] = []
    for gpu_id, shard, frame_count in zip(gpu_ids, shards, frame_loads):
        if not shard:
            continue
        manifest = cluster_dir / f"manifest.gpu{gpu_id}.txt"
        manifest.write_text("".join(f"{spec.path.absolute()}\n" for spec in shard), encoding="utf-8")
        log_path = cluster_dir / f"worker.gpu{gpu_id}.log"
        progress_path = cluster_dir / f"progress.gpu{gpu_id}.json"
        summary_name = Path("_cluster") / f"summary.gpu{gpu_id}.json"
        command = [
            sys.executable,
            str(worker_script),
            "--input", str(input_root),
            "--manifest", str(manifest),
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
