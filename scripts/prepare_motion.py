#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fk_compare.motion import prepare_motion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-fps", type=float, default=120.0)
    parser.add_argument("--output-fps", type=float, default=50.0)
    args = parser.parse_args()
    prepare_motion(args.input, args.output, args.input_fps, args.output_fps)
    print(f"Wrote canonical motion to {args.output}")


if __name__ == "__main__":
    main()
