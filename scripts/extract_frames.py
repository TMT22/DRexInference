#!/usr/bin/env python
"""Extract frames from an MP4 and resize/pad them to the model's expected resolution.

Source videos of any aspect ratio are scaled down to fit inside the target
resolution (preserving aspect ratio), then padded symmetrically with black on
both sides (pillarbox or letterbox) to reach the exact target size.

Default target: 1280 × 704 px  (W × H, landscape, matches the ERA dataset)

Usage
-----
    python scripts/extract_frames.py \\
        --input   source.mp4 \\
        --output  workspace/frames/ \\
        [--width  1280] \\
        [--height 704] \\
        [--start  0] \\
        [--end    299]    # inclusive; omit for all frames
        [--fps    24]     # output frame rate (pass 0 to keep source fps)

Outputs files named frame_{i:06d}.png (0-indexed, from --start).
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Extract and resize frames from MP4")
    p.add_argument("--input",  required=True, help="Input MP4 file")
    p.add_argument("--output", required=True, help="Output directory for PNG frames")
    p.add_argument("--width",  type=int, default=1280, help="Target width  (default: 1280)")
    p.add_argument("--height", type=int, default=704,  help="Target height (default: 704)")
    p.add_argument("--start",  type=int, default=0,    help="First frame index to extract (default: 0)")
    p.add_argument("--end",    type=int, default=None, help="Last frame index (inclusive); omit = all")
    p.add_argument("--fps",    type=float, default=0,
                   help="Output frame rate (0 = keep source fps, default: 0)")
    return p.parse_args()


def main():
    args = parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"[ERROR] Input not found: {src}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    W, H = args.width, args.height

    # Build ffmpeg filter:
    # 1. scale to fit inside W×H preserving aspect ratio
    # 2. pad symmetrically to exactly W×H
    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black"
    )

    # Frame selection via trim filter
    # ffmpeg select= is 0-indexed; we use pts-based select for reliability
    select_parts = []
    if args.start > 0:
        select_parts.append(f"gte(n\\,{args.start})")
    if args.end is not None:
        select_parts.append(f"lte(n\\,{args.end})")

    if select_parts:
        select_expr = "*".join(select_parts)
        vf = f"select='{select_expr}'," + vf
        # When using select we need vsync=0 to avoid duplicate frames
        vsync = ["-vsync", "0"]
    else:
        vsync = []

    # Output filename: 0-indexed from the start frame
    out_pattern = str(out_dir / "frame_%06d.png")

    # Subtract start offset so first output file is frame_000000.png
    start_number = args.start

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
    ]
    if args.fps > 0:
        cmd += ["-r", str(args.fps)]
    cmd += [
        "-vf", vf,
        "-start_number", str(start_number),
        *vsync,
        out_pattern,
    ]

    print(f"[INFO] Target resolution: {W}×{H}")
    print(f"[INFO] Frame range: {args.start} → {args.end if args.end is not None else 'end'}")
    print(f"[INFO] Output: {out_dir}")
    print(f"[CMD]  {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[ERROR] ffmpeg failed.", file=sys.stderr)
        sys.exit(result.returncode)

    written = sorted(out_dir.glob("frame_*.png"))
    print(f"\n[INFO] Done — {len(written)} frames written to {out_dir}")


if __name__ == "__main__":
    main()
