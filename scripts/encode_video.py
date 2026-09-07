#!/usr/bin/env python
"""Encode a directory of PNG frames into an MP4 video.

Frames are read in lexicographic order.  The ffmpeg filter pads to even
dimensions (required by libx264) before encoding.

Usage
-----
    python scripts/encode_video.py \\
        --frames_dir  workspace/merged/ \\
        --output      result.mp4 \\
        [--fps        24] \\
        [--pattern    "*.png"]   # glob used to find frames
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Encode PNG frames to MP4")
    p.add_argument("--frames_dir", required=True, help="Directory containing PNG frames")
    p.add_argument("--output",     required=True, help="Output MP4 path")
    p.add_argument("--fps",        type=float, default=24.0, help="Frame rate (default: 24)")
    p.add_argument("--pattern",    default="*.png", help="Glob pattern for frames (default: *.png)")
    p.add_argument("--crf",        type=int,   default=18,
                   help="libx264 CRF quality (0=lossless, 51=worst; default: 18)")
    p.add_argument("--gamma",      type=float, default=1.0,
                   help="Gamma correction (>1 brightens, <1 darkens; default: 1.0 = no change)")
    return p.parse_args()


def main():
    args = parse_args()

    frames_dir = Path(args.frames_dir)
    frames     = sorted(frames_dir.glob(args.pattern))

    if not frames:
        print(f"[ERROR] No frames found in {frames_dir} matching '{args.pattern}'", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Encoding {len(frames)} frames → {out_path}  (fps={args.fps}, gamma={args.gamma})")

    # Symlink frames into a temp dir with sequential numbering for ffmpeg
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            (Path(tmp) / f"{i:06d}.png").symlink_to(f.resolve())

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-r", str(args.fps),
            "-i", f"{tmp}/%06d.png",
            "-vf", f"eq=gamma={args.gamma},pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264",
            "-crf", str(args.crf),
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]

        result = subprocess.run(cmd)

    if result.returncode != 0:
        print("[ERROR] ffmpeg failed.", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"[INFO] Done — {out_path}")


if __name__ == "__main__":
    main()
