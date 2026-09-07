#!/usr/bin/env python
"""Relight an image sequence using the D-Rex DiffusionRenderer model.

Default (single-chunk) mode
---------------------------
Relights the first 57 frames as one chunk and writes them directly as
``frame_{i:06d}.png`` — ready to encode with encode_video.py, no merge step
needed.

    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=.:cosmos_transfer1 \\
        python scripts/relight.py \\
            --source_dir  workspace/frames/ \\
            --envmap_path envmaps/target.hdr \\
            --checkpoint  checkpoints/subject.pt \\
            --output_dir  workspace/relit/

Long-form mode (--long)
-----------------------
Processes the full sequence in overlapping temporal chunks (default T=57,
overlap=32).  Outputs use the ``{chunk:04d}.{frame:04d}.relit.png`` naming
convention and must be blended with merge_chunks.py afterwards.

    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=.:cosmos_transfer1 \\
        python scripts/relight.py --long \\
            --source_dir  workspace/frames/ \\
            --envmap_path envmaps/target.hdr \\
            --checkpoint  checkpoints/subject.pt \\
            --output_dir  workspace/chunks/
"""

import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

from cosmos_predict1.utils import misc

from drex.config import load_config, TrainerConfig
from drex.inference import BatchedImageDirLoader, BatchedEnvLoaderSingle
from drex.utils.model import load_model_for_inference
from drex.utils.io import save_image_tensor_to_jpg
import drex.utils.batch as batch_utils


def parse_args():
    p = argparse.ArgumentParser(
        description="Relight an image sequence with D-Rex DiffusionRenderer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--source_dir",  required=True, help="Input directory of PNG frames")
    p.add_argument("--envmap_path", required=True, help="Path to HDR environment map")
    p.add_argument("--output_dir",  required=True, help="Output directory")

    p.add_argument("--checkpoint",      default=None,   help="Fine-tuned LoRA checkpoint (.pt)")
    p.add_argument("--checkpoint_type", default="lora", choices=["lora", "zero-shot"])
    p.add_argument("--checkpoint_dir", default="cosmos_transfer1/checkpoints",
                   help="Base model weights directory")
    p.add_argument("--lora_cfg", default="configs/lora.yaml",
                   help="YAML with lora_rank / lora_scale / lora_first_nblocks")

    p.add_argument("--trajectory_file", default=None,
                   help="JSON trajectory for per-frame env-map poses")
    p.add_argument("--fixed_frame", type=int, default=-1,
                   help="Fix input to this single frame index")

    p.add_argument("--num_steps",   type=int, default=15, help="Diffusion steps (default: 15)")
    p.add_argument("--start_frame", type=int, default=0,  help="Skip leading frames")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--dtype",       default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])

    # ── Long-form options (only relevant with --long) ─────────────────────────
    p.add_argument("--long", action="store_true",
                   help="Process the full sequence in overlapping chunks (requires merge_chunks.py afterwards)")
    p.add_argument("--chunk_size", type=int, default=57,
                   help="[--long] Frames per chunk, must be a multiple of 57 (default: 57)")
    p.add_argument("--overlap",    type=int, default=32,
                   help="[--long] Overlap between consecutive chunks (default: 32)")
    p.add_argument("--max_frames", type=int, default=10_000_000,
                   help="[--long] Stop after this many source frames")

    return p.parse_args()


def run_chunk(model, img_loader, env_loader, args, out_dir, chunk_id, long_form):
    """Run one chunk and save output frames. Returns False when no more chunks."""
    found, loader_ret = img_loader.next_chunk()
    if not found:
        return False

    frame_ids  = loader_ret["frame_ids"]
    data_batch = loader_ret["data_batch"]
    envs       = env_loader.load_envmaps(frame_ids)

    data_batch["fps"]     = data_batch["fps"][0]
    data_batch["env_ldr"] = envs["env_ldr"]
    data_batch["env_log"] = envs["env_log"]
    data_batch["env_nrm"] = envs["env_nrm"]

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    with torch.no_grad():
        video = batch_utils.run_custom_diffrender_model(
            model, data_batch, num_steps=args.num_steps, seed=args.seed,
        )
        video = ((1.0 + video).clamp(0, 2) / 2).float()

        for fid in range(video.shape[2]):
            if long_form:
                fname = out_dir / f"{chunk_id:04d}.{fid:04d}.relit.png"
            else:
                fname = out_dir / f"frame_{fid:06d}.png"
            save_image_tensor_to_jpg(video, fname, frame_idx=fid)

    return True


def main():
    args = parse_args()
    random.seed(args.seed)
    misc.set_random_seed(args.seed)

    source_dir = Path(args.source_dir)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  mode        : {'long-form (overlapping chunks)' if args.long else 'single chunk (57 frames)'}")
    print(f"  source      : {source_dir}")
    print(f"  envmap      : {args.envmap_path}")
    print(f"  output      : {out_dir}")
    print(f"  checkpoint  : {args.checkpoint}  [{args.checkpoint_type}]")
    print(f"  steps       : {args.num_steps}")
    print("=" * 60)

    # Auto-detect trajectory JSON
    traj_file = args.trajectory_file
    if traj_file is None:
        candidate = source_dir / "traj.json"
        if candidate.exists():
            print(f"[INFO] Auto-detected trajectory: {candidate}")
            traj_file = str(candidate)
        else:
            print("[INFO] No trajectory file — static env-map pose.")

    # ── Model ─────────────────────────────────────────────────────────────────
    lora_cfg = load_config(args.lora_cfg, TrainerConfig)
    lora_cfg.checkpoint_dir = args.checkpoint_dir

    model = load_model_for_inference(
        checkpoint_type = args.checkpoint_type,
        checkpoint_dir  = args.checkpoint_dir,
        checkpoint_path = args.checkpoint if args.checkpoint_type != "zero-shot" else None,
        lora_config     = lora_cfg.build_lora_config() if args.checkpoint_type == "lora" else None,
        device          = args.device,
        dtype           = {"float32": torch.float32, "float16": torch.float16,
                           "bfloat16": torch.bfloat16}[args.dtype],
    )
    model.eval()
    print("[INFO] Model ready.")

    # ── Loaders ───────────────────────────────────────────────────────────────
    if args.long:
        chunk_size = args.chunk_size
        overlap    = args.overlap
    else:
        chunk_size = 57
        overlap    = 0

    img_loader = BatchedImageDirLoader(
        source_dir,
        T_batch_size = chunk_size,
        start_frame  = args.start_frame,
        overlap      = overlap,
        fixed_frame  = args.fixed_frame,
    )
    env_loader = BatchedEnvLoaderSingle(args.envmap_path, trajectory_file=traj_file)

    # ── Inference ─────────────────────────────────────────────────────────────
    if not args.long:
        # Single chunk: process exactly one chunk and finish
        ok = run_chunk(model, img_loader, env_loader, args, out_dir, 0, long_form=False)
        if not ok:
            print(f"[ERROR] Not enough frames in {source_dir} for a single chunk of 57.")
        else:
            print(f"[INFO] Done — 57 frames written to {out_dir}")
    else:
        # Long-form: iterate all available chunks
        n = len(sorted(source_dir.glob("*.png"))) - args.start_frame
        total = max(0, 1 + (n - chunk_size) // (chunk_size - overlap)) if n >= chunk_size else 0
        print(f"[INFO] ~{total} chunks  (T={chunk_size}, overlap={overlap})")

        chunk_id = 0
        with tqdm(total=total, desc="Relighting") as pbar:
            while True:
                frame_start = img_loader.chunk_pointer
                if frame_start > args.max_frames:
                    break
                ok = run_chunk(model, img_loader, env_loader, args, out_dir, chunk_id, long_form=True)
                if not ok:
                    break
                chunk_id += 1
                pbar.update(1)

        print(f"\n[INFO] Done — {chunk_id} chunks written to {out_dir}")
        print("[INFO] Run merge_chunks.py to blend overlapping regions.")


if __name__ == "__main__":
    main()
