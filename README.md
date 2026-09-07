# DRexInference — D-Rex Relighting Demo

Minimal self-contained demo for relighting image sequences with D-Rex
(Diffusion Renderer + LoRA fine-tuning on [Cosmos DiffusionRenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer)).

All required data (base weights, subject LoRA checkpoints, capture data,
environment maps) is fetched from public/external sources — see
[Data & checkpoints](#data--checkpoints) below.

## Setup

### 1. Clone and initialise

```bash
git clone <this-repo> DRexInference
cd DRexInference
bash setup.sh          # initialises submodule + applies patches
```

### 2. Create the conda environment

The environment is defined by the submodule.  Run these commands from the
repo root (the `cosmos_transfer1/` submodule must already be checked out):

```bash
# Create and activate the environment
conda env create --file cosmos_transfer1/cosmos-predict1.yaml
conda activate cosmos-predict1

# Install pip dependencies
pip install -r cosmos_transfer1/requirements.txt

# Patch Transformer Engine linking issues in conda environments
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10

# Install Transformer Engine
pip install transformer-engine[pytorch]==1.12.0
```

> **Note**: inference only — the `apex` package required for training is not needed here.

### 3. Base model weights

The pretrained weights are read from the path set in `configs/lora.yaml`
(`checkpoint_dir`) or overridden via `--checkpoint_dir` at runtime.
Base model weights must be placed (or symlinked) at:

```
cosmos_transfer1/checkpoints/
├── Diffusion_Renderer_Forward_Cosmos_7B/model.pt
└── Cosmos-Tokenize1-CV8x8x8-720p/          (tokenizer files)
```

They are published by NVIDIA on Hugging Face (~56 GB) and can be fetched with
the downloader script that ships with the submodule (see the [upstream
README](cosmos_transfer1/README.md#download-model-weights-56gb) for details):

```bash
huggingface-cli login    # needs a HF access token, "Read" permission
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=cosmos_transfer1 \
python cosmos_transfer1/scripts/download_diffusion_renderer_checkpoints.py \
    --checkpoint_dir cosmos_transfer1/checkpoints
```

### 4. Subject LoRA checkpoints and example data

See [Data & checkpoints](#data--checkpoints) below.

## End-to-end pipeline

### 1. Extract frames from source video

```bash
python scripts/extract_frames.py \
    --input   source.mp4 \
    --output  workspace/frames/ \
    --width   1280 --height 704    # model resolution
    # --start 0 --end 299          # optional frame range
```

Frames are scaled to fit 1280×704, padded with black on both sides, saved as
`frame_{i:06d}.png`.

### 2. Relight

By default `relight.py` processes a single 57-frame chunk and writes
`frame_{i:06d}.png` directly, ready for `encode_video.py` — no merge step
needed:

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=.:cosmos_transfer1 \
python scripts/relight.py \
    --source_dir  workspace/frames/ \
    --envmap_path envmaps/examples/colored_dot.hdr \
    --checkpoint  checkpoints/subject01_lora.pt \
    --output_dir  workspace/relit/
```

For sequences longer than 57 frames, pass `--long` to process the full
sequence in overlapping temporal chunks (output uses the
`{chunk:04d}.{frame:04d}.relit.png` naming convention and must be blended
with `merge_chunks.py` afterwards):

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=.:cosmos_transfer1 \
python scripts/relight.py --long \
    --source_dir  workspace/frames/ \
    --envmap_path envmaps/examples/colored_dot.hdr \
    --checkpoint  checkpoints/subject01_lora.pt \
    --output_dir  workspace/chunks/
```

Zero-shot (no fine-tuning) works the same way, just omit `--checkpoint` and
set `--checkpoint_type zero-shot`.

Key options:

| Flag | Default | Description |
|---|---|---|
| `--checkpoint_type` | `lora` | `lora` / `zero-shot` |
| `--checkpoint_dir`  | `cosmos_transfer1/checkpoints` | Base model weights dir |
| `--lora_cfg`        | `configs/lora.yaml` | LoRA rank/scale settings |
| `--long`            | off | Process the full sequence in overlapping chunks |
| `--chunk_size`      | `57` | `--long` only: frames per chunk (must be a multiple of 57) |
| `--overlap`         | `32` | `--long` only: overlap between chunks |
| `--num_steps`       | `15` | Diffusion steps |
| `--trajectory_file` | auto-detect `traj.json` | Per-frame env-map poses (JSON) |

### 3. Merge overlapping chunks (only needed after `--long`)

```bash
python scripts/merge_chunks.py \
    --input_dir   workspace/chunks/ \
    --output_dir  workspace/merged/ \
    --t_batch_size 57 \
    --overlap      32
```

### 4. Encode final video

```bash
python scripts/encode_video.py \
    --frames_dir  workspace/merged/ \
    --output      result.mp4 \
    --fps         24
```

## Checkpoint types

- **`lora`** — LoRA fine-tuned checkpoint (default). Requires `--checkpoint`.
- **`zero-shot`** — Base pretrained model, no fine-tuning. Omit `--checkpoint`.

## Data & checkpoints

### Example environment maps

Two example environment maps ship in
[`envmaps/examples/`](envmaps/examples/) so you can try the pipeline without
downloading anything else:

- `colored_dot.hdr` — colored point light on a dark background
- `white_dot.hdr` — single white point light on a dark background

A larger environment-map library is published at
**https://gvv-assets.mpi-inf.mpg.de/rhc/** alongside the EVA/GT data (see
below).

### EVA sequences and GT captures

Expressive video-avatar (EVA) driving sequences and the corresponding 1K
ground-truth multi-view captures are served at:

**https://gvv-assets.mpi-inf.mpg.de/rhc/** (registration required)

This portal also hosts the broader RHC dataset's subjects, not just the D-Rex
subset. **All consent/subject notes in this README apply only to the D-Rex
subjects (Subject01, Subject03)** — other subjects on the portal are
unrelated to this release.

Only Subject01 and Subject03 are published for D-Rex — **Subject02 and
Subject04 did not consent to publication** and are withheld in their
entirety (not just their LoRA checkpoints; see
[Subject LoRA checkpoints](#subject-lora-checkpoints) below). Each published
archive contains, per subject:

```
SubjectNN/
├── EVA/            per-camera driving sequences (+ per-stream coverage files)
├── CAPTURE_DS/      1K ground-truth capture frames
└── shared/
    └── cameras.calib   camera calibration, required for inference
```

### Environment maps used for training (per-frame mapping)

Each frame used to fine-tune the released checkpoints is paired with an
environment map from a shared library of 1015 HDRIs (`indoors_{i:06d}.hdr`),
published as `envs/train/output_hdr/` alongside the EVA/GT data (use the
`train` split; `test` is an unrelated set of HDRIs). The published archive's
`Frame_{frame_id:06d}/EnvMapSampled.hdr` files follow this convention byte
for byte — matching what
[`BatchedEnvLoaderDataset`](drex/inference/loaders.py) expects — so a
frame's envmap can be looked up directly.

To find the envmap used for a given frame, compute its index into the
library:

```
env_idx = (frame_id - 1 + env_offset + 1015 * 5) % 1015
```

then look up `envs/train/output_hdr/indoors_{env_idx:06d}.hdr`.
`env_offset` is fixed per subject (chosen to decorrelate the lighting
sequence seen by each subject during training):

| Subject | `env_offset` |
|---|---|
| Subject01 | 190 |
| Subject03 | 0 |

For example, frame 100 of Subject01 (`env_idx = (100 - 1 + 190 + 1015*5) %
1015 = 289`) uses `envs/train/output_hdr/indoors_000289.hdr`.

This mapping is unrelated to the `relight.py` demo path above, which applies
one user-supplied envmap (optionally reprojected per frame via
`--trajectory_file`), not a different HDRI per frame.

### Subject LoRA checkpoints

Only Subject01 and Subject03 consented to publishing their fine-tuned
checkpoints (see above) — Subject02/04 must not be redistributed.

A step-15,000 demo checkpoint (not the fully-converged final checkpoint) for
each subject is bundled in this repo, ready to use out of the box:

| Subject | File | Size | Notes |
|---|---|---|---|
| Subject01 | [`checkpoints/subject01_lora.pt`](checkpoints/subject01_lora.pt) | ~66.7 MB | step 15,000 |
| Subject03 | [`checkpoints/subject03_lora.pt`](checkpoints/subject03_lora.pt) | ~66.7 MB | step 15,000; one camera missing from the capture rig |

Fully-converged final checkpoints (step 59,500 for Subject01, step 66,500
for Subject03) give better relighting quality and are recommended beyond a
quick demo — hosting isn't finalized yet, check back here or watch the
release announcement for a download link. Full training history (saved
every 500 steps, ~4.5–5 GB per subject) is only needed for
resuming/inspecting training, not for running this demo.

### Datasheet (short)

- **Subjects**: 4 captured (Subject01–04); 2 published (Subject01,
  Subject03) under explicit consent. Subject02/04 did not consent and are
  withheld entirely — not merely excluded from the checkpoint release.
- **Modality**: multi-view video capture, 1K ground truth, synchronized
  per-stream coverage metadata, camera calibration.
- **Derived data**: per-subject EVA driving sequences; per-subject LoRA
  fine-tunes of the Cosmos DiffusionRenderer forward model (rank 8, first
  28 transformer blocks).
- **Consent scope**: per-subject, covering the published EVA/GT captures
  and, where noted above, the fine-tuned LoRA checkpoint. Unreleased
  subjects' data/checkpoints must not be redistributed.
- **Known caveats**: Subject03's capture is missing one camera stream.
- **Access**: EVA/GT/envmap data via https://gvv-assets.mpi-inf.mpg.de/rhc/
  (registration required); LoRA checkpoints distributed separately (see
  [EVA sequences and GT captures](#eva-sequences-and-gt-captures) above for
  subject-scope details).

## License and Attribution

Built on [NVIDIA Cosmos](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer).

- **This repository's code** (`drex/`, `scripts/`, `configs/`, docs) is
  released under the [Apache License 2.0](LICENSE).
- **The `cosmos_transfer1` submodule** (NVIDIA's Cosmos-Transfer1-DiffusionRenderer)
  is separately licensed under the
  [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) (fetched
  at setup time, not vendored here) — see
  [`cosmos_transfer1/LICENSE`](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer/blob/main/LICENSE)
  and [`cosmos_transfer1/ATTRIBUTIONS.md`](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer/blob/main/ATTRIBUTIONS.md)
  after running `setup.sh`. `patches/cosmos_transfer1.patch` documents our
  modifications to that source, per Apache 2.0 §4(b).
- **Base model weights and the bundled subject LoRA checkpoints**
  (`checkpoints/subject01_lora.pt`, `checkpoints/subject03_lora.pt`) are
  Derivative Models of NVIDIA's Diffusion Renderer Cosmos weights, licensed
  by NVIDIA Corporation under the
  [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
  Using them means agreeing to that license, including its Trustworthy-AI /
  safety-guardrail terms.

If you find this work useful, please cite:

```bibtex
@inproceedings{teufel2026drex,
    author    = {Timo Teufel and Xilong Zhou and Umar Iqbal and Jan Kautz and
        Marc Habermann and Vladislav Golyanik and Christian Theobalt},
    title     = {D-Rex: Diffusion Rendering for Relightable Expressive Avatars},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2026}
}
```

Please also cite the base DiffusionRenderer paper this pipeline builds on:

```bibtex
@inproceedings{DiffusionRenderer,
    author = {Ruofan Liang and Zan Gojcic and Huan Ling and Jacob Munkberg and
        Jon Hasselgren and Zhi-Hao Lin and Jun Gao and Alexander Keller and
        Nandita Vijaykumar and Sanja Fidler and Zian Wang},
    title = {DiffusionRenderer: Neural Inverse and Forward Rendering with Video Diffusion Models},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2025}
}
```

If you make use of the expressive multi-view data in your project, please also cite:

```bibtex
@InProceedings{singh2025rhc,
    title     = {Relightable Holoported Characters: Capturing and Relighting Dynamic Human Performance from Sparse Views},
    author    = {Singh, Kunwar Maheep and Chen, Jianchun and Golyanik, Vladislav and Garbin, Stephan J. and Beeler, Thabo and Dabral, Rishabh and Habermann, Marc and Theobalt, Christian},
    booktitle = {CVPR},
    year      = {2026}
}
```

## Submodule patches

`patches/cosmos_transfer1.patch` contains modifications to the upstream
`cosmos-transfer1-diffusion-renderer` submodule required for training and
inference compatibility.  `setup.sh` applies them automatically.
To re-generate the patch after making further changes inside `cosmos_transfer1/`:
```bash
cd cosmos_transfer1
git diff HEAD > ../patches/cosmos_transfer1.patch
```

---

Claude was used in the creation of this repository.
