# GPU Control Panel — Design Spec

Date: 2026-06-07

## Purpose

A Flask app that lets the user deploy and run the full LoRA training/generation
pipeline on a rented GPU box, targeting **E2E Networks** (India-based, UPI/INR
billing, standard SSH on port 22 + security groups, pre-built PyTorch
container templates — see Deployment target, below). It replaces manually
SSHing in and typing long `sdxl_train_network.py` / `sdxl_gen_img.py` commands
with a web control panel: upload/select a dataset, point at a base model,
queue training and generation jobs, watch logs live, and download results —
all without leaving the browser.

Built API-first; the web UI is a thin layer that calls the same JSON API.

## Non-goals

- Not a multi-user system — single shared password, single operator.
- Not a general-purpose ML platform — it wraps the specific Kohya
  `sdxl_train_network.py` / `sdxl_gen_img.py` SDXL workflows for this project's
  manhwa/webtoon LoRA, not arbitrary training scripts or other model families
  (the SD1.5 path validated in the local smoke test (`train_smoke.log` /
  `test_gen.log`) was a pipeline proof-of-concept; the production target is
  SDXL — see Model choice, below).
- No literal pause/resume of a *running* subprocess (see Job queue below) —
  mid-epoch process state can't be safely suspended and restored.

## Model choice

The base model is an **SDXL-family checkpoint — Illustrious-XL (or a close
variant such as WAI-Illustrious / NoobAI-XL)** rather than the SD1.5 checkpoint
used in the local smoke test. Illustrious-class models are purpose-built for
high-quality anime/illustration output and are the most common base for
manhwa/webtoon-style LoRAs, and **832×1216 — the user's panel aspect ratio —
is one of SDXL's native bucket resolutions**, a much better match than SD1.5's
~512×768 ceiling.

Practical implications for the control panel:
- Training and generation use Kohya's **SDXL-specific scripts**
  (`sdxl_train_network.py`, `sdxl_gen_img.py`) and argument set — these differ
  from the `train_network.py` / `gen_img.py` invocations validated locally
  (e.g. dual text encoders requiring `--cache_text_encoder_outputs`, no
  `--v2`, `--no_half_vae` to avoid SDXL VAE fp16 NaNs).
- Illustrious-class checkpoints are typically distributed via **Civitai**
  rather than Hugging Face, so the `download_model` job takes a plain `url`
  (a direct download link) — not an HF-specific repo-id lookup.
- SDXL training is significantly more VRAM-hungry than SD1.5 — the rented box
  needs a GPU with materially more headroom than the local 6GB 3060 (an
  A100 40GB on E2E Networks is the planned target tier).

## Storage layout

```
work/
  app.db                      # SQLite: jobs, dataset registry, model registry
  datasets/<name>/            # extracted uploaded datasets (Kohya <repeats>_<trigger> folders)
  models/<name>.safetensors   # base models, downloaded server-side by direct URL (e.g. Civitai)
  jobs/<job_id>/
    log.txt                   # combined stdout/stderr from the subprocess, tailed for the UI
    output/                   # LoRA checkpoint(s) or generated images, depending on job type
```

## Job model & queue

Each job is a row in SQLite:

```
id, type (train | generate | download_model),
status (queued | running | cancelled | done | failed),
params (JSON), order_index, pid, created_at
```

A single background worker thread polls SQLite for the next `queued` job (by
`order_index`), launches it with `subprocess.Popen`, streams combined
stdout/stderr into `jobs/<id>/log.txt`, and updates status to `running` →
`done` / `failed` based on the exit code.

- **Queued jobs** can be reordered (move up/down → updates `order_index`) or
  removed from the queue entirely.
- **Running jobs** can be **cancelled**: the runner kills the subprocess
  (and its child tree, since `accelerate`/`torch` spawn children), marks the
  job `cancelled`, and **keeps whatever output was written so far** — e.g. an
  epoch checkpoint saved before cancellation remains usable. The UI makes clear
  that "pause" only applies to queued jobs (i.e. "don't start this yet"); a
  running job can only be cancelled, not suspended.
- Model downloads are modeled as a `download_model` job so their progress
  shows in the same queue/history view as training and generation.

## API surface

All routes except `/api/login` require an authenticated session
(see Auth, below).

**Auth**
- `POST /api/login` — checks password against `CONTROL_PANEL_PASSWORD`
  (env var / config), sets a signed session cookie on success.

**Files & models**
- `POST /api/datasets` — upload a `.zip`; server extracts to
  `work/datasets/<name>/` and validates it contains
  `<repeats>_<trigger>`-style folders (Kohya convention) before accepting it.
  This is exactly the shape produced by the local labeling kit's
  `run.py package` stage (`tag → review → emit → package`), which bundles
  auto-tagged, human-reviewed image+caption pairs into a `<repeats>_<trigger>`
  folder and zips it — so the hand-off from local labeling to remote training
  is "run `package`, drag the zip into Upload Dataset," no manual reshuffling.
- `GET /api/datasets` — list available datasets.
- `POST /api/models/download` — body `{url, filename}`; enqueues a
  `download_model` job that streams the file (e.g. a Civitai direct-download
  link for an Illustrious-XL checkpoint) into `work/models/`.
- `GET /api/models` — list downloaded base models.
- `GET /api/checkpoints` — list LoRA `.safetensors` produced by past training
  jobs, with download links.

**Jobs**
- `POST /api/jobs` — create a job. Body shape depends on `type`:
  - `train`: dataset, base model, output name, resolution, epochs,
    network_dim/alpha, learning rate, min_snr_gamma, etc. — fields pre-filled
    with the SDXL/anime LoRA defaults described in Execution & environment,
    below.
  - `generate`: base model, LoRA checkpoint(s), prompt, negative prompt,
    sampler params, batch size.
- `GET /api/jobs` — list all jobs with status/order (the queue/history view).
- `GET /api/jobs/<id>` — job detail: params, status, output file list.
- `GET /api/jobs/<id>/log?since=<byte_offset>` — incremental log tail for
  polling-based live log viewing.
- `POST /api/jobs/<id>/cancel` — kill the running subprocess tree, mark
  `cancelled`, keep partial output.
- `POST /api/jobs/<id>/reorder` — move a *queued* job up or down.
- `GET /api/jobs/<id>/output/<filename>` — download a checkpoint or
  generated image from `jobs/<id>/output/`.

## Web UI (thin layer over the API)

- **Login** page.
- **Dashboard**: queue/history table (type, status, latest log line as a
  progress hint, [Cancel] / [↑↓ reorder] for queued jobs) plus action buttons:
  "New Training Job", "New Generation Job", "Download Base Model",
  "Upload Dataset".
- **New Training Job** form: dataset and base model pickers populated from
  `/api/datasets` and `/api/models`; numeric fields pre-filled with the
  SDXL/anime LoRA defaults (dim/alpha 32/32, lr 3e-5, resolution 1024 +
  bucketing, min_snr_gamma 5, ~30 epochs) — all overridable.
- **New Generation Job** form: base model + LoRA checkpoint picker(s),
  prompt / negative prompt / sampler fields.
- **Job detail** page: live-tailing log viewer (polls `/log?since=`); for
  `generate` and `download_model` jobs, a gallery/file list of `output/` with
  download links.

## Auth

Single shared password gate: `CONTROL_PANEL_PASSWORD` set via env var or
config file. `/api/login` checks it and sets a signed Flask session cookie;
a `before_request` hook redirects unauthenticated requests (except
`/api/login` and the login page) to the login form. This is enough to stop
randoms who stumble onto the exposed port/proxy URL from touching the GPU box
or its files — it is not meant to support multiple distinct users.

## Deployment target

**E2E Networks** (e2enetworks.com), chosen because it bills in INR with **UPI
support** (matches the user's payment method) and uses a standard SSH model:

- Launch a GPU node from E2E's **TIR platform** using their pre-built
  **PyTorch container template** — CUDA/PyTorch already configured, so no
  repeat of the local driver/venv setup pain.
- Reach it over **SSH on port 22** (security group must allow it); the Flask
  app's port is reached the same way RunPod/Vast.ai-style boxes work — via an
  SSH tunnel or an exposed port through the security group.
- Planned GPU tier: **A100 40GB** (~₹180-250/hr) for real SDXL training runs —
  SDXL needs materially more VRAM than the SD1.5 smoke test used locally on
  the 6GB 3060. Cheaper tiers (e.g. L4) are an option for running the control
  panel itself or for generation-only sessions between training runs.
- Persistent storage: E2E's containers are container-native, so the app's
  `work/` directory (datasets, models, job outputs, `app.db`) must live on a
  mounted persistent volume, not the container's ephemeral filesystem —
  confirmed during deployment setup, not assumed here.

## Execution & environment

The job runner builds the CLI invocation for **`sdxl_train_network.py`** /
**`sdxl_gen_img.py`** from the job's stored params. Defaults are pre-filled
from community-validated SDXL/anime LoRA presets (not the SD1.5 smoke-test
numbers, which only proved the pipeline plumbing):

```bash
python sd-scripts/sdxl_train_network.py \
  --pretrained_model_name_or_path <illustrious_xl_checkpoint> \
  --train_data_dir <dataset_dir> \
  --output_dir <job_output_dir> --output_name <lora_name> \
  --resolution 1024,1024 --enable_bucket --bucket_reso_steps 64 \
    --min_bucket_reso 256 --max_bucket_reso 2048 \
  --train_batch_size 4 --max_train_epochs <epochs> --save_every_n_epochs 1 \
  --network_module networks.lora --network_dim 32 --network_alpha 32 \
  --learning_rate 3e-5 --min_snr_gamma 5 \
  --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk \
  --no_half_vae --mixed_precision fp16 --gradient_checkpointing --sdpa \
  --caption_extension .txt --save_model_as safetensors
```

Pre-filled defaults for the **New Training Job** form: `network_dim`/`alpha`
**32/32**, learning rate **3e-5** (try ~5e-5 for stronger anime-style
transfer), resolution **1024 base with bucketing** (matches the user's
832×1216 panels as a native SDXL bucket), `min_snr_gamma` **5**, ~**30
epochs** as a starting point. These are starting points the user will tune
once real runs are underway — the form lets every value be overridden.

The runner sets `cwd` to the sd-scripts checkout and uses the venv's `python`.
Linux rental boxes shouldn't need the Windows-only env vars from the local
smoke test (`PYTHONUTF8`, `PYTHONIOENCODING`, `OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS`) — those worked around cp1252 console encoding and
OpenBLAS/RAM pressure issues specific to the Windows laptop used for local
testing — but the runner applies them conditionally via `platform.system()`
since they're harmless no-ops on Linux.

## Error handling

- Subprocess exits non-zero → job marked `failed`; `log.txt` retains the
  traceback for display in the job detail view.
- Uploads that aren't valid zips, or don't contain Kohya-style
  `<repeats>_<trigger>` folders, are rejected with a clear error before
  extraction — nothing partially-written is left behind.
- Model download failures (bad URL/repo id, disk full, network error) are
  caught by the runner and surface as a `failed` `download_model` job rather
  than crashing the worker thread.

## Testing plan

Before deploying to the rental box, smoke-test the whole flow locally:

1. Log in with the shared password.
2. Upload a small dataset zip — produced via `run.py package --img panels
   --trigger mywebtoon --repeats 5 --out work` on a small subset (e.g. the
   existing 15-pair test dataset), which yields `work/5_mywebtoon.zip` in
   exactly the shape the upload endpoint expects.
3. Use the "Download Base Model" flow to fetch a small/quick-to-grab SDXL
   checkpoint first (to validate the `download_model` job end-to-end without
   waiting on a multi-GB Illustrious-XL download), then separately validate
   against the real Illustrious-XL checkpoint before any production run.
4. Queue a training job using `sdxl_train_network.py` with a short
   `max_train_epochs` (e.g. 1) purely to prove the job runs end-to-end and
   produces a `.safetensors` checkpoint in `jobs/<id>/output/` — this is a
   pipeline smoke test, not a quality run (mirrors how the local SD1.5 100-step
   run validated plumbing, not output quality).
5. Queue a generation job (`sdxl_gen_img.py`) against that checkpoint, confirm
   images appear in `jobs/<id>/output/` and are downloadable.
6. Exercise the queue controls: queue two jobs, reorder them, cancel a running
   one, and confirm partial output is retained.

Once this passes — first locally if VRAM allows a tiny SDXL run, otherwise
directly on a rented E2E box — the same app (with its venv) is what gets
deployed to the E2E Networks GPU node for real training runs.
