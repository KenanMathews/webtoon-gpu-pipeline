# GPU Control Panel — Design Spec

Date: 2026-06-07

## Purpose

A Flask app that lets the user deploy and run the full LoRA training/generation
pipeline (validated locally as a smoke test in `train_smoke.log` /
`test_gen.log`) on a rented GPU box (RunPod / Vast.ai style: SSH in, run the
app, reach it via an exposed port or proxy URL). It replaces manually SSHing in
and typing long `train_network.py` / `gen_img.py` commands with a web control
panel: upload/select a dataset, point at a base model, queue training and
generation jobs, watch logs live, and download results — all without leaving
the browser.

Built API-first; the web UI is a thin layer that calls the same JSON API.

## Non-goals

- Not a multi-user system — single shared password, single operator.
- Not a general-purpose ML platform — it wraps the specific Kohya
  `train_network.py` / `gen_img.py` workflows already validated for this
  project, not arbitrary training scripts.
- No literal pause/resume of a *running* subprocess (see Job queue below) —
  mid-epoch process state can't be safely suspended and restored.

## Storage layout

```
work/
  app.db                      # SQLite: jobs, dataset registry, model registry
  datasets/<name>/            # extracted uploaded datasets (Kohya <repeats>_<trigger> folders)
  models/<name>.safetensors   # base models, downloaded server-side by URL / HF repo id
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
- `GET /api/datasets` — list available datasets.
- `POST /api/models/download` — body `{url_or_repo_id, filename}`; enqueues a
  `download_model` job that streams the file into `work/models/`.
- `GET /api/models` — list downloaded base models.
- `GET /api/checkpoints` — list LoRA `.safetensors` produced by past training
  jobs, with download links.

**Jobs**
- `POST /api/jobs` — create a job. Body shape depends on `type`:
  - `train`: dataset, base model, output name, resolution, max steps,
    network_dim/alpha, learning rate, etc. — fields pre-filled with the
    defaults from the validated golden command (see Reference command, below).
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
  `/api/datasets` and `/api/models`; numeric fields pre-filled with the golden
  command's defaults.
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

## Execution & environment

The job runner builds the exact CLI invocation for `train_network.py` /
`gen_img.py` from the job's stored params, mirroring the golden command
validated locally:

```powershell
$env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"; $env:OPENBLAS_NUM_THREADS=1; $env:OMP_NUM_THREADS=1
.venv\Scripts\python.exe ..\sd-scripts\train_network.py `
  --pretrained_model_name_or_path <base_model> `
  --train_data_dir <dataset_dir> `
  --output_dir <job_output_dir> --output_name <lora_name> `
  --resolution 512,768 --enable_bucket --min_bucket_reso 256 --max_bucket_reso 1024 `
  --train_batch_size 1 --max_train_steps <steps> --save_every_n_epochs 1 `
  --max_data_loader_n_workers 0 `
  --network_module networks.lora --network_dim <dim> --network_alpha <alpha> `
  --learning_rate 1e-4 --mixed_precision fp16 --gradient_checkpointing --sdpa `
  --caption_extension .txt --save_model_as safetensors
```

The runner sets `cwd` appropriately and uses the venv's `python` /
`python.exe` for the target platform. The Windows-specific environment
variables (`PYTHONUTF8`, `PYTHONIOENCODING`, `OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS`) are applied conditionally based on `platform.system()` —
harmless to set on Linux, but Linux rental boxes generally won't need them
since the underlying issues (cp1252 console encoding, OpenBLAS/RAM pressure
from multiprocessing dataloader workers on a memory-constrained Windows
laptop) are Windows/host-specific quirks observed during local smoke testing.

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
2. Upload a small dataset zip (e.g. the existing 15-pair test dataset).
3. Point the model picker at the already-downloaded
   `test_models/v1-5-pruned-emaonly-fp16.safetensors` (no need to re-download).
4. Queue a training job, watch the live log, confirm it completes and produces
   a `.safetensors` checkpoint in `jobs/<id>/output/`.
5. Queue a generation job against that checkpoint, confirm images appear in
   `jobs/<id>/output/` and are downloadable.
6. Exercise the queue controls: queue two jobs, reorder them, cancel a running
   one, and confirm partial output is retained.

Once this passes locally, the same app (with its venv) is the thing that gets
copied to the rented box and run there.
