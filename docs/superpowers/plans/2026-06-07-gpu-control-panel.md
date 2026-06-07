# GPU Control Panel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Flask control panel that lets the operator upload datasets, download base models, queue SDXL LoRA training and generation jobs, tail live logs, and download results — all from a browser, deployed on an E2E Networks GPU node.

**Architecture:** SQLite-backed job queue with a daemon worker thread that launches Kohya `sdxl_train_network.py` / `sdxl_gen_img.py` as subprocesses and streams their stdout/stderr into per-job log files. Flask serves a JSON API first; a thin Jinja2 + vanilla-JS web UI calls that API.

**Tech Stack:** Python 3.12, Flask, SQLite (stdlib), subprocess (stdlib), threading (stdlib), pytest

---

## File Map

```
control_panel/
  app.py           Flask factory — registers blueprints, starts worker, creates work dirs
  config.py        Env-var config (WORK_DIR, SD_SCRIPTS_DIR, PYTHON_BIN, passwords)
  db.py            Schema, get_db(), init_db(), close_db()
  auth.py          POST /api/login, GET /login, before_request auth guard
  commands.py      build_train_command(), build_generate_command(), build_download_command()
  jobs.py          POST|GET /api/jobs, GET|POST /api/jobs/<id>, /cancel, /reorder, /log, /output
  runner.py        JobRunner daemon thread — polls queue, Popen, log capture, status updates
  datasets.py      POST|GET /api/datasets
  models.py        POST /api/models/download, GET /api/models, GET /api/checkpoints
  views.py         HTML page routes (/, /login, /jobs/train/new, /jobs/<id>, etc.)
  templates/
    base.html
    login.html
    dashboard.html
    job_new_train.html
    job_new_generate.html
    job_detail.html
    model_download.html
    dataset_upload.html
  static/
    style.css
    app.js
tests/
  conftest.py      Flask test client + authed_client fixtures
  test_db.py
  test_auth.py
  test_commands.py
  test_jobs.py
  test_runner.py
  test_datasets.py
  test_models.py
deploy/
  setup.sh         One-time install on E2E TIR node
  start.sh         Launch the app
```

---

## Task 1: Scaffold — config, DB, app factory

**Files:**
- Create: `control_panel/__init__.py`
- Create: `control_panel/config.py`
- Create: `control_panel/db.py`
- Create: `control_panel/app.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_db.py`
- Modify: `requirements.txt` (add flask, pytest)

- [ ] **Step 1: Add flask and pytest to requirements.txt**

Open `requirements.txt` and add:
```
flask
pytest
```

- [ ] **Step 2: Create `control_panel/__init__.py`** (empty)

```python
```

- [ ] **Step 3: Create `control_panel/config.py`**

```python
import os

WORK_DIR = os.environ.get("WORK_DIR", "work")
SD_SCRIPTS_DIR = os.environ.get("SD_SCRIPTS_DIR", "../sd-scripts")
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python")
CONTROL_PANEL_PASSWORD = os.environ.get("CONTROL_PANEL_PASSWORD", "")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
PORT = int(os.environ.get("PORT", "7860"))
```

- [ ] **Step 4: Create `control_panel/db.py`**

```python
import os, sqlite3
from flask import g

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    type         TEXT NOT NULL,
    status       TEXT NOT NULL,
    params       TEXT NOT NULL,
    order_index  INTEGER NOT NULL,
    pid          INTEGER,
    created_at   TEXT NOT NULL
);
"""

def get_db():
    from flask import current_app
    if "db" not in g:
        path = os.path.join(current_app.config["WORK_DIR"], "app.db")
        g.db = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db(app):
    with app.app_context():
        db = get_db()
        db.executescript(SCHEMA)
        db.commit()
```

- [ ] **Step 5: Create `control_panel/app.py`**

```python
import os
from flask import Flask
from . import config as cfg
from .db import close_db, init_db

def create_app(work_dir=None):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["WORK_DIR"]                = work_dir or cfg.WORK_DIR
    app.config["SD_SCRIPTS_DIR"]          = cfg.SD_SCRIPTS_DIR
    app.config["PYTHON_BIN"]              = cfg.PYTHON_BIN
    app.config["CONTROL_PANEL_PASSWORD"]  = cfg.CONTROL_PANEL_PASSWORD
    app.secret_key                        = cfg.SECRET_KEY

    wd = app.config["WORK_DIR"]
    for sub in ("datasets", "models", "jobs"):
        os.makedirs(os.path.join(wd, sub), exist_ok=True)

    app.teardown_appcontext(close_db)
    init_db(app)

    from .auth     import auth_bp
    from .jobs     import jobs_bp
    from .datasets import datasets_bp
    from .models   import models_bp
    from .views    import views_bp
    for bp in (auth_bp, jobs_bp, datasets_bp, models_bp, views_bp):
        app.register_blueprint(bp)

    from .runner import start_worker
    start_worker(app)

    return app
```

- [ ] **Step 6: Create `tests/__init__.py`** (empty)

```python
```

- [ ] **Step 7: Create `tests/conftest.py`**

```python
import os, tempfile, pytest
from control_panel.app import create_app

@pytest.fixture
def app():
    tmp = tempfile.mkdtemp()
    application = create_app(work_dir=tmp)
    application.config["TESTING"] = True
    application.config["CONTROL_PANEL_PASSWORD"] = "testpass"
    yield application

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def authed_client(client):
    client.post("/api/login", json={"password": "testpass"})
    return client
```

- [ ] **Step 8: Write `tests/test_db.py`**

```python
from control_panel.db import get_db

def test_schema_creates_jobs_table(app):
    with app.app_context():
        db = get_db()
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
        ).fetchall()
        assert len(rows) == 1

def test_get_db_returns_same_connection(app):
    with app.app_context():
        assert get_db() is get_db()
```

- [ ] **Step 9: Install deps and run tests**

```bash
cd control_panel && pip install flask pytest
cd .. && pytest tests/test_db.py -v
```

Expected:
```
tests/test_db.py::test_schema_creates_jobs_table PASSED
tests/test_db.py::test_get_db_returns_same_connection PASSED
```

- [ ] **Step 10: Commit**

```bash
git add control_panel/ tests/ requirements.txt
git commit -m "feat: scaffold Flask app factory, config, SQLite DB"
```

---

## Task 2: Auth — login endpoint + request guard

**Files:**
- Create: `control_panel/auth.py`
- Create: `control_panel/views.py` (stub — login page route only, expanded in Task 7)
- Create: `control_panel/templates/login.html`
- Create: `tests/test_auth.py`

- [ ] **Step 1: Write `tests/test_auth.py`**

```python
def test_login_success(client):
    r = client.post("/api/login", json={"password": "testpass"})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True

def test_login_wrong_password(client):
    r = client.post("/api/login", json={"password": "nope"})
    assert r.status_code == 401

def test_unauthenticated_api_returns_401(client):
    r = client.get("/api/jobs")
    assert r.status_code == 401

def test_authenticated_api_passes(authed_client):
    r = authed_client.get("/api/jobs")
    assert r.status_code == 200

def test_logout_clears_session(authed_client):
    authed_client.post("/api/logout")
    r = authed_client.get("/api/jobs")
    assert r.status_code == 401
```

- [ ] **Step 2: Run to confirm all fail**

```bash
pytest tests/test_auth.py -v
```

Expected: all FAIL (auth_bp not registered yet).

- [ ] **Step 3: Create `control_panel/auth.py`**

```python
from flask import Blueprint, request, session, jsonify, redirect, url_for, current_app, render_template

auth_bp = Blueprint("auth", __name__)

@auth_bp.before_app_request
def require_login():
    if session.get("authed"):
        return
    path = request.path
    if path in ("/api/login", "/login") or path.startswith("/static"):
        return
    if path.startswith("/api/"):
        return jsonify(error="unauthorized"), 401
    return redirect(url_for("auth.login_page"))

@auth_bp.route("/login")
def login_page():
    return render_template("login.html")

@auth_bp.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    if data.get("password") == current_app.config["CONTROL_PANEL_PASSWORD"]:
        session["authed"] = True
        return jsonify(ok=True)
    return jsonify(error="wrong password"), 401

@auth_bp.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify(ok=True)
```

- [ ] **Step 4: Create stub `control_panel/views.py`**

```python
from flask import Blueprint, render_template

views_bp = Blueprint("views", __name__)

@views_bp.route("/")
def dashboard():
    return render_template("dashboard.html")

@views_bp.route("/jobs/train/new")
def new_train_job():
    return render_template("job_new_train.html")

@views_bp.route("/jobs/generate/new")
def new_generate_job():
    return render_template("job_new_generate.html")

@views_bp.route("/jobs/<job_id>")
def job_detail(job_id):
    return render_template("job_detail.html", job_id=job_id)

@views_bp.route("/models/download")
def model_download_page():
    return render_template("model_download.html")

@views_bp.route("/datasets/upload")
def dataset_upload_page():
    return render_template("dataset_upload.html")
```

- [ ] **Step 5: Create placeholder templates** (just enough to not crash — real HTML comes in Task 7)

Create `control_panel/templates/login.html`:
```html
<!doctype html><title>Login</title>
<form id="f"><input type="password" id="pw" placeholder="Password"><button>Login</button></form>
<script>
document.getElementById('f').onsubmit=async e=>{e.preventDefault();
const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:document.getElementById('pw').value})});
if(r.ok)location='/';};
</script>
```

Create `control_panel/templates/dashboard.html`:
```html
<!doctype html><title>Dashboard</title><body>Dashboard placeholder</body>
```

Create `control_panel/templates/job_new_train.html`:
```html
<!doctype html><title>New Training Job</title><body>Train form placeholder</body>
```

Create `control_panel/templates/job_new_generate.html`:
```html
<!doctype html><title>New Generation Job</title><body>Generate form placeholder</body>
```

Create `control_panel/templates/job_detail.html`:
```html
<!doctype html><title>Job Detail</title><body>Job detail placeholder for {{ job_id }}</body>
```

Create `control_panel/templates/model_download.html`:
```html
<!doctype html><title>Download Model</title><body>Model download placeholder</body>
```

Create `control_panel/templates/dataset_upload.html`:
```html
<!doctype html><title>Upload Dataset</title><body>Upload placeholder</body>
```

- [ ] **Step 6: Create stub remaining blueprints** so `app.py` can import them without error

Create `control_panel/jobs.py`:
```python
from flask import Blueprint, jsonify
jobs_bp = Blueprint("jobs", __name__)

@jobs_bp.route("/api/jobs", methods=["GET"])
def list_jobs():
    return jsonify([])
```

Create `control_panel/datasets.py`:
```python
from flask import Blueprint, jsonify
datasets_bp = Blueprint("datasets", __name__)
```

Create `control_panel/models.py`:
```python
from flask import Blueprint, jsonify
models_bp = Blueprint("models", __name__)
```

Create `control_panel/runner.py`:
```python
def start_worker(app):
    pass  # expanded in Task 5

def kill_job(pid):
    pass  # expanded in Task 5
```

- [ ] **Step 7: Run tests**

```bash
pytest tests/test_auth.py -v
```

Expected: all 5 PASS.

- [ ] **Step 8: Commit**

```bash
git add control_panel/ tests/test_auth.py
git commit -m "feat: auth blueprint — login/logout + request guard"
```

---

## Task 3: Command builders

**Files:**
- Create: `control_panel/commands.py`
- Create: `tests/test_commands.py`

- [ ] **Step 1: Write `tests/test_commands.py`**

```python
import os
from control_panel.commands import (
    build_train_command, build_generate_command, build_download_command
)

CFG = {"PYTHON_BIN": "python", "SD_SCRIPTS_DIR": "/sd"}

BASE_TRAIN = {
    "base_model": "/models/ilxl.safetensors",
    "dataset_dir": "/data/5_mywebtoon",
    "output_dir": "/jobs/abc/output",
    "output_name": "mywebtoon_lora",
    "epochs": 30,
}

def test_train_uses_sdxl_script():
    cmd, _, cwd = build_train_command(BASE_TRAIN, CFG)
    assert "sdxl_train_network.py" in " ".join(cmd)
    assert cwd == "/sd"

def test_train_default_network_dim():
    cmd, _, _ = build_train_command(BASE_TRAIN, CFG)
    idx = cmd.index("--network_dim")
    assert cmd[idx + 1] == "32"

def test_train_override_network_dim():
    params = {**BASE_TRAIN, "network_dim": 64}
    cmd, _, _ = build_train_command(params, CFG)
    assert cmd[cmd.index("--network_dim") + 1] == "64"

def test_train_has_sdxl_flags():
    cmd, _, _ = build_train_command(BASE_TRAIN, CFG)
    joined = " ".join(cmd)
    assert "--no_half_vae" in joined
    assert "--cache_text_encoder_outputs" in joined
    assert "--min_snr_gamma" in joined

def test_train_default_lr():
    cmd, _, _ = build_train_command(BASE_TRAIN, CFG)
    assert cmd[cmd.index("--learning_rate") + 1] == "3e-5"

def test_generate_uses_sdxl_script():
    params = {"base_model": "m", "output_dir": "o", "prompt": "1girl", "lora_paths": []}
    cmd, _, _ = build_generate_command(params, CFG)
    assert "sdxl_gen_img.py" in " ".join(cmd)

def test_generate_sdpa_by_default():
    params = {"base_model": "m", "output_dir": "o", "prompt": "1girl", "lora_paths": []}
    cmd, _, _ = build_generate_command(params, CFG)
    assert "--sdpa" in cmd

def test_generate_negative_prompt_included():
    params = {"base_model": "m", "output_dir": "o", "prompt": "1girl",
              "negative_prompt": "bad quality", "lora_paths": []}
    cmd, _, _ = build_generate_command(params, CFG)
    assert "--negative_prompt" in cmd
    assert "bad quality" in cmd

def test_generate_lora_paths_included():
    params = {"base_model": "m", "output_dir": "o", "prompt": "1girl",
              "lora_paths": ["/ckpts/ep1.safetensors"]}
    cmd, _, _ = build_generate_command(params, CFG)
    assert "--network_weights" in cmd

def test_download_command_contains_url_and_dest():
    params = {"url": "https://civitai.com/api/dl/12345",
              "dest_path": "/models/ilxl.safetensors"}
    cmd, _, _ = build_download_command(params, CFG)
    assert "https://civitai.com/api/dl/12345" in cmd
    assert "/models/ilxl.safetensors" in cmd
```

- [ ] **Step 2: Run to confirm all fail**

```bash
pytest tests/test_commands.py -v
```

Expected: ImportError / all FAIL.

- [ ] **Step 3: Create `control_panel/commands.py`**

```python
import os, platform

def _win_env():
    if platform.system() == "Windows":
        return {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
    return {}

def build_train_command(params, config):
    """Return (cmd_list, env_dict, cwd) for sdxl_train_network.py."""
    py  = config["PYTHON_BIN"]
    sd  = config["SD_SCRIPTS_DIR"]
    cmd = [
        py, os.path.join(sd, "sdxl_train_network.py"),
        "--pretrained_model_name_or_path", params["base_model"],
        "--train_data_dir",                params["dataset_dir"],
        "--output_dir",                    params["output_dir"],
        "--output_name",                   params["output_name"],
        "--resolution",                    params.get("resolution", "1024,1024"),
        "--enable_bucket",
        "--bucket_reso_steps",             "64",
        "--min_bucket_reso",               "256",
        "--max_bucket_reso",               "2048",
        "--train_batch_size",              str(params.get("batch_size", 4)),
        "--max_train_epochs",              str(params["epochs"]),
        "--save_every_n_epochs",           "1",
        "--network_module",                "networks.lora",
        "--network_dim",                   str(params.get("network_dim", 32)),
        "--network_alpha",                 str(params.get("network_alpha", 32)),
        "--learning_rate",                 str(params.get("learning_rate", "3e-5")),
        "--min_snr_gamma",                 str(params.get("min_snr_gamma", 5)),
        "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk",
        "--no_half_vae",
        "--mixed_precision",               "fp16",
        "--gradient_checkpointing",
        "--sdpa",
        "--caption_extension",             ".txt",
        "--save_model_as",                 "safetensors",
    ]
    return cmd, {**os.environ, **_win_env()}, sd

def build_generate_command(params, config):
    """Return (cmd_list, env_dict, cwd) for sdxl_gen_img.py."""
    py  = config["PYTHON_BIN"]
    sd  = config["SD_SCRIPTS_DIR"]
    cmd = [
        py, os.path.join(sd, "sdxl_gen_img.py"),
        "--ckpt",          params["base_model"],
        "--outdir",        params["output_dir"],
        "--sdpa",
        "--W",             str(params.get("width", 832)),
        "--H",             str(params.get("height", 1216)),
        "--steps",         str(params.get("steps", 28)),
        "--scale",         str(params.get("cfg_scale", 7)),
        "--batch_size",    str(params.get("batch_size", 1)),
        "--images_per_prompt", str(params.get("images_per_prompt", 4)),
        "--prompt",        params["prompt"],
    ]
    if params.get("negative_prompt"):
        cmd += ["--negative_prompt", params["negative_prompt"]]
    loras = params.get("lora_paths", [])
    if loras:
        cmd += [
            "--network_module",  "networks.lora",
            "--network_weights", ",".join(loras),
        ]
    return cmd, {**os.environ, **_win_env()}, sd

def build_download_command(params, config):
    """Return (cmd_list, env_dict, cwd) to download a file via Python stdlib."""
    py   = config["PYTHON_BIN"]
    url  = params["url"]
    dest = params["dest_path"]
    script = (
        "import urllib.request, sys, os\n"
        "url, dest = sys.argv[1], sys.argv[2]\n"
        "os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)\n"
        "print(f'Downloading {url}', flush=True)\n"
        "def hook(n, bs, ts):\n"
        "    if ts > 0: print(f'{min(n*bs,ts)}/{ts} bytes', flush=True)\n"
        "urllib.request.urlretrieve(url, dest, reporthook=hook)\n"
        "print('Done.', flush=True)\n"
    )
    return [py, "-c", script, url, dest], dict(os.environ), "."
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_commands.py -v
```

Expected: all 11 PASS.

- [ ] **Step 5: Commit**

```bash
git add control_panel/commands.py tests/test_commands.py
git commit -m "feat: command builders for train/generate/download_model"
```

---

## Task 4: Jobs CRUD API

**Files:**
- Overwrite: `control_panel/jobs.py`
- Create: `tests/test_jobs.py`

- [ ] **Step 1: Write `tests/test_jobs.py`**

```python
import json

def _create(client, job_type="train", params=None):
    return client.post("/api/jobs", json={
        "type": job_type,
        "params": params or {"output_name": "test", "epochs": 1}
    })

def test_create_job_returns_201(authed_client):
    r = _create(authed_client)
    assert r.status_code == 201
    assert "id" in r.get_json()

def test_create_invalid_type_returns_400(authed_client):
    r = authed_client.post("/api/jobs", json={"type": "badtype", "params": {}})
    assert r.status_code == 400

def test_list_jobs(authed_client):
    _create(authed_client)
    r = authed_client.get("/api/jobs")
    assert r.status_code == 200
    assert len(r.get_json()) >= 1

def test_get_job_detail(authed_client):
    job_id = _create(authed_client).get_json()["id"]
    r = authed_client.get(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    d = r.get_json()
    assert d["status"] == "queued"
    assert "output_files" in d

def test_get_missing_job_returns_404(authed_client):
    r = authed_client.get("/api/jobs/does-not-exist")
    assert r.status_code == 404

def test_cancel_queued_job(authed_client):
    job_id = _create(authed_client).get_json()["id"]
    r = authed_client.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 200
    assert authed_client.get(f"/api/jobs/{job_id}").get_json()["status"] == "cancelled"

def test_cancel_already_done_returns_400(authed_client):
    from control_panel.db import get_db
    job_id = _create(authed_client).get_json()["id"]
    with authed_client.application.app_context():
        db = get_db()
        db.execute("UPDATE jobs SET status='done' WHERE id=?", (job_id,))
        db.commit()
    r = authed_client.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 400

def test_reorder_moves_job_up(authed_client):
    id1 = _create(authed_client).get_json()["id"]
    id2 = _create(authed_client).get_json()["id"]
    authed_client.post(f"/api/jobs/{id2}/reorder", json={"direction": "up"})
    jobs = authed_client.get("/api/jobs").get_json()
    order = [j["id"] for j in jobs]
    assert order.index(id2) < order.index(id1)

def test_reorder_invalid_direction_returns_400(authed_client):
    job_id = _create(authed_client).get_json()["id"]
    r = authed_client.post(f"/api/jobs/{job_id}/reorder", json={"direction": "sideways"})
    assert r.status_code == 400

def test_log_returns_empty_when_no_log(authed_client):
    job_id = _create(authed_client).get_json()["id"]
    r = authed_client.get(f"/api/jobs/{job_id}/log")
    assert r.status_code == 200
    assert r.data == b""

def test_log_returns_content_and_size_header(authed_client, app):
    import os
    job_id = _create(authed_client).get_json()["id"]
    log_path = os.path.join(app.config["WORK_DIR"], "jobs", job_id, "log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write("step 1/10 loss=0.5\n")
    r = authed_client.get(f"/api/jobs/{job_id}/log")
    assert b"step 1/10" in r.data
    assert int(r.headers["X-Log-Size"]) == 19
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/test_jobs.py -v
```

Expected: most FAIL (stub jobs.py only has GET /api/jobs).

- [ ] **Step 3: Overwrite `control_panel/jobs.py`**

```python
import json, os, uuid
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, Response, send_from_directory
from .db import get_db

jobs_bp = Blueprint("jobs", __name__)

def _row_to_dict(row):
    d = dict(row)
    d["params"] = json.loads(d["params"])
    return d

@jobs_bp.route("/api/jobs", methods=["GET"])
def list_jobs():
    rows = get_db().execute(
        "SELECT * FROM jobs ORDER BY order_index ASC, created_at DESC"
    ).fetchall()
    return jsonify([_row_to_dict(r) for r in rows])

@jobs_bp.route("/api/jobs", methods=["POST"])
def create_job():
    data = request.get_json(force=True)
    if data.get("type") not in ("train", "generate", "download_model"):
        return jsonify(error="type must be train | generate | download_model"), 400
    db  = get_db()
    max_idx = db.execute("SELECT MAX(order_index) FROM jobs").fetchone()[0] or 0
    job_id  = str(uuid.uuid4())
    now     = datetime.now(timezone.utc).isoformat()
    db.execute(
        "INSERT INTO jobs (id, type, status, params, order_index, pid, created_at)"
        " VALUES (?, ?, 'queued', ?, ?, NULL, ?)",
        (job_id, data["type"], json.dumps(data.get("params", {})), max_idx + 1, now),
    )
    db.commit()
    os.makedirs(
        os.path.join(current_app.config["WORK_DIR"], "jobs", job_id, "output"),
        exist_ok=True,
    )
    return jsonify(id=job_id), 201

@jobs_bp.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    row = get_db().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        return jsonify(error="not found"), 404
    d = _row_to_dict(row)
    out_dir = os.path.join(current_app.config["WORK_DIR"], "jobs", job_id, "output")
    d["output_files"] = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []
    return jsonify(d)

@jobs_bp.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    from .runner import kill_job
    db  = get_db()
    row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        return jsonify(error="not found"), 404
    if row["status"] not in ("queued", "running"):
        return jsonify(error=f"cannot cancel job with status '{row['status']}'"), 400
    if row["status"] == "running" and row["pid"]:
        kill_job(row["pid"])
    db.execute("UPDATE jobs SET status='cancelled' WHERE id=?", (job_id,))
    db.commit()
    return jsonify(ok=True)

@jobs_bp.route("/api/jobs/<job_id>/reorder", methods=["POST"])
def reorder_job(job_id):
    direction = (request.get_json(force=True) or {}).get("direction")
    if direction not in ("up", "down"):
        return jsonify(error="direction must be 'up' or 'down'"), 400
    db  = get_db()
    row = db.execute(
        "SELECT * FROM jobs WHERE id=? AND status='queued'", (job_id,)
    ).fetchone()
    if row is None:
        return jsonify(error="job not found or not queued"), 404
    cur = row["order_index"]
    if direction == "up":
        nb = db.execute(
            "SELECT * FROM jobs WHERE status='queued' AND order_index < ?"
            " ORDER BY order_index DESC LIMIT 1", (cur,)
        ).fetchone()
    else:
        nb = db.execute(
            "SELECT * FROM jobs WHERE status='queued' AND order_index > ?"
            " ORDER BY order_index ASC LIMIT 1", (cur,)
        ).fetchone()
    if nb:
        db.execute("UPDATE jobs SET order_index=? WHERE id=?", (nb["order_index"], job_id))
        db.execute("UPDATE jobs SET order_index=? WHERE id=?", (cur, nb["id"]))
        db.commit()
    return jsonify(ok=True)

@jobs_bp.route("/api/jobs/<job_id>/log")
def stream_log(job_id):
    since    = request.args.get("since", 0, type=int)
    log_path = os.path.join(current_app.config["WORK_DIR"], "jobs", job_id, "log.txt")
    if not os.path.exists(log_path):
        return Response(b"", mimetype="text/plain", headers={"X-Log-Size": "0"})
    with open(log_path, "rb") as f:
        f.seek(since)
        data = f.read()
    return Response(
        data, mimetype="text/plain",
        headers={"X-Log-Size": str(since + len(data))}
    )

@jobs_bp.route("/api/jobs/<job_id>/output/<path:filename>")
def download_output(job_id, filename):
    out_dir = os.path.join(current_app.config["WORK_DIR"], "jobs", job_id, "output")
    return send_from_directory(out_dir, filename, as_attachment=True)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_jobs.py -v
```

Expected: all 11 PASS.

- [ ] **Step 5: Commit**

```bash
git add control_panel/jobs.py tests/test_jobs.py
git commit -m "feat: jobs CRUD API — create/list/get/cancel/reorder/log"
```

---

## Task 5: Background worker

**Files:**
- Overwrite: `control_panel/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write `tests/test_runner.py`**

```python
import os, time, sqlite3, json, tempfile
from control_panel.runner import kill_job, start_worker

def test_kill_invalid_pid_does_not_raise():
    kill_job(999999)  # must not raise

def test_worker_completes_download_model_job(app):
    """Queue a download_model job using a local file:// URL, wait for done."""
    import platform
    # Write a dummy "model" file to use as download source
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors")
    tmp.write(b"fake-model-data")
    tmp.close()

    url = "file:///" + tmp.name.replace(os.sep, "/") if platform.system() == "Windows" \
          else "file://" + tmp.name

    db_path = os.path.join(app.config["WORK_DIR"], "app.db")
    job_id  = "runner-test-job-01"
    job_dir = os.path.join(app.config["WORK_DIR"], "jobs", job_id, "output")
    os.makedirs(job_dir, exist_ok=True)
    models_dir = os.path.join(app.config["WORK_DIR"], "models")
    os.makedirs(models_dir, exist_ok=True)

    params = {"url": url, "filename": "fake.safetensors"}
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO jobs (id, type, status, params, order_index, pid, created_at)"
        " VALUES (?, 'download_model', 'queued', ?, 1, NULL, datetime('now'))",
        (job_id, json.dumps(params))
    )
    conn.commit()
    conn.close()

    start_worker(app)

    for _ in range(30):         # up to 15 seconds
        time.sleep(0.5)
        conn2 = sqlite3.connect(db_path)
        row   = conn2.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        conn2.close()
        if row and row[0] in ("done", "failed"):
            break

    assert row[0] == "done", f"Expected done, got {row[0]}"

    dest = os.path.join(app.config["WORK_DIR"], "models", "fake.safetensors")
    assert os.path.exists(dest)
    os.unlink(tmp.name)
```

- [ ] **Step 2: Run to confirm the test fails**

```bash
pytest tests/test_runner.py::test_worker_completes_download_model_job -v
```

Expected: FAIL (stub `start_worker` does nothing).

- [ ] **Step 3: Overwrite `control_panel/runner.py`**

```python
import json, os, platform, signal, sqlite3, subprocess, threading, time

_lock   = threading.Lock()
_thread = None

def kill_job(pid):
    try:
        if platform.system() == "Windows":
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        pass

def _conn(db_path):
    c = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    c.row_factory = sqlite3.Row
    return c

def _resolve_params(row, work_dir):
    """Expand filenames in params to absolute paths and inject output_dir."""
    params   = json.loads(row["params"])
    job_id   = row["id"]
    out_dir  = os.path.join(work_dir, "jobs", job_id, "output")
    os.makedirs(out_dir, exist_ok=True)

    if row["type"] == "train":
        params["output_dir"]  = out_dir
        params["dataset_dir"] = os.path.join(
            work_dir, "datasets", params.get("dataset", ""))
        params["base_model"]  = os.path.join(
            work_dir, "models", params.get("base_model_filename", ""))

    elif row["type"] == "generate":
        params["output_dir"]  = out_dir
        params["base_model"]  = os.path.join(
            work_dir, "models", params.get("base_model_filename", ""))
        params["lora_paths"]  = [
            os.path.join(work_dir, "jobs", jid, "output", fname)
            for jid, fname in params.get("lora_checkpoints", [])
        ]

    elif row["type"] == "download_model":
        params["dest_path"]   = os.path.join(
            work_dir, "models", params["filename"])

    return params, out_dir

def _worker(app):
    from .commands import (
        build_train_command, build_generate_command, build_download_command
    )
    work_dir = app.config["WORK_DIR"]
    db_path  = os.path.join(work_dir, "app.db")

    while True:
        time.sleep(2)
        conn = _conn(db_path)
        row  = conn.execute(
            "SELECT * FROM jobs WHERE status='queued'"
            " ORDER BY order_index ASC LIMIT 1"
        ).fetchone()
        if row is None:
            conn.close()
            continue

        job_id   = row["id"]
        log_path = os.path.join(work_dir, "jobs", job_id, "log.txt")
        params, out_dir = _resolve_params(row, work_dir)

        builders = {
            "train":          build_train_command,
            "generate":       build_generate_command,
            "download_model": build_download_command,
        }
        if row["type"] not in builders:
            conn.execute("UPDATE jobs SET status='failed' WHERE id=?", (job_id,))
            conn.commit()
            conn.close()
            continue

        cmd, env, cwd = builders[row["type"]](params, app.config)

        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "env":    env,
        }
        if platform.system() != "Windows":
            popen_kwargs["start_new_session"] = True

        try:
            proc = subprocess.Popen(cmd, cwd=cwd, **popen_kwargs)
        except Exception as exc:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"Failed to start process: {exc}\n")
            conn.execute("UPDATE jobs SET status='failed' WHERE id=?", (job_id,))
            conn.commit()
            conn.close()
            continue

        conn.execute(
            "UPDATE jobs SET status='running', pid=? WHERE id=?", (proc.pid, job_id)
        )
        conn.commit()
        conn.close()

        with open(log_path, "wb") as log_f:
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
        proc.wait()

        conn2   = _conn(db_path)
        current = conn2.execute(
            "SELECT status FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        if current and current["status"] == "running":
            final = "done" if proc.returncode == 0 else "failed"
            conn2.execute(
                "UPDATE jobs SET status=?, pid=NULL WHERE id=?", (final, job_id)
            )
            conn2.commit()
        conn2.close()

def start_worker(app):
    global _thread
    with _lock:
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_worker, args=(app,), daemon=True)
            _thread.start()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_runner.py -v
```

Expected: both PASS (the download job should finish in a few seconds).

- [ ] **Step 5: Commit**

```bash
git add control_panel/runner.py tests/test_runner.py
git commit -m "feat: background worker thread — poll queue, launch subprocesses, capture logs"
```

---

## Task 6: Datasets and Models API

**Files:**
- Overwrite: `control_panel/datasets.py`
- Overwrite: `control_panel/models.py`
- Create: `tests/test_datasets.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write `tests/test_datasets.py`**

```python
import io, zipfile

def _make_zip(folder_name, files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(f"{folder_name}/{name}", content)
    buf.seek(0)
    return buf

def test_upload_valid_dataset(authed_client):
    z = _make_zip("5_mywebtoon", {"0001.webp": b"img", "0001.txt": "mywebtoon, 1girl"})
    r = authed_client.post(
        "/api/datasets",
        data={"file": (z, "ds.zip")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 201
    assert r.get_json()["name"] == "5_mywebtoon"

def test_upload_no_kohya_folder_returns_422(authed_client):
    z = _make_zip("images", {"img.webp": b"x"})
    r = authed_client.post(
        "/api/datasets",
        data={"file": (z, "ds.zip")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 422
    assert "repeats" in r.get_json()["error"]

def test_upload_non_zip_returns_400(authed_client):
    r = authed_client.post(
        "/api/datasets",
        data={"file": (io.BytesIO(b"not a zip"), "data.tar.gz")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400

def test_upload_duplicate_returns_409(authed_client):
    z1 = _make_zip("5_mywebtoon", {"a.webp": b"x", "a.txt": "t"})
    authed_client.post("/api/datasets", data={"file": (z1, "d.zip")},
                       content_type="multipart/form-data")
    z2 = _make_zip("5_mywebtoon", {"b.webp": b"x", "b.txt": "t"})
    r  = authed_client.post("/api/datasets", data={"file": (z2, "d2.zip")},
                            content_type="multipart/form-data")
    assert r.status_code == 409

def test_list_datasets(authed_client):
    z = _make_zip("6_testtoon", {"a.webp": b"x", "a.txt": "t"})
    authed_client.post("/api/datasets", data={"file": (z, "d.zip")},
                       content_type="multipart/form-data")
    names = authed_client.get("/api/datasets").get_json()
    assert "6_testtoon" in names
```

- [ ] **Step 2: Write `tests/test_models.py`**

```python
import os, json

def test_list_models_empty(authed_client):
    assert authed_client.get("/api/models").get_json() == []

def test_list_models_with_file(authed_client, app):
    path = os.path.join(app.config["WORK_DIR"], "models", "ilxl.safetensors")
    open(path, "wb").write(b"fake")
    assert "ilxl.safetensors" in authed_client.get("/api/models").get_json()

def test_queue_download_valid(authed_client):
    r = authed_client.post("/api/models/download", json={
        "url": "https://civitai.com/api/dl/12345",
        "filename": "ilxl.safetensors",
    })
    assert r.status_code == 201
    assert "id" in r.get_json()

def test_queue_download_bad_extension(authed_client):
    r = authed_client.post("/api/models/download", json={
        "url": "https://example.com/f.bin", "filename": "model.bin",
    })
    assert r.status_code == 400

def test_queue_download_missing_fields(authed_client):
    r = authed_client.post("/api/models/download", json={"url": ""})
    assert r.status_code == 400

def test_list_checkpoints_empty(authed_client):
    assert authed_client.get("/api/checkpoints").get_json() == []

def test_list_checkpoints_after_done_train_job(authed_client, app):
    from control_panel.db import get_db
    import uuid, json as _json
    job_id = str(uuid.uuid4())
    out_dir = os.path.join(app.config["WORK_DIR"], "jobs", job_id, "output")
    os.makedirs(out_dir, exist_ok=True)
    # write a fake LoRA checkpoint
    open(os.path.join(out_dir, "mywebtoon_ep10.safetensors"), "wb").write(b"ckpt")
    with app.app_context():
        db = get_db()
        db.execute(
            "INSERT INTO jobs (id,type,status,params,order_index,pid,created_at)"
            " VALUES (?,?,?,?,1,NULL,datetime('now'))",
            (job_id, "train", "done", _json.dumps({"output_name": "mywebtoon"}))
        )
        db.commit()
    results = authed_client.get("/api/checkpoints").get_json()
    assert any(c["filename"] == "mywebtoon_ep10.safetensors" for c in results)
```

- [ ] **Step 3: Run to confirm failures**

```bash
pytest tests/test_datasets.py tests/test_models.py -v
```

Expected: most FAIL (stub blueprints).

- [ ] **Step 4: Overwrite `control_panel/datasets.py`**

```python
import os, re, zipfile, tempfile
from flask import Blueprint, request, jsonify, current_app

datasets_bp = Blueprint("datasets", __name__)

_KOHYA_RE = re.compile(r"^\d+_.+$")

def _find_kohya_folders(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        tops = {n.split("/")[0] for n in zf.namelist() if "/" in n}
    found = [f for f in tops if _KOHYA_RE.match(f)]
    if not found:
        return None, ("zip must contain at least one <repeats>_<trigger> folder"
                      " (e.g. 5_mywebtoon) — create it with 'run.py package'")
    return found, None

@datasets_bp.route("/api/datasets", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify(error="no file uploaded"), 400
    f = request.files["file"]
    if not f.filename.endswith(".zip"):
        return jsonify(error="file must be a .zip"), 400
    ds_dir = os.path.join(current_app.config["WORK_DIR"], "datasets")
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=".zip", dir=ds_dir)
    f.save(tmp.name)
    tmp.close()
    folders, err = _find_kohya_folders(tmp.name)
    if err:
        os.unlink(tmp.name)
        return jsonify(error=err), 422
    name = folders[0]
    dest = os.path.join(ds_dir, name)
    if os.path.exists(dest):
        os.unlink(tmp.name)
        return jsonify(error=f"dataset '{name}' already exists"), 409
    with zipfile.ZipFile(tmp.name) as zf:
        zf.extractall(ds_dir)
    os.unlink(tmp.name)
    return jsonify(name=name), 201

@datasets_bp.route("/api/datasets", methods=["GET"])
def list_datasets():
    d     = os.path.join(current_app.config["WORK_DIR"], "datasets")
    names = [
        n for n in os.listdir(d)
        if os.path.isdir(os.path.join(d, n)) and _KOHYA_RE.match(n)
    ] if os.path.isdir(d) else []
    return jsonify(sorted(names))
```

- [ ] **Step 5: Overwrite `control_panel/models.py`**

```python
import glob, json, os, uuid
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app
from .db import get_db

models_bp = Blueprint("models", __name__)

@models_bp.route("/api/models", methods=["GET"])
def list_models():
    d     = os.path.join(current_app.config["WORK_DIR"], "models")
    files = []
    if os.path.isdir(d):
        for pat in ("*.safetensors", "*.ckpt"):
            files += [os.path.basename(p) for p in glob.glob(os.path.join(d, pat))]
    return jsonify(sorted(files))

@models_bp.route("/api/models/download", methods=["POST"])
def queue_download():
    data     = request.get_json(force=True)
    url      = (data.get("url") or "").strip()
    filename = (data.get("filename") or "").strip()
    if not url or not filename:
        return jsonify(error="url and filename are required"), 400
    if not filename.endswith((".safetensors", ".ckpt")):
        return jsonify(error="filename must end in .safetensors or .ckpt"), 400
    db      = get_db()
    max_idx = db.execute("SELECT MAX(order_index) FROM jobs").fetchone()[0] or 0
    job_id  = str(uuid.uuid4())
    now     = datetime.now(timezone.utc).isoformat()
    db.execute(
        "INSERT INTO jobs (id, type, status, params, order_index, pid, created_at)"
        " VALUES (?, 'download_model', 'queued', ?, ?, NULL, ?)",
        (job_id, json.dumps({"url": url, "filename": filename}), max_idx + 1, now),
    )
    db.commit()
    os.makedirs(
        os.path.join(current_app.config["WORK_DIR"], "jobs", job_id, "output"),
        exist_ok=True,
    )
    return jsonify(id=job_id), 201

@models_bp.route("/api/checkpoints", methods=["GET"])
def list_checkpoints():
    jobs_dir = os.path.join(current_app.config["WORK_DIR"], "jobs")
    results  = []
    if not os.path.isdir(jobs_dir):
        return jsonify(results)
    rows = get_db().execute(
        "SELECT id, params FROM jobs WHERE type='train' AND status='done'"
    ).fetchall()
    for row in rows:
        out_dir = os.path.join(jobs_dir, row["id"], "output")
        if not os.path.isdir(out_dir):
            continue
        for fname in os.listdir(out_dir):
            if fname.endswith(".safetensors"):
                results.append({"job_id": row["id"], "filename": fname})
    return jsonify(results)
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_datasets.py tests/test_models.py -v
```

Expected: all PASS.

- [ ] **Step 7: Run full suite to check no regressions**

```bash
pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add control_panel/datasets.py control_panel/models.py \
        tests/test_datasets.py tests/test_models.py
git commit -m "feat: dataset upload/list + model download queue + checkpoints list"
```

---

## Task 7: Web UI — static assets and templates

**Files:**
- Create: `control_panel/static/style.css`
- Create: `control_panel/static/app.js`
- Overwrite: all `control_panel/templates/*.html`

No new API routes — all pages call the JSON API built in Tasks 2-6.

- [ ] **Step 1: Create `control_panel/static/style.css`**

```css
*, *::before, *::after { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; background: #111; color: #eee; }
nav  { display: flex; align-items: center; gap: 12px; padding: 10px 16px;
       background: #1a1a1a; border-bottom: 1px solid #333; }
nav strong { flex: 1; }
main { padding: 16px; max-width: 1100px; margin: 0 auto; }
body.center { display: flex; align-items: center; justify-content: center; height: 100vh; }
h2 { margin-top: 0; }
button { padding: 6px 14px; font-size: 13px; cursor: pointer;
         background: #2a2a2a; color: #eee; border: 1px solid #444; border-radius: 4px; }
button:hover { background: #3a3a3a; }
input, select, textarea {
  background: #1e1e1e; border: 1px solid #444; color: #eee;
  border-radius: 4px; padding: 6px 8px; font-size: 13px; width: 100%; }
label  { display: block; margin: 10px 0 4px; font-size: 13px; color: #aaa; }
form   { max-width: 540px; }
form button[type=submit] { margin-top: 18px; width: 100%; background: #2563eb; border-color: #2563eb; }
table  { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #2a2a2a; }
th     { color: #888; font-weight: 500; }
.toolbar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
.status-queued   { color: #aaa; }
.status-running  { color: #3b82f6; }
.status-done     { color: #22c55e; }
.status-failed   { color: #ef4444; }
.status-cancelled{ color: #f59e0b; }
.progress { font-size: 11px; color: #666; max-width: 280px;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
pre { background: #0a0a0a; padding: 12px; border-radius: 6px; font-size: 12px;
      max-height: 500px; overflow-y: auto; white-space: pre-wrap; }
#login-form { text-align: center; }
#login-form input { width: 240px; margin-bottom: 10px; }
.gallery { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.gallery img { max-width: 220px; border-radius: 4px; border: 1px solid #333; }
```

- [ ] **Step 2: Create `control_panel/static/app.js`**

```javascript
// Shared helpers used by multiple pages
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok && r.status === 401) { location = '/login'; return null; }
  return r;
}

// Dashboard — job queue table + log polling
async function dashboardInit() {
  async function loadJobs() {
    const r = await api('/api/jobs');
    if (!r) return;
    const jobs = await r.json();
    const tbody = document.getElementById('jobs-body');
    if (!tbody) return;
    tbody.innerHTML = jobs.map(j => {
      const name = j.params.output_name || j.params.filename || j.id.slice(0, 8);
      const canCancel  = ['queued','running'].includes(j.status);
      const canReorder = j.status === 'queued';
      return `<tr>
        <td>${j.type}</td>
        <td><a href="/jobs/${j.id}">${name}</a></td>
        <td class="status-${j.status}">${j.status}</td>
        <td class="progress" id="prog-${j.id}"></td>
        <td style="white-space:nowrap">
          ${canReorder ? `<button onclick="reorder('${j.id}','up')">↑</button>
                          <button onclick="reorder('${j.id}','down')">↓</button> ` : ''}
          ${canCancel  ? `<button onclick="cancelJob('${j.id}')">Cancel</button>` : ''}
        </td></tr>`;
    }).join('');
    jobs.filter(j => j.status === 'running').forEach(j => pollProgress(j.id));
  }

  window.reorder = async (id, dir) => {
    await api(`/api/jobs/${id}/reorder`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({direction: dir})
    });
    loadJobs();
  };

  window.cancelJob = async (id) => {
    if (!confirm('Cancel this job? Partial output is kept.')) return;
    await api(`/api/jobs/${id}/cancel`, {method: 'POST'});
    loadJobs();
  };

  async function pollProgress(id) {
    let offset = 0;
    const el   = document.getElementById(`prog-${id}`);
    const tick = async () => {
      const r = await api(`/api/jobs/${id}/log?since=${offset}`);
      if (!r) return;
      const text = await r.text();
      if (text && el) el.textContent = text.trim().split('\n').pop().slice(0, 90);
      offset = parseInt(r.headers.get('X-Log-Size') || offset, 10);
      const jr = await api(`/api/jobs/${id}`);
      if (!jr) return;
      const job = await jr.json();
      if (job.status === 'running') setTimeout(tick, 3000);
      else loadJobs();
    };
    tick();
  }

  loadJobs();
  setInterval(loadJobs, 12000);
}

// Job detail — log tail + output gallery
async function jobDetailInit(jobId) {
  let offset = 0;
  const logEl = document.getElementById('log');

  async function pollLog() {
    const r = await api(`/api/jobs/${jobId}/log?since=${offset}`);
    if (!r) return;
    const text = await r.text();
    if (text && logEl) { logEl.textContent += text; logEl.scrollTop = logEl.scrollHeight; }
    offset = parseInt(r.headers.get('X-Log-Size') || offset, 10);
    const jr  = await api(`/api/jobs/${jobId}`);
    if (!jr) return;
    const job = await jr.json();
    // Update status badge
    const badge = document.getElementById('status-badge');
    if (badge) { badge.textContent = job.status; badge.className = 'status-' + job.status; }
    // Update output list/gallery
    refreshOutputs(job);
    if (job.status === 'running') setTimeout(pollLog, 3000);
  }

  function refreshOutputs(job) {
    const files = job.output_files || [];
    const el    = document.getElementById('output-list');
    if (!el) return;
    const imgs  = files.filter(f => /\.(png|jpg|jpeg|webp)$/i.test(f));
    const ckpts = files.filter(f => f.endsWith('.safetensors'));
    el.innerHTML =
      (imgs.length ? `<div class="gallery">${imgs.map(f =>
        `<a href="/api/jobs/${jobId}/output/${f}" download>
           <img src="/api/jobs/${jobId}/output/${f}" alt="${f}"></a>`).join('')}</div>` : '') +
      ckpts.map(f =>
        `<p><a href="/api/jobs/${jobId}/output/${f}" download>⬇ ${f}</a></p>`).join('');
  }

  pollLog();
}

// New Training Job form — populate dataset and model pickers
async function trainFormInit() {
  const [dsR, mdR] = await Promise.all([api('/api/datasets'), api('/api/models')]);
  if (!dsR || !mdR) return;
  const datasets = await dsR.json();
  const models   = await mdR.json();
  const dsEl = document.getElementById('dataset');
  const mdEl = document.getElementById('base_model_filename');
  if (dsEl) dsEl.innerHTML = datasets.map(n => `<option value="${n}">${n}</option>`).join('');
  if (mdEl) mdEl.innerHTML = models.map(n => `<option value="${n}">${n}</option>`).join('');

  document.getElementById('train-form').onsubmit = async e => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const params = Object.fromEntries(fd.entries());
    // cast numerics
    ['epochs','batch_size','network_dim','network_alpha','min_snr_gamma'].forEach(k => {
      if (params[k]) params[k] = Number(params[k]);
    });
    const r = await api('/api/jobs', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({type: 'train', params})
    });
    if (r && r.ok) location = '/';
    else alert('Failed to create job');
  };
}

// New Generation Job form — populate model and checkpoint pickers
async function generateFormInit() {
  const [mdR, ckR] = await Promise.all([api('/api/models'), api('/api/checkpoints')]);
  if (!mdR || !ckR) return;
  const models = await mdR.json();
  const ckpts  = await ckR.json();
  const mdEl   = document.getElementById('base_model_filename');
  const ckEl   = document.getElementById('lora_checkpoint');
  if (mdEl) mdEl.innerHTML = models.map(n => `<option value="${n}">${n}</option>`).join('');
  if (ckEl) ckEl.innerHTML = ckpts.map(c =>
    `<option value="${c.job_id}|${c.filename}">${c.filename} (job ${c.job_id.slice(0,8)})</option>`
  ).join('');

  document.getElementById('gen-form').onsubmit = async e => {
    e.preventDefault();
    const fd     = new FormData(e.target);
    const params = Object.fromEntries(fd.entries());
    const sel    = params.lora_checkpoint;
    if (sel) {
      const [jid, fname] = sel.split('|');
      params.lora_checkpoints = [[jid, fname]];
    }
    delete params.lora_checkpoint;
    ['width','height','steps','batch_size','images_per_prompt'].forEach(k => {
      if (params[k]) params[k] = Number(params[k]);
    });
    params.cfg_scale = Number(params.cfg_scale || 7);
    const r = await api('/api/jobs', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({type: 'generate', params})
    });
    if (r && r.ok) location = '/';
    else alert('Failed to create job');
  };
}

// Model download form
async function modelDownloadInit() {
  document.getElementById('dl-form').onsubmit = async e => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const r  = await api('/api/models/download', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({url: fd.get('url'), filename: fd.get('filename')})
    });
    if (r && r.ok) location = '/';
    else { const d = await r.json(); alert(d.error || 'Failed'); }
  };
}

// Dataset upload form
async function datasetUploadInit() {
  document.getElementById('upload-form').onsubmit = async e => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const r  = await api('/api/datasets', {method: 'POST', body: fd});
    if (r && r.ok) { alert(`Uploaded: ${(await r.json()).name}`); location = '/'; }
    else { const d = await r.json(); alert(d.error || 'Failed'); }
  };
}
```

- [ ] **Step 3: Overwrite `control_panel/templates/base.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{% block title %}GPU Control Panel{% endblock %}</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
{% block nav %}
<nav>
  <strong>GPU Control Panel</strong>
  <a href="/">Dashboard</a>
  <button onclick="fetch('/api/logout',{method:'POST'}).then(()=>location='/login')">Logout</button>
</nav>
{% endblock %}
<main>{% block content %}{% endblock %}</main>
<script src="{{ url_for('static', filename='app.js') }}"></script>
{% block scripts %}{% endblock %}
</body>
</html>
```

- [ ] **Step 4: Overwrite `control_panel/templates/login.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><title>Login — GPU Control Panel</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body class="center">
<div id="login-form">
  <h2>GPU Control Panel</h2>
  <input type="password" id="pw" placeholder="Password" autofocus><br>
  <button onclick="doLogin()">Login</button>
  <p id="err" style="color:#ef4444;display:none">Wrong password</p>
</div>
<script>
async function doLogin() {
  const r = await fetch('/api/login', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({password: document.getElementById('pw').value})
  });
  if (r.ok) location = '/';
  else document.getElementById('err').style.display = '';
}
document.addEventListener('keydown', e => { if (e.key === 'Enter') doLogin(); });
</script>
</body>
</html>
```

- [ ] **Step 5: Overwrite `control_panel/templates/dashboard.html`**

```html
{% extends "base.html" %}
{% block title %}Dashboard{% endblock %}
{% block content %}
<div class="toolbar">
  <button onclick="location='/jobs/train/new'">+ Training Job</button>
  <button onclick="location='/jobs/generate/new'">+ Generation Job</button>
  <button onclick="location='/models/download'">↓ Download Model</button>
  <button onclick="location='/datasets/upload'">↑ Upload Dataset</button>
</div>
<table>
  <thead><tr><th>Type</th><th>Name / ID</th><th>Status</th><th>Progress</th><th>Actions</th></tr></thead>
  <tbody id="jobs-body"></tbody>
</table>
{% endblock %}
{% block scripts %}<script>dashboardInit();</script>{% endblock %}
```

- [ ] **Step 6: Overwrite `control_panel/templates/job_new_train.html`**

```html
{% extends "base.html" %}
{% block title %}New Training Job{% endblock %}
{% block content %}
<h2>New Training Job</h2>
<form id="train-form">
  <label>Dataset <select id="dataset" name="dataset" required></select></label>
  <label>Base model <select id="base_model_filename" name="base_model_filename" required></select></label>
  <label>Output name <input name="output_name" placeholder="mywebtoon_v1" required></label>
  <label>Epochs <input type="number" name="epochs" value="30" min="1" required></label>
  <label>Network dim <input type="number" name="network_dim" value="32" min="1"></label>
  <label>Network alpha <input type="number" name="network_alpha" value="32" min="1"></label>
  <label>Learning rate <input name="learning_rate" value="3e-5"></label>
  <label>Min SNR gamma <input type="number" name="min_snr_gamma" value="5" step="0.5"></label>
  <label>Batch size <input type="number" name="batch_size" value="4" min="1"></label>
  <label>Resolution <input name="resolution" value="1024,1024"></label>
  <button type="submit">Queue Training Job</button>
</form>
{% endblock %}
{% block scripts %}<script>trainFormInit();</script>{% endblock %}
```

- [ ] **Step 7: Overwrite `control_panel/templates/job_new_generate.html`**

```html
{% extends "base.html" %}
{% block title %}New Generation Job{% endblock %}
{% block content %}
<h2>New Generation Job</h2>
<form id="gen-form">
  <label>Base model <select id="base_model_filename" name="base_model_filename" required></select></label>
  <label>LoRA checkpoint <select id="lora_checkpoint" name="lora_checkpoint"></select></label>
  <label>Prompt <textarea name="prompt" rows="3" placeholder="mywebtoon, 1girl, solo, long hair" required></textarea></label>
  <label>Negative prompt <textarea name="negative_prompt" rows="2" placeholder="worst quality, low quality"></textarea></label>
  <label>Width <input type="number" name="width" value="832"></label>
  <label>Height <input type="number" name="height" value="1216"></label>
  <label>Steps <input type="number" name="steps" value="28"></label>
  <label>CFG scale <input type="number" name="cfg_scale" value="7" step="0.5"></label>
  <label>Images per prompt <input type="number" name="images_per_prompt" value="4"></label>
  <button type="submit">Queue Generation Job</button>
</form>
{% endblock %}
{% block scripts %}<script>generateFormInit();</script>{% endblock %}
```

- [ ] **Step 8: Overwrite `control_panel/templates/job_detail.html`**

```html
{% extends "base.html" %}
{% block title %}Job Detail{% endblock %}
{% block content %}
<h2>Job <code>{{ job_id[:8] }}</code> — <span id="status-badge"></span></h2>
<h3>Log</h3>
<pre id="log"></pre>
<h3>Outputs</h3>
<div id="output-list"></div>
{% endblock %}
{% block scripts %}
<script>jobDetailInit("{{ job_id }}");</script>
{% endblock %}
```

- [ ] **Step 9: Overwrite `control_panel/templates/model_download.html`**

```html
{% extends "base.html" %}
{% block title %}Download Base Model{% endblock %}
{% block content %}
<h2>Download Base Model</h2>
<p>Paste a direct download URL (e.g. Civitai) and give the file a name ending in <code>.safetensors</code>.</p>
<form id="dl-form">
  <label>URL <input name="url" placeholder="https://civitai.com/api/download/models/..." required></label>
  <label>Save as <input name="filename" placeholder="illustrious-xl-v1.safetensors" required></label>
  <button type="submit">Queue Download</button>
</form>
{% endblock %}
{% block scripts %}<script>modelDownloadInit();</script>{% endblock %}
```

- [ ] **Step 10: Overwrite `control_panel/templates/dataset_upload.html`**

```html
{% extends "base.html" %}
{% block title %}Upload Dataset{% endblock %}
{% block content %}
<h2>Upload Dataset</h2>
<p>Upload a <code>.zip</code> produced by <code>run.py package</code>.
   It must contain a <code>&lt;repeats&gt;_&lt;trigger&gt;</code> folder
   (e.g. <code>5_mywebtoon/</code>) with image + <code>.txt</code> caption pairs.</p>
<form id="upload-form" enctype="multipart/form-data">
  <label>Dataset zip <input type="file" name="file" accept=".zip" required></label>
  <button type="submit">Upload</button>
</form>
{% endblock %}
{% block scripts %}<script>datasetUploadInit();</script>{% endblock %}
```

- [ ] **Step 11: Smoke-test by running the app locally and opening the browser**

```bash
cd webtoon_gpu_pipeline
pip install flask
CONTROL_PANEL_PASSWORD=test python -c "
from control_panel.app import create_app
create_app().run(host='127.0.0.1', port=7860, debug=True)
"
```

Open `http://127.0.0.1:7860` — login with `test`, verify:
- Dashboard loads with empty job table and four action buttons
- "Upload Dataset" page loads and shows the form
- "Download Model" page loads and shows the form
- "New Training Job" page loads and dropdowns are (empty until you add datasets/models)
- "New Generation Job" page loads

- [ ] **Step 12: Commit**

```bash
git add control_panel/static/ control_panel/templates/
git commit -m "feat: web UI — dashboard, job forms, log viewer, output gallery"
```

---

## Task 8: Deployment scripts for E2E Networks

**Files:**
- Create: `deploy/setup.sh`
- Create: `deploy/start.sh`

No tests — these are shell scripts verified by running them on the actual target node.

- [ ] **Step 1: Create `deploy/setup.sh`**

```bash
#!/usr/bin/env bash
# Run once on the E2E TIR node (PyTorch template container).
# Usage: bash deploy/setup.sh [REPO_URL]
set -e

REPO="${1:-https://github.com/KenanMathews/webtoon-gpu-pipeline}"
APP="/workspace/app"
SD="/workspace/sd-scripts"
WORK="/workspace/work"

echo "=== Cloning app ==="
git clone "$REPO" "$APP"

echo "=== Cloning Kohya sd-scripts ==="
git clone https://github.com/kohya-ss/sd-scripts "$SD"
cd "$SD" && pip install -r requirements.txt && cd "$APP"

echo "=== Installing control panel deps ==="
pip install flask

echo "=== Creating persistent work directories ==="
# Mount your E2E persistent volume at /workspace/work BEFORE running this script.
mkdir -p "$WORK/datasets" "$WORK/models" "$WORK/jobs"

echo ""
echo "Setup complete."
echo "Set CONTROL_PANEL_PASSWORD, then run: bash $APP/deploy/start.sh"
```

- [ ] **Step 2: Create `deploy/start.sh`**

```bash
#!/usr/bin/env bash
# Start the GPU control panel on the E2E node.
# The port must be opened in your TIR Security Group before accessing it.
set -e

export WORK_DIR="${WORK_DIR:-/workspace/work}"
export SD_SCRIPTS_DIR="${SD_SCRIPTS_DIR:-/workspace/sd-scripts}"
export PYTHON_BIN="${PYTHON_BIN:-$(which python)}"
export CONTROL_PANEL_PASSWORD="${CONTROL_PANEL_PASSWORD:?Set CONTROL_PANEL_PASSWORD before starting}"
export SECRET_KEY="${SECRET_KEY:-$(python -c 'import secrets; print(secrets.token_hex(32))')}"
export PORT="${PORT:-7860}"

echo "Starting control panel on port $PORT ..."
echo "Access via: http://<your-node-ip>:$PORT"
echo "(open port $PORT in TIR Security Group if not already done)"

cd /workspace/app
python -c "
from control_panel.app import create_app
app = create_app()
app.run(host='0.0.0.0', port=$PORT)
"
```

- [ ] **Step 3: Make scripts executable and commit**

```bash
chmod +x deploy/setup.sh deploy/start.sh
git add deploy/
git commit -m "feat: E2E Networks deployment scripts (setup.sh + start.sh)"
```

---

## Task 9: Final integration and full suite

- [ ] **Step 1: Run the complete test suite**

```bash
pytest tests/ -v
```

Expected output (all green):
```
tests/test_auth.py::test_login_success PASSED
tests/test_auth.py::test_login_wrong_password PASSED
tests/test_auth.py::test_unauthenticated_api_returns_401 PASSED
tests/test_auth.py::test_authenticated_api_passes PASSED
tests/test_auth.py::test_logout_clears_session PASSED
tests/test_commands.py::test_train_uses_sdxl_script PASSED
... (all tests)
tests/test_runner.py::test_kill_invalid_pid_does_not_raise PASSED
tests/test_runner.py::test_worker_completes_download_model_job PASSED
```

- [ ] **Step 2: Manual smoke-test — full pipeline flow locally**

```bash
CONTROL_PANEL_PASSWORD=localtest python -c "
from control_panel.app import create_app
create_app().run(host='127.0.0.1', port=7860, debug=False)
"
```

Run through in browser:
1. Login with `localtest`
2. Upload Dataset — drag in `work/5_mywebtoon.zip` (produced by `run.py package`), confirm `5_mywebtoon` appears
3. Queue Model Download — use any small `.safetensors` URL, confirm `download_model` job appears queued then running then done
4. New Training Job — select dataset and downloaded model, reduce epochs to 1, queue it
5. Job detail — watch log appear and scroll, confirm checkpoint appears in Outputs when done
6. New Generation Job — select model + checkpoint, enter a prompt, queue it
7. Job detail — confirm images appear in Outputs gallery with download links
8. Queue two training jobs, reorder them, cancel the second one, confirm partial output is kept

- [ ] **Step 3: Push everything**

```bash
git push
```

---

## Self-Review

**Spec coverage:**
- ✅ SQLite job table — Task 1 (db.py)
- ✅ Background worker thread — Task 5 (runner.py)
- ✅ train / generate / download_model job types — Tasks 4 + 5 + 6
- ✅ Job reorder (queued) / cancel (running, keeps output) — Task 4
- ✅ Log tail via `GET /api/jobs/<id>/log?since=` — Task 4
- ✅ Dataset upload (zip, Kohya validation) — Task 6
- ✅ Model download by URL (Civitai direct link) — Task 6
- ✅ Checkpoints list from done train jobs — Task 6
- ✅ Output file download — Task 4 (`/api/jobs/<id>/output/<filename>`)
- ✅ Password auth + session cookie + before_request guard — Task 2
- ✅ Dashboard + log viewer + output gallery — Task 7
- ✅ Train form pre-filled with SDXL defaults (dim 32, alpha 32, lr 3e-5, min_snr_gamma 5) — Task 7
- ✅ Generate form with 832×1216 defaults — Task 7
- ✅ E2E Networks deployment — Task 8
- ✅ Persistent volume note addressed in setup.sh — Task 8
- ✅ Windows-only env vars applied via `platform.system()` — Task 3 (commands.py)
