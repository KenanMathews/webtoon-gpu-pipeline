# Webtoon Panel Labeling Kit

Run entirely on your own machine. Turns webtoon pages/panels into a labeled,
Kohya-ready training dataset.

## Schema
Each panel gets structured fields:
- **scene_type** (controlled vocab): `background | one_character | two_character | three_plus`
- **character**: who's in the panel (free text)
- **background**: setting / environment (free text)
- **tags**: WD14 auto-tags, cleaned (free text, comma-separated)
- **trigger**: your style token, prepended to every caption

Caption is assembled in weighted order:
`trigger, scene_type, character, background, tags`

Empty fields drop out automatically (e.g. a `background` panel has no character).

## Setup (one time)
```bash
# Python 3.10+ recommended
pip install opencv-python-headless numpy

# Only needed for the optional `tag` stage (WD14 auto-tagging):
git clone https://github.com/kohya-ss/sd-scripts
cd sd-scripts && pip install -r requirements.txt
pip install onnx onnxruntime    # ONNX is the fast tagger path
cd ..
```

## Workflow

### If you already have split panels (recommended for your 600)
Put them in `./panels/` then:
```bash
# 1. (optional) auto-tag to prefill the `tags` field
python run.py tag --img panels --sd-scripts /path/to/sd-scripts

# 2. build the review UI
python run.py review --img panels --trigger mywebtoon

# 3. open work/review.html in a browser
#    - Filter by scene_type, bulk-assign, fill character/background
#    - Click "Export" -> downloads labels_edited.json

# 4. turn your labels into Kohya captions
python run.py emit --img panels --trigger mywebtoon --json labels_edited.json
```

### If you have raw author PAGES (long vertical strips)
```bash
python run.py prep --pages author_pages --trigger mywebtoon   # joins + splits
# then point review/emit at the produced folder:
python run.py review --img dataset/6_mywebtoon --trigger mywebtoon
python run.py emit   --img dataset/6_mywebtoon --trigger mywebtoon --json labels_edited.json
```

## Labeling tips (600 panels, ~scenario LoRA)
- **scene_type first**: it's just counting people — fastest pass. Use the filter +
  "Apply to visible" bulk button to set it in chunks.
- Do all `background` (no-people) panels first; they need no character field.
- Reuse exact vocabulary ("rooftop" not "roof top") so prompts match training later.
- Group panels of the same scene type share most fields — label one, copy to siblings.
- Trim `tags` to ~10-20 so scene_type/character/background dominate the signal.

## Files
- `run.py`          one-shot runner (prep / tag / review / emit)
- `prep_webtoon.py` join author pages -> split into panels
- `label_panels.py` build labels.json + review.html, emit Kohya captions
- `control_panel/`  Flask app for running training/generation on a rented GPU
  (see `docs/superpowers/specs/2026-06-07-gpu-control-panel-design.md`)

## Notes
- `scene_type` values live in `label_panels.py` (`SCENE_TYPES` list) — edit to taste.
- The review.html inlines thumbnails; for 600 panels it's a few MB. If sluggish,
  label in batches (e.g. split panels into subfolders by chapter and run per folder).
- Nothing here uploads anywhere — fully local.

## Data layout
All datasets, panels, captions, checkpoints, generated images, and downloaded
models live under `data/` (and the legacy `panels/` / `work/` / `test_models/`
locations if you're pointing at an existing setup). **Nothing under those
paths is ever committed or pushed** — see `.gitignore`.

Loose local-only data artifacts that might land outside those folders (e.g. a
labels/metadata JSON) should use a `local.` prefix — e.g. `local.labels.json`
— which `.gitignore` matches via `**/local.*` regardless of location.
