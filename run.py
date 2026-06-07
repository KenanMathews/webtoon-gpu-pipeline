#!/usr/bin/env python3
"""
One-shot local runner for webtoon panel labeling.

STAGES:
  prep     : join author pages -> split into panels       (needs: prep_webtoon.py)
  tag      : WD14 auto-tag panels (fills the `tags` field) (needs: sd-scripts)
  review   : build labels.json + review.html               (label_panels.py)
  emit     : merge your edited json -> Kohya .txt captions  (label_panels.py)

TYPICAL FLOW (already-split panels in ./panels):
  python run.py review --img panels --trigger mywebtoon
  # open work/review.html in your browser, label, click Export -> labels_edited.json
  python run.py emit   --img panels --trigger mywebtoon --json labels_edited.json

IF you have raw author PAGES (long vertical strips) instead of panels:
  python run.py prep --pages author_pages --trigger mywebtoon   # -> dataset/6_mywebtoon
  then run review/emit with --img dataset/6_mywebtoon

OPTIONAL auto-tagging (run from inside your sd-scripts checkout, or pass --sd-scripts PATH):
  python run.py tag --img panels --sd-scripts /path/to/sd-scripts
"""
import argparse, subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))

def sh(cmd):
    print(">>", " ".join(cmd)); subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep","tag","review","emit"])
    ap.add_argument("--img", default="panels")
    ap.add_argument("--pages", default="author_pages")
    ap.add_argument("--trigger", default="mystyle")
    ap.add_argument("--out", default="work")
    ap.add_argument("--json", default="labels_edited.json")
    ap.add_argument("--sd-scripts", default=".", help="path to sd-scripts checkout")
    ap.add_argument("--repeats", type=int, default=6)
    a = ap.parse_args()
    py = sys.executable

    if a.stage == "prep":
        sh([py, os.path.join(HERE,"prep_webtoon.py"), a.pages,
            "--out","dataset","--trigger",a.trigger,"--repeats",str(a.repeats)])

    elif a.stage == "tag":
        tagger = os.path.join(a.sd_scripts, "finetune", "tag_images_by_wd14_tagger.py")
        if not os.path.exists(tagger):
            sys.exit(f"WD14 tagger not found at {tagger}\n"
                     f"Clone it:  git clone https://github.com/kohya-ss/sd-scripts\n"
                     f"then pass --sd-scripts /path/to/sd-scripts")
        sh([py, tagger, "--onnx",
            "--repo_id","SmilingWolf/wd-swinv2-tagger-v3",
            "--batch_size","4","--remove_underscore",
            "--character_tags_first", a.img])

    elif a.stage == "review":
        sh([py, os.path.join(HERE,"label_panels.py"), a.img,
            "--trigger",a.trigger,"--out",a.out])
        print(f"\nOpen {a.out}/review.html in your browser. Label, then Export.")

    elif a.stage == "emit":
        sh([py, os.path.join(HERE,"label_panels.py"), a.img,
            "--trigger",a.trigger,"--out",a.out,
            "--from-json",a.json,"--emit-captions"])
        print(f"\nDone. Kohya .txt captions written next to images in {a.img}/")

if __name__ == "__main__":
    main()
