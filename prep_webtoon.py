#!/usr/bin/env python3
"""
Webtoon pages -> trainable panel dataset.
Pipeline: numeric-sort -> trim margins -> join (batched) -> gutter-split
          -> filter -> resize -> write Kohya folder.
Captioning is a separate step (WD14, see README at bottom).

Usage:
    python prep_webtoon.py /path/to/author_pages  --out dataset --trigger mywebtoon
"""
import cv2, numpy as np, os, re, glob, argparse

def numeric_sort(files):
    def key(f):
        m = re.findall(r'\d+', os.path.basename(f))
        return [int(x) for x in m] if m else [0]
    return sorted(files, key=key)

def trim_margins(img, thresh=8):
    """Crop near-uniform top/bottom bands so pages abut cleanly when joined."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row_std = g.std(axis=1)
    content = np.where(row_std > thresh)[0]
    if len(content) == 0:
        return img
    return img[content[0]:content[-1] + 1]

def normalize_width(imgs):
    w = min(i.shape[1] for i in imgs)
    out = []
    for i in imgs:
        h = int(i.shape[0] * w / i.shape[1])
        out.append(cv2.resize(i, (w, h), interpolation=cv2.INTER_AREA))
    return out

def find_gutters(strip, std_thresh=6, min_gutter=8):
    """Return cut points: midpoints of contiguous low-variance (gutter) bands."""
    g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    row_std = g.std(axis=1)
    is_gut = row_std < std_thresh
    cuts, run_start = [], None
    for y, val in enumerate(is_gut):
        if val and run_start is None:
            run_start = y
        elif not val and run_start is not None:
            if y - run_start >= min_gutter:
                cuts.append((run_start + y) // 2)
            run_start = None
    return cuts

def split_panels(strip, cuts):
    bounds = [0] + cuts + [strip.shape[0]]
    return [strip[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]

def keep_panel(p, min_h=400, max_white=0.92):
    """Drop slivers and near-blank (text-bubble / gutter) panels."""
    if p.shape[0] < min_h:
        return False
    g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    white_ratio = (g > 240).mean()
    if white_ratio > max_white:
        return False
    if g.std() < 6:  # flat
        return False
    return True

def resize_cap(p, max_dim=1024):
    h, w = p.shape[:2]
    if max(h, w) <= max_dim:
        return p
    s = max_dim / max(h, w)
    return cv2.resize(p, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages_dir")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--trigger", default="mystyle")
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--batch", type=int, default=10, help="pages joined per strip (memory)")
    ap.add_argument("--max_dim", type=int, default=1024)
    args = ap.parse_args()

    out_dir = os.path.join(args.out, f"{args.repeats}_{args.trigger}")
    os.makedirs(out_dir, exist_ok=True)

    files = numeric_sort(
        [f for ext in ("png", "jpg", "jpeg", "webp")
         for f in glob.glob(os.path.join(args.pages_dir, f"*.{ext}"))]
    )
    if not files:
        print("No images found."); return
    print(f"{len(files)} pages found.")

    idx = 0
    for b in range(0, len(files), args.batch):
        chunk = files[b:b + args.batch]
        imgs = [trim_margins(cv2.imread(f)) for f in chunk]
        imgs = [i for i in imgs if i is not None and i.size]
        if not imgs:
            continue
        strip = np.vstack(normalize_width(imgs))
        for panel in split_panels(strip, find_gutters(strip)):
            if keep_panel(panel):
                panel = resize_cap(panel, args.max_dim)
                idx += 1
                cv2.imwrite(os.path.join(out_dir, f"panel_{idx:05d}.png"), panel)
        print(f"  batch {b//args.batch+1}: total panels so far = {idx}")

    print(f"\nDone. {idx} panels -> {out_dir}")
    print(f"Next: caption with WD14 (prefix trigger '{args.trigger}'), then train.")

if __name__ == "__main__":
    main()
