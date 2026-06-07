#!/usr/bin/env python3
"""
Structured labeling for webtoon panels.
Schema: scene_type (controlled), character, background, tags[], trigger.
scene_type values: background | one_character | two_character | three_plus | (extend as needed)
Caption order: trigger, scene_type, character, background, tags
"""
import os, re, glob, json, base64, argparse

SCENE_TYPES = ["background", "one_character", "two_character", "three_plus"]

def numeric_key(f):
    m = re.findall(r'\d+', os.path.basename(f)); return [int(x) for x in m] if m else [0]

def load_images(d):
    files=[f for ext in ("png","jpg","jpeg","webp") for f in glob.glob(os.path.join(d,f"*.{ext}"))]
    return sorted(files, key=numeric_key)

def read_wd14(img):
    txt=os.path.splitext(img)[0]+".txt"
    if os.path.exists(txt):
        return [t.strip() for t in open(txt,encoding="utf-8").read().split(",") if t.strip()]
    return []

def build_labels(files, trigger):
    L={}
    for f in files:
        pid=os.path.splitext(os.path.basename(f))[0]
        L[pid]={"panel_id":pid,"file":os.path.basename(f),"scene_type":"",
                "character":"","background":"","tags":read_wd14(f),"trigger":trigger}
    return L

def assemble(rec):
    parts=[rec["trigger"], rec["scene_type"], rec["character"], rec["background"]]
    cap=[p.strip() for p in parts if p.strip()]
    cap+=[t for t in rec["tags"] if t.strip()]
    return ", ".join(cap)

def emit(L, img_dir):
    n=0
    for pid,rec in L.items():
        open(os.path.join(img_dir,pid+".txt"),"w",encoding="utf-8").write(assemble(rec)); n+=1
    return n

def thumb(path,w=120):
    import cv2
    img=cv2.imread(path)
    if img is None: return ""
    h=int(img.shape[0]*w/img.shape[1]); img=cv2.resize(img,(w,h))
    ok,buf=cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,65])
    return base64.b64encode(buf).decode() if ok else ""

def review_html(L, img_dir, out):
    opts="".join(f"<option value='{s}'>{s}</option>" for s in SCENE_TYPES)
    cards=[]
    for pid,rec in L.items():
        b64=thumb(os.path.join(img_dir,rec["file"]))
        sel="".join(f"<option value='{s}'{' selected' if rec['scene_type']==s else ''}>{s}</option>" for s in [""]+SCENE_TYPES)
        cards.append(f"""<div class=card data-id={pid} data-scene="{rec['scene_type']}">
<img src="data:image/jpeg;base64,{b64}">
<div class=fields><div class=pid>{pid}</div>
<select class=scene_type onchange="this.closest('.card').dataset.scene=this.value">{sel}</select>
<input class=character placeholder=character value="{rec['character']}">
<input class=background placeholder=background value="{rec['background']}">
<textarea class=tags placeholder=tags>{', '.join(rec['tags'])}</textarea></div></div>""")
    html=f"""<!doctype html><meta charset=utf-8><style>
body{{font-family:system-ui;margin:16px;background:#111;color:#eee}}
.bar{{position:sticky;top:0;background:#111;padding:10px;border-bottom:1px solid #333;z-index:9}}
button,select{{padding:6px 10px;font-size:13px}} input,textarea{{background:#262626;border:1px solid #444;color:#eee;border-radius:4px;padding:4px;font-size:12px;width:100%}}
textarea{{height:42px}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px;margin-top:10px}}
.card{{background:#1c1c1c;border:1px solid #333;border-radius:8px;padding:8px;display:flex;gap:8px}}
.card img{{width:120px;height:auto;border-radius:4px}} .fields{{flex:1;display:flex;flex-direction:column;gap:4px}} .pid{{font-size:11px;color:#888}}
.hidden{{display:none}}</style>
<div class=bar>
Filter: <select id=filter onchange=applyFilter()><option value="">all</option>{opts}</select>
&nbsp; Bulk set visible scene_type: <select id=bulk>{opts}</select> <button onclick=bulkApply()>Apply to visible</button>
<button onclick=exportJson()>Export</button> <span id=count style="margin-left:10px;color:#888"></span>
</div><div class=grid>{''.join(cards)}</div><script>
const cards=[...document.querySelectorAll('.card')];
function recount(){{document.getElementById('count').textContent=cards.filter(c=>!c.classList.contains('hidden')).length+' visible / '+cards.length}}
function applyFilter(){{const v=filter.value;cards.forEach(c=>c.classList.toggle('hidden',v&&c.dataset.scene!==v));recount()}}
function bulkApply(){{const v=bulk.value;cards.filter(c=>!c.classList.contains('hidden')).forEach(c=>{{c.querySelector('.scene_type').value=v;c.dataset.scene=v}})}}
function exportJson(){{const o={{}};cards.forEach(c=>{{o[c.dataset.id]={{panel_id:c.dataset.id,scene_type:c.querySelector('.scene_type').value,character:c.querySelector('.character').value.trim(),background:c.querySelector('.background').value.trim(),tags:c.querySelector('.tags').value.split(',').map(t=>t.trim()).filter(Boolean)}}}});const b=new Blob([JSON.stringify(o,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='labels_edited.json';a.click()}}
recount();</script>"""
    open(out,"w",encoding="utf-8").write(html)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("img_dir"); ap.add_argument("--trigger",default="mystyle")
    ap.add_argument("--out",default="work"); ap.add_argument("--emit-captions",action="store_true"); ap.add_argument("--from-json")
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    files=load_images(a.img_dir); print(f"{len(files)} panels."); L=build_labels(files,a.trigger)
    if a.from_json and os.path.exists(a.from_json):
        ed=json.load(open(a.from_json))
        for pid,e in ed.items():
            if pid in L: L[pid].update({k:e[k] for k in ("scene_type","character","background","tags") if k in e})
    json.dump(L,open(os.path.join(a.out,"labels.json"),"w"),indent=2)
    if a.emit_captions: print(f"Wrote {emit(L,a.img_dir)} captions.")
    else: review_html(L,a.img_dir,os.path.join(a.out,"review.html")); print(f"Review UI -> {a.out}/review.html")

if __name__=="__main__": main()
