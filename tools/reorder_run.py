"""
Recompute the draw order of an existing run with autosplit_core.ordering and
write the result as a schema-2 copy (parts are copied untouched).

Lets you compare ordering methods on old runs without re-running ComfyUI:

    python tools/reorder_run.py --image <the picture that was split> \
        --run <pass folder> --out <new folder> [--facial]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ComfyUI-AutoSplit-Orchestrator"))
from autosplit_core import ordering  # noqa: E402
from partsio import load_pass  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="", help='e.g. "notch=0,depth=0.4"')
    ap.add_argument("--depth", help="depth map image saved from the run (optional)")
    ap.add_argument("--depth-near", default="bright", choices=["bright", "dark"])
    args = ap.parse_args()
    weights = {k: float(v) for k, v in (kv.split("=") for kv in args.weights.split(",") if kv)}

    src = Image.open(args.image).convert("RGB")
    W, H = src.size
    parts = load_pass(args.run)
    facial = bool(parts and parts[0].facial)
    if facial:  # order facial parts in their own (crop) space is unnecessary:
        pass    # partsio already mapped them to the full canvas
    masks = {p.tag: p.full_mask(W, H) for p in parts}
    near = None
    if args.depth:
        from autosplit_core import masks as M
        d = np.asarray(Image.open(args.depth).convert("L"), np.float32) / 255.0
        if d.shape != (H, W):
            d = np.asarray(Image.fromarray(d).resize((W, H), Image.BILINEAR), np.float32)
        near = M.nearness(d, args.depth_near)
    res = ordering.compute_order(np.asarray(src), masks, near=near, weights=weights or None)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.run, "parts_metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    entries = meta if isinstance(meta, list) else meta["parts"]
    for e in entries:
        t = e.get("tag")
        if t in res["z_order"]:
            e["z_order"] = res["z_order"][t]
            e["draw_index"] = res["back_to_front"].index(t)
        fp = os.path.join(args.run, e.get("file", ""))
        if os.path.exists(fp):
            shutil.copyfile(fp, os.path.join(args.out, e["file"]))
    out = {"schema_version": 2, "pass": "facial" if facial else "body",
           "order_method": "evidence", "parts": entries,
           "order_edges": res["edges"]}
    if isinstance(meta, dict):
        for k in ("character", "lr_convention", "crop_transform"):
            if k in meta:
                out[k] = meta[k]
    if facial and not out.get("crop_transform"):
        for cand in (os.path.join(args.run, "head_crop_transform.json"),):
            if os.path.exists(cand):
                out["crop_transform"] = json.load(open(cand, encoding="utf-8"))
    with open(os.path.join(args.out, "parts_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("order (back -> front):", " | ".join(res["back_to_front"]))
    for e in res["edges"][:40]:
        print("  %-16s over %-16s conf %.2f  %s" % (e["front"], e["back"], e["confidence"], e["cues"]))
    if res["dropped"]:
        print("dropped (cycles):", [(d["front"], d["back"]) for d in res["dropped"]])


if __name__ == "__main__":
    main()
