"""
Show where a run's parts differ from the hand-cut rig, part by part.

For each grouped part one row of four panels, cropped to the part:

  rig      the rigger's whole part (his paint where it is hidden, the picture
           where it is visible)
  run      the run's part
  diff     green  both cover it and the colour matches (what Rig Match pays)
           yellow both cover it but the colour is wrong
           red    only the run covers it (extra)
           blue   only the rig covers it (missing)
           dark   the rig's hidden pixels are outlined in white
  score    the part's Rig Match, how much of the total it loses, and the
           share of the part's area in each diff colour

    python tools/part_diff.py --gt <gt folder> --map tools/gt_maps/salena_fa.json \
        --run <body pass> [--run <facial pass>] [--tags torso,hair_back] \
        [--worst 6] --out sheet.jpg

Needs numpy, Pillow and OpenCV.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import colour_sim, norm_tag  # noqa: E402
from partsio import GroundTruth, back_to_front, load_pass  # noqa: E402

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def groups_from_map(gt, mp):
    mapping = {k.lower(): v for k, v in mp["parts"].items()}
    solid = [l for l in gt.layers if not l.get("is_overlay")]
    groups = {}
    for tag, pats in mapping.items():
        rx = [re.compile(p, re.I) for p in pats]
        groups[tag] = [l for l in solid if any(r.fullmatch(l["name"].strip()) for r in rx)]
    return mapping, groups


def part_arrays(gt, layers, plist, order_pos):
    """(rig mask, rig reference colour, rig hidden mask, run mask, run colour)."""
    W, H = gt.W, gt.H
    top = gt.top_index_map()
    src = gt.composite()[..., :3]
    g = np.zeros((H, W), bool)
    ref = np.zeros((H, W, 3), np.uint8)
    for l in sorted(layers, key=lambda l: l["index"]):
        g |= gt.layer_mask(l)
        gt.paint_layer(l, ref)
    vis = np.isin(top, [l["index"] for l in layers])
    ref[vis] = src[vis]
    p = np.zeros((H, W), bool)
    col = np.zeros((H, W, 3), np.uint8)
    for part in sorted(plist, key=lambda q: order_pos[id(q)]):
        part.paint_onto(col, p)
    return g, ref, g & ~vis, p, col


def diff_panel(g, ref, hid, p, col, bg):
    both = g & p
    sim = np.zeros(g.shape)
    sim[both] = colour_sim(col[both], ref[both])
    out = np.empty(g.shape + (3,), np.uint8)
    out[:] = bg
    out[both & (sim >= 0.5)] = (70, 170, 90)
    out[both & (sim < 0.5)] = (230, 200, 40)
    out[p & ~g] = (220, 60, 60)
    out[g & ~p] = (60, 110, 230)
    if cv2 is not None and hid.any():
        edge = cv2.morphologyEx(hid.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
        out[edge] = (255, 255, 255)
    counts = {"ok": int((both & (sim >= 0.5)).sum()), "colour": int((both & (sim < 0.5)).sum()),
              "extra": int((p & ~g).sum()), "missing": int((g & ~p).sum()),
              "missing_hidden": int((hid & ~p).sum())}
    return out, float(sim[both].sum()), int((g | p).sum()), counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--tags", default="", help="comma-separated parts (default: the worst ones)")
    ap.add_argument("--worst", type=int, default=6, help="how many parts when --tags is not given")
    ap.add_argument("--tile", type=int, default=300, help="panel size in pixels")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = GroundTruth(args.gt)
    with open(args.map, encoding="utf-8") as f:
        mp = json.load(f)
    mapping, groups = groups_from_map(gt, mp)
    aliases = {k.lower(): v.lower() for k, v in mp.get("aliases", {}).items()}
    parts = []
    for folder in args.run:
        parts += load_pass(folder)
    pos = {id(p): i for i, p in enumerate(back_to_front(parts))}
    pred = {}
    for p in parts:
        key = norm_tag(p.tag, mapping, aliases)
        if key is not None:
            pred.setdefault(key, []).append(p)

    hexbg = str(gt.meta.get("background") or "#535353")
    bg = np.array([int(hexbg[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
    rows = []
    for tag in pred:
        if not groups.get(tag):
            continue
        g, ref, hid, p, col = part_arrays(gt, groups[tag], pred[tag], pos)
        d, earned, union, counts = diff_panel(g, ref, hid, p, col, bg)
        rows.append({"tag": tag, "arrays": (g, ref, p, col, d), "earned": earned, "union": union,
                     "counts": counts})
    total = sum(r["union"] for r in rows)
    for r in rows:
        r["match"] = r["earned"] / max(1, r["union"])
        r["loss"] = (r["union"] - r["earned"]) / max(1, total)
    if args.tags:
        want = [t.strip().lower() for t in args.tags.split(",") if t.strip()]
        rows = [r for t in want for r in rows if r["tag"] == t]
    else:
        rows = sorted(rows, key=lambda r: -r["loss"])[:args.worst]

    T = args.tile
    sheet = Image.new("RGB", (T * 3 + 230, T * len(rows)), (30, 30, 30))
    dr = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        g, ref, p, col, d = r["arrays"]
        ys, xs = np.nonzero(g | p)
        pad = 20
        y0, y1 = max(0, ys.min() - pad), min(g.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(g.shape[1], xs.max() + pad)
        panels = []
        for m, c in ((g, ref), (p, col)):
            a = np.empty(g.shape + (3,), np.uint8)
            a[:] = bg
            a[m] = c[m]
            panels.append(a)
        panels.append(d)
        for j, a in enumerate(panels):
            im = Image.fromarray(a[y0:y1, x0:x1])
            im.thumbnail((T, T))
            sheet.paste(im, (j * T + (T - im.width) // 2, i * T + (T - im.height) // 2))
        dr.text((3 * T + 10, i * T + 10), r["tag"], fill=(255, 255, 255))
        dr.text((3 * T + 10, i * T + 30), "Rig Match %.3f" % r["match"], fill=(220, 220, 220))
        dr.text((3 * T + 10, i * T + 50), "loses %.1f%% of total" % (100 * r["loss"]), fill=(220, 220, 220))
        c = r["counts"]
        line = "  ".join("%s %.0f%%" % (k, 100.0 * v / max(1, r["union"])) for k, v in c.items())
        dr.text((3 * T + 10, i * T + 70), line.replace("  ", "\n"), fill=(180, 180, 180))
        print("%-18s match %.3f  loses %.1f%%   %s" % (r["tag"], r["match"], 100 * r["loss"], line))
    dr.text((10, 2), "rig | run | diff (green ok, yellow colour, red extra, blue missing, white = rig hidden)",
            fill=(255, 255, 255))
    sheet.save(args.out, quality=88)
    print("sheet ->", args.out)


if __name__ == "__main__":
    main()
