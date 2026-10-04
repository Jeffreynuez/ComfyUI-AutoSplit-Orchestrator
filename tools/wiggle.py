"""
Wiggle test: show what the rig will look like once parts move.

A split where every part holds only its visible pixels rebuilds the picture
perfectly at rest, whatever the draw order, so a rest-pose check in Spine
cannot catch ordering mistakes or missing hidden areas. This script pushes
every part away from the character's centre ("exploded view") and renders
the result in the run's draw order, beside the rest pose. Wrong order shows
up as a part sliding behind something it should cover; missing hidden areas
show up as holes.

Usage:
    python tools/wiggle.py --run <body pass> [--run <facial pass>] \
        --out wiggle.png [--gt <gt folder>] [--strength 0.08 0.18]

With --gt, a second row renders the hand-cut rig the same way for comparison.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from partsio import GroundTruth, Part, back_to_front, load_pass  # noqa: E402

BG = (83, 83, 83, 255)


def centroid(p):
    a = p.alpha >= 128
    if not a.any():
        return p.x0, p.y0
    ys, xs = np.nonzero(a)
    return p.x0 + xs.mean(), p.y0 + ys.mean()


def render(parts_btf, W, H, strength, center):
    canvas = Image.new("RGBA", (W, H), BG)
    cx, cy = center
    for p in parts_btf:
        px, py = centroid(p)
        dx, dy = (px - cx) * strength, (py - cy) * strength
        im = Image.fromarray(p.rgba)
        x, y = int(round(p.x0 + dx)), int(round(p.y0 + dy))
        cxp, cyp = max(0, -x), max(0, -y)
        canvas.alpha_composite(im.crop((cxp, cyp, im.width, im.height)), (max(0, x), max(0, y)))
    return canvas


def gt_parts(gt):
    out = []
    for l in gt.layers:  # back to front
        im = np.asarray(Image.open(os.path.join(gt.folder, l["file"])).convert("RGBA")).copy()
        p = Part(l["name"], im, l["bbox"][0], l["bbox"][1], 0, False, {"draw_index": l["index"]})
        out.append(p)
    return out


def crop_to(img, box, height):
    c = img.crop(box)
    w = max(1, round(c.width * height / c.height))
    return c.resize((w, height), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--gt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, nargs="+", default=[0.08, 0.18])
    ap.add_argument("--height", type=int, default=900, help="row height in the sheet")
    args = ap.parse_args()

    parts = []
    for folder in args.run:
        parts += load_pass(folder)
    order = back_to_front(parts)
    if args.gt:
        gt = GroundTruth(args.gt)
        W, H = gt.W, gt.H
    else:
        W = max(p.x0 + p.rgba.shape[1] for p in parts)
        H = max(p.y0 + p.rgba.shape[0] for p in parts)

    union = np.zeros((H, W), bool)
    for p in parts:
        union |= p.full_mask(W, H)
    ys, xs = np.nonzero(union)
    center = (xs.mean(), ys.mean())
    pad = int(0.6 * max(np.ptp(xs), np.ptp(ys)) * max(args.strength)) + 20
    box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
           min(W, xs.max() + pad), min(H, ys.max() + pad))

    rows = [("this run", order)]
    if args.gt:
        rows.append(("hand-cut rig", gt_parts(gt)))

    tiles = []
    for label, plist in rows:
        row = [crop_to(render(plist, W, H, 0.0, center), box, args.height)]
        for s in args.strength:
            row.append(crop_to(render(plist, W, H, s, center), box, args.height))
        tiles.append((label, row))

    tw = sum(t.width for t in tiles[0][1]) + 16 * (len(tiles[0][1]) + 1)
    th = (args.height + 40) * len(tiles) + 16
    sheet = Image.new("RGB", (tw, th), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    y = 16
    for label, row in tiles:
        x = 16
        d.text((x, y), "%s   (rest, then exploded x%s)" % (
            label, ", x".join(str(s) for s in args.strength)), fill=(230, 230, 230))
        for t in row:
            sheet.paste(t.convert("RGB"), (x, y + 24))
            x += t.width + 16
        y += args.height + 40
    sheet.save(args.out)
    print("wiggle sheet ->", args.out)


if __name__ == "__main__":
    main()
