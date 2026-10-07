"""
Extract a hand-cut rig from a layered PSD as ground truth for evaluate.py.

A rigger's PSD already holds the answer AutoSplit is trying to reach: every
part cut whole, named with the character's own left/right, stacked in draw
order. This script pulls one view (a group such as "FA") out of the PSD into a
plain folder the metrics harness can read without Photoshop or psd-tools:

    <out>/composite.png        the view rendered alone, RGBA, full canvas
    <out>/composite_flat.png   the same on a flat background (the test input)
    <out>/layers/NNN_<slug>.png one RGBA crop per layer, NNN = draw order
                                (000 = back-most)
    <out>/gt.json              canvas size, background colour and, per layer:
                                index, name, slug, bbox, blend mode, opacity,
                                is_overlay (shadow/highlight layers)

Keep the output OUT of git: it is the artist's artwork.

Usage:
    pip install psd-tools
    python tools/gt_from_psd.py "Salena Turnaround AI image test.psd" \
        --group FA --out "C:/dev/AI Work/AutoSplit/ground_truth/salena_fa"
"""
import argparse
import json
import os
import re

import numpy as np
from PIL import Image

OVERLAY_WORDS = ("shadow", "highlight", "shade", "light ", "glow")


def slugify(name):
    s = re.sub(r"\[[^\]]*\]", "", name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "layer"


def clean_name(name):
    return re.sub(r"\[[^\]]*\]", "", name).strip()


def find_group(psd, wanted):
    wanted = wanted.strip().lower()
    for layer in psd.descendants():
        if layer.is_group() and clean_name(layer.name).lower() == wanted:
            return layer
    raise SystemExit("No group named %r in the PSD" % wanted)


def pixel_layers(group):
    """Pixel/shape layers inside the group, back-most first, skipping hidden
    layers and anything inside hidden subgroups."""
    out = []

    def walk(g):
        for layer in g:  # psd-tools iterates bottom (back) to top (front)
            if not layer.visible:
                continue
            if layer.is_group():
                walk(layer)
            else:
                out.append(layer)
    walk(group)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("psd")
    ap.add_argument("--group", required=True, help="view group name, e.g. FA")
    ap.add_argument("--out", required=True)
    ap.add_argument("--background", default="#FFFFFF",
                    help="flat background for composite_flat.png")
    args = ap.parse_args()

    from psd_tools import PSDImage

    psd = PSDImage.open(args.psd)
    group = find_group(psd, args.group)
    W, H = psd.width, psd.height
    os.makedirs(os.path.join(args.out, "layers"), exist_ok=True)

    # Render the view on its own: hide every other branch of the tree.
    keep = set()
    node = group
    while node is not None and node is not psd:
        keep.add(id(node))
        node = node.parent
    for d in group.descendants():
        keep.add(id(d))
    saved = {}
    for layer in psd.descendants():
        if id(layer) not in keep and layer.visible:
            saved[id(layer)] = layer
            layer.visible = False
    comp = psd.composite(force=True, color=0.0, alpha=0.0)
    for layer in saved.values():
        layer.visible = True
    comp = comp.convert("RGBA")
    if comp.size != (W, H):
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        canvas.paste(comp, (0, 0))
        comp = canvas
    comp.save(os.path.join(args.out, "composite.png"))

    bg = args.background.lstrip("#")
    bg_rgb = tuple(int(bg[i:i + 2], 16) for i in (0, 2, 4))
    flat = Image.new("RGBA", (W, H), bg_rgb + (255,))
    flat.alpha_composite(comp)
    flat.convert("RGB").save(os.path.join(args.out, "composite_flat.png"))

    layers = []
    for idx, layer in enumerate(pixel_layers(group)):
        img = layer.composite(force=True)
        if img is None:
            continue
        img = img.convert("RGBA")
        x0, y0 = layer.left, layer.top
        a = np.asarray(img)[..., 3]
        if a.max() == 0:
            continue
        name = clean_name(layer.name)
        slug = slugify(layer.name)
        fn = "%03d_%s.png" % (idx, slug)
        img.save(os.path.join(args.out, "layers", fn))
        layers.append({
            "index": idx,
            "name": name,
            "slug": slug,
            "file": "layers/" + fn,
            "bbox": [int(x0), int(y0), int(x0 + img.width), int(y0 + img.height)],
            "blend_mode": str(getattr(layer, "blend_mode", "")).split(".")[-1].lower(),
            "opacity": int(getattr(layer, "opacity", 255)),
            "is_overlay": any(w in name.lower() + " " for w in OVERLAY_WORDS),
        })

    meta = {
        "source": os.path.basename(args.psd),
        "group": clean_name(group.name),
        "canvas": [W, H],
        "background": "#" + bg.upper(),
        "draw_order": "index 0 is back-most; higher index draws in front",
        "lr_convention": "anatomical (as named by the rigger)",
        "layers": layers,
    }
    with open(os.path.join(args.out, "gt.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("wrote %d layers + composite to %s" % (len(layers), args.out))


if __name__ == "__main__":
    main()
