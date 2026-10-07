"""
Spine export for the AutoSplit Studio Krita plugin.

Builds a Spine 4.3.22 project (images/ + <character>.json) from one split run,
using Krita's bundled QImage (no Pillow needed). Geometry and draw order come
from comfy_client, the same code the layer import uses, so Krita and Spine
always agree:

  * position: comfy_client.placement (facial parts mapped back from the head
    crop and shrunk to full-image size);
  * draw order: comfy_client.draw_order (facial parts directly in front of the
    face, not on top of every body part).

Still one root bone and one region attachment per part; a starter skeleton is
Phase 3.
"""
import json
import os
import shutil

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage

from . import comfy_client


def _san(tag):
    return str(tag).strip().lower().replace(" ", "_").replace("/", "_")


def _geometry(entries):
    """Full-image boxes in back-to-front draw order."""
    out = []
    for e in comfy_client.draw_order(entries):
        x, y, w, h, sc = comfy_client.placement(e)
        out.append({"tag": _san(e.get("tag", "part")), "src": e.get("_path"),
                    "x": float(x), "y": float(y), "w": float(w), "h": float(h),
                    "resize": bool(e.get("_facial")) and sc != 1.0})
    return out


def export_to_spine(entries, out_dir, character, spine_version="4.3.22", progress=None):
    """Write a Spine project (images/ + <character>.json) and return the JSON path."""
    def say(m):
        if progress:
            progress(m)

    if not entries:
        raise RuntimeError("No parts to export - run a split first.")

    parts = _geometry(entries)
    seen = {}
    for p in parts:  # Spine needs unique slot names
        n = p["tag"]
        if n in seen:
            seen[n] += 1
            p["tag"] = "%s_%d" % (n, seen[n])
        else:
            seen[n] = 1

    minx = min(p["x"] for p in parts)
    maxx = max(p["x"] + p["w"] for p in parts)
    miny = min(p["y"] for p in parts)
    maxy = max(p["y"] + p["h"] for p in parts)
    origin_x = (minx + maxx) / 2.0
    origin_y = maxy   # bottom-centre: feet near y=0

    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    slots, attachments = [], {}
    for p in parts:
        name = p["tag"]
        out_png = os.path.join(img_dir, name + ".png")
        if p["resize"]:
            img = QImage(p["src"]).convertToFormat(QImage.Format_ARGB32)
            img = img.scaled(max(1, int(round(p["w"]))), max(1, int(round(p["h"]))),
                             Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            img.save(out_png, "PNG")
        else:
            shutil.copyfile(p["src"], out_png)

        cx = p["x"] + p["w"] / 2.0
        cy = p["y"] + p["h"] / 2.0
        slots.append({"name": name, "bone": "root", "attachment": name})
        attachments[name] = {
            name: {"x": round(cx - origin_x, 2), "y": round(origin_y - cy, 2),
                   "width": int(round(p["w"])), "height": int(round(p["h"])),
                   "rotation": 0}
        }

    skeleton = {
        "skeleton": {
            "spine": spine_version,
            "x": round(minx - origin_x, 2), "y": 0.0,
            "width": round(maxx - minx, 2), "height": round(maxy - miny, 2),
            "images": "./images/", "audio": "",
        },
        "bones": [{"name": "root"}],
        "slots": slots,
        "skins": [{"name": "default", "attachments": attachments}],
        "animations": {},
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, character + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(skeleton, f, indent=2)
    say("Spine project: %s (%d parts) -> %s" % (os.path.basename(json_path), len(parts), out_dir))
    return json_path
