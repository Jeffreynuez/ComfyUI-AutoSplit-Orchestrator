"""
Spine export for the AutoSplit Studio Krita plugin.

Same validated logic as the standalone spine_export.py, but uses Krita's bundled
QImage for facial downscaling (no Pillow needed). Builds a Spine 4.3.22 project
from the SAM3 split output: images/ + <character>.json.
"""
import json
import os
import shutil

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage


def _san(tag):
    return str(tag).strip().lower().replace(" ", "_").replace("/", "_")


def _geometry(entries, transform):
    """Map split parts into full-image geometry and back-to-front draw order
    (body back-to-front, then facial features on top)."""
    sc = transform.get("scale", 1.0) or 1.0
    ox = transform.get("x_min", 0)
    oy = transform.get("y_min", 0)
    body, facial = [], []
    for e in entries:
        xyxy = e.get("xyxy") or [0, 0, 0, 0]
        src = e.get("_path")
        img = QImage(src)
        pw, ph = img.width(), img.height()
        is_facial = bool(e.get("_facial"))
        if is_facial and sc != 1.0:
            w, h, x, y = pw / sc, ph / sc, ox + xyxy[0] / sc, oy + xyxy[1] / sc
        elif is_facial:
            w, h, x, y = pw, ph, ox + xyxy[0], oy + xyxy[1]
        else:
            w, h, x, y = pw, ph, xyxy[0], xyxy[1]
        rec = {
            "tag": _san(e.get("tag", "part")), "src": src, "facial": is_facial,
            "x": float(x), "y": float(y), "w": float(w), "h": float(h),
            "z": e.get("z_order", e.get("z_rank", 0)),
        }
        (facial if is_facial else body).append(rec)
    body.sort(key=lambda p: -p["z"])
    facial.sort(key=lambda p: -p["z"])
    return body + facial, sc


def export_to_spine(entries, transform, out_dir, character,
                    spine_version="4.3.22", progress=None):
    """Write a Spine project (images/ + <character>.json) and return the JSON path."""
    def say(m):
        if progress:
            progress(m)

    if not entries:
        raise RuntimeError("No parts to export - run a split first.")

    parts, sc = _geometry(entries, transform)

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
        if p["facial"] and sc != 1.0:
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
    json.dump(skeleton, open(json_path, "w", encoding="utf-8"), indent=2)
    say("Spine project: %s (%d parts) -> %s" % (os.path.basename(json_path), len(parts), out_dir))
    return json_path
