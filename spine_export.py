"""
2D AutoSplit Studio - standalone Spine export (Pillow, no Krita).

Reads one split run and writes a Spine 4.3.22 project:
    <OUT_DIR>/images/<part>.png
    <OUT_DIR>/<CHARACTER>.json     root bone, one slot + region attachment per
                                   part, default skin, y-up coordinates

    python spine_export.py --run-id 20261004-153012-a1b2 [--character Salena]
    python spine_export.py --body <folder> --face <folder>
    python spine_export.py                       (legacy fixed folders)

Geometry and draw order come from the same client the Krita plugin uses
(comfy_client.placement / draw_order): facial parts are mapped back from the
head crop and drawn directly in front of the face.

Coordinate notes: image space is y-down from the top-left; Spine is y-up. The
origin is the character's bounding-box bottom-centre, so the feet land near y=0.
"""
import argparse
import json
import os
import shutil
import sys

from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "krita_plugin", "autosplit_studio"))
import comfy_client  # noqa: E402
from comfy_autosplit_test import COMFY_OUTPUT_DIR  # noqa: E402

BODY_SUBDIR = "split_parts_sam3"
FACIAL_SUBDIR = "split_parts_sam3_facial"
SPINE_VERSION = "4.3.22"


def log(m):
    print("[spine] " + str(m), flush=True)


def sanitize(tag):
    return str(tag).strip().lower().replace(" ", "_").replace("/", "_")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default=os.environ.get("AUTOSPLIT_RUN_ID"))
    ap.add_argument("--body")
    ap.add_argument("--face")
    ap.add_argument("--out", default=os.environ.get("AUTOSPLIT_OUT",
                                                    os.path.join(SCRIPT_DIR, "spine_export")))
    ap.add_argument("--character", default=os.environ.get("AUTOSPLIT_CHARACTER", "character"))
    args = ap.parse_args()

    if args.body or args.face:
        passes = [{"folder": f, "pass": p} for f, p in ((args.body, "body"), (args.face, "facial")) if f]
    elif args.run_id:
        passes = [{"folder": os.path.join(BODY_SUBDIR, args.run_id), "pass": "body"},
                  {"folder": os.path.join(FACIAL_SUBDIR, args.run_id), "pass": "facial"}]
    else:
        passes = [{"folder": BODY_SUBDIR, "pass": "body"}, {"folder": FACIAL_SUBDIR, "pass": "facial"}]
    entries = comfy_client.read_parts(COMFY_OUTPUT_DIR, passes=passes)
    if not entries:
        log("no parts found in %s" % [p["folder"] for p in passes])
        sys.exit(1)

    parts = []
    seen = {}
    for e in comfy_client.draw_order(entries):
        x, y, w, h, sc = comfy_client.placement(e)
        name = sanitize(e.get("tag", "part"))
        if name in seen:
            seen[name] += 1
            name = "%s_%d" % (name, seen[name])
        else:
            seen[name] = 1
        parts.append({"name": name, "src": e["_path"], "x": x, "y": y, "w": w, "h": h,
                      "resize": bool(e.get("_facial")) and sc != 1.0})

    minx = min(p["x"] for p in parts)
    maxx = max(p["x"] + p["w"] for p in parts)
    miny = min(p["y"] for p in parts)
    maxy = max(p["y"] + p["h"] for p in parts)
    origin_x, origin_y = (minx + maxx) / 2.0, maxy

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    slots, attachments = [], {}
    for p in parts:
        dst = os.path.join(img_dir, p["name"] + ".png")
        if p["resize"]:
            with Image.open(p["src"]) as im:
                im.convert("RGBA").resize((max(1, p["w"]), max(1, p["h"])), Image.LANCZOS).save(dst)
        else:
            shutil.copyfile(p["src"], dst)
        cx, cy = p["x"] + p["w"] / 2.0, p["y"] + p["h"] / 2.0
        slots.append({"name": p["name"], "bone": "root", "attachment": p["name"]})
        attachments[p["name"]] = {p["name"]: {
            "x": round(cx - origin_x, 2), "y": round(origin_y - cy, 2),
            "width": int(p["w"]), "height": int(p["h"]), "rotation": 0}}

    skeleton = {
        "skeleton": {"spine": SPINE_VERSION, "x": round(minx - origin_x, 2), "y": 0.0,
                     "width": round(maxx - minx, 2), "height": round(maxy - miny, 2),
                     "images": "./images/", "audio": ""},
        "bones": [{"name": "root"}],
        "slots": slots,
        "skins": [{"name": "default", "attachments": attachments}],
        "animations": {},
    }
    path = os.path.join(args.out, args.character + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(skeleton, f, indent=2)
    log("wrote %s (%d parts, back -> front: %s)" % (path, len(parts),
                                                     " | ".join(p["name"] for p in parts)))


if __name__ == "__main__":
    main()
