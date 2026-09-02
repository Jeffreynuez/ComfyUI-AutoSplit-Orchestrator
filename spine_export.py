"""
2D AutoSplit Studio - Phase 2: Spine export (standalone).

Reads the SAM3 split output (part PNGs + metadata + head-crop transform) and
writes a Spine 4.3.22 project:
    <OUT_DIR>/images/<part>.png      one cut-out per part (facial parts downscaled)
    <OUT_DIR>/<CHARACTER>.json       skeleton: root bone, a slot + region
                                     attachment per part, default skin, y-up coords

Import into Spine via 'Import Data...', pick the JSON, and point the images path
at the images/ folder. The character appears assembled and ready to rig.

Coordinate notes:
  - Image space is y-DOWN, origin top-left. Spine is y-UP.
  - Origin is the character bounding-box bottom-centre (feet near y=0).
  - Facial parts come from the 2x upscaled head crop, so they're downscaled by
    1/scale and offset by the crop origin to land on the full image.
  - Draw order (Spine slot order, first=back): body back-to-front, then facial
    features back-to-front on top of the face.
"""
import json
import os
import shutil

try:
    from PIL import Image
except ImportError:
    Image = None

# ------------------------------ CONFIG ------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMFY_ROOT = os.environ.get(
    "AUTOSPLIT_COMFY_ROOT",
    os.path.join(os.path.dirname(SCRIPT_DIR), "ComfyUI-Easy-Install", "ComfyUI"))
COMFY_OUTPUT_DIR = os.environ.get(
    "AUTOSPLIT_COMFY_OUTPUT", os.path.join(COMFY_ROOT, "output"))
BODY_SUBDIR = "split_parts_sam3"
FACIAL_SUBDIR = "split_parts_sam3_facial"
OUT_DIR = os.environ.get("AUTOSPLIT_OUT", os.path.join(SCRIPT_DIR, "spine_export"))
CHARACTER = os.environ.get("AUTOSPLIT_CHARACTER", "character")
SPINE_VERSION = "4.3.22"
# --------------------------------------------------------------------


def log(m):
    print("[spine] " + str(m), flush=True)


def sanitize(tag):
    return str(tag).strip().lower().replace(" ", "_").replace("/", "_")


def load_transform():
    p = os.path.join(COMFY_OUTPUT_DIR, "head_crop_transform.json")
    try:
        t = json.load(open(p, encoding="utf-8"))
        return (int(t.get("x_min", 0)), int(t.get("y_min", 0)),
                float(t.get("scale", 1.0)) or 1.0)
    except Exception:
        return 0, 0, 1.0


def collect():
    """Return (parts, scale). Each part: tag, src, facial, x, y, w, h (full-image
    space, top-left origin), z_order, order_index (global back-to-front)."""
    ox, oy, sc = load_transform()
    body, facial = [], []
    for sub, is_facial in ((BODY_SUBDIR, False), (FACIAL_SUBDIR, True)):
        d = os.path.join(COMFY_OUTPUT_DIR, sub)
        meta = os.path.join(d, "parts_metadata.json")
        if not os.path.exists(meta):
            continue
        data = json.load(open(meta, encoding="utf-8"))
        entries = data if isinstance(data, list) else data.get("parts", [])
        for e in entries:
            fp = os.path.join(d, e.get("file", ""))
            if not os.path.exists(fp):
                continue
            xyxy = e.get("xyxy") or [0, 0, 0, 0]
            with Image.open(fp) as im:
                pw, ph = im.size
            if is_facial and sc != 1.0:
                w, h = pw / sc, ph / sc
                x, y = ox + xyxy[0] / sc, oy + xyxy[1] / sc
            elif is_facial:
                w, h, x, y = pw, ph, ox + xyxy[0], oy + xyxy[1]
            else:
                w, h, x, y = pw, ph, xyxy[0], xyxy[1]
            rec = {
                "tag": sanitize(e.get("tag", "part")),
                "src": fp, "facial": is_facial,
                "x": float(x), "y": float(y), "w": float(w), "h": float(h),
                "z_order": e.get("z_order", e.get("z_rank", 0)),
            }
            (facial if is_facial else body).append(rec)
    # draw order, back-to-front: body (high z_order first) then facial on top
    body.sort(key=lambda p: -p["z_order"])
    facial.sort(key=lambda p: -p["z_order"])
    parts = body + facial
    for i, p in enumerate(parts):
        p["order_index"] = i
    return parts, sc


def main():
    if Image is None:
        log("Pillow required:  pip install pillow")
        return
    parts, sc = collect()
    if not parts:
        log("No parts found - run a split first.")
        return

    # origin = character bounding-box bottom-centre
    minx = min(p["x"] for p in parts)
    maxx = max(p["x"] + p["w"] for p in parts)
    miny = min(p["y"] for p in parts)
    maxy = max(p["y"] + p["h"] for p in parts)
    origin_x = (minx + maxx) / 2.0
    origin_y = maxy

    img_dir = os.path.join(OUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)

    slots, attachments = [], {}
    for p in parts:
        name = p["tag"]
        out_png = os.path.join(img_dir, name + ".png")
        if p["facial"] and sc != 1.0:
            with Image.open(p["src"]) as im:
                im = im.convert("RGBA").resize(
                    (max(1, int(round(p["w"]))), max(1, int(round(p["h"])))),
                    Image.LANCZOS)
                im.save(out_png)
        else:
            shutil.copyfile(p["src"], out_png)

        cx = p["x"] + p["w"] / 2.0
        cy = p["y"] + p["h"] / 2.0
        sx = round(cx - origin_x, 2)
        sy = round(origin_y - cy, 2)   # flip Y (image y-down -> Spine y-up)
        slots.append({"name": name, "bone": "root", "attachment": name})
        attachments[name] = {
            name: {"x": sx, "y": sy,
                   "width": int(round(p["w"])), "height": int(round(p["h"])),
                   "rotation": 0}
        }

    skeleton = {
        "skeleton": {
            "spine": SPINE_VERSION,
            "x": round(minx - origin_x, 2), "y": 0.0,
            "width": round(maxx - minx, 2), "height": round(maxy - miny, 2),
            "images": "./images/", "audio": "",
        },
        "bones": [{"name": "root"}],
        "slots": slots,
        "skins": [{"name": "default", "attachments": attachments}],
        "animations": {},
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, CHARACTER + ".json")
    json.dump(skeleton, open(json_path, "w", encoding="utf-8"), indent=2)

    log("wrote %s" % json_path)
    log("images: %d -> %s" % (len(parts), img_dir))
    log("origin (bottom-centre): (%.0f, %.0f)  character %dx%d"
        % (origin_x, origin_y, round(maxx - minx), round(maxy - miny)))
    log("draw order (back->front): %s" % ", ".join(p["tag"] for p in parts))


if __name__ == "__main__":
    main()
