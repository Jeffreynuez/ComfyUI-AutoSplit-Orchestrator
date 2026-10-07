"""
Shared loaders for AutoSplit output and ground truth (numpy + Pillow only).

Used by evaluate.py and wiggle.py. Understands both output layouts:

  * schema 1 (June 2026): fixed folders such as output/split_parts_sam3 and
    output/split_parts_sam3_facial, with the head-crop transform in a global
    output/head_crop_transform.json;
  * schema 2 (Phase 1): one folder per run, each parts_metadata.json carrying
    its own crop_transform, xyxy_full and a global draw_index.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

_NODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "ComfyUI-AutoSplit-Orchestrator")
if _NODE_DIR not in sys.path:
    sys.path.insert(0, _NODE_DIR)
from autosplit_core import drawing  # noqa: E402


# --------------------------------------------------------------------------
# predicted parts
# --------------------------------------------------------------------------
class Part:
    """One predicted part placed on the full canvas."""

    def __init__(self, tag, rgba, x0, y0, z_order, facial, meta):
        self.tag = tag
        self.rgba = rgba            # HxWx4 uint8 crop
        self.x0, self.y0 = int(x0), int(y0)
        self.z_order = z_order      # 0 = front within its own pass
        self.facial = facial
        self.meta = meta
        self.draw_index = meta.get("draw_index")  # global, 0 = back-most

    @property
    def alpha(self):
        return self.rgba[..., 3]

    def full_mask(self, W, H, thresh=128):
        m = np.zeros((H, W), bool)
        a = self.alpha >= thresh
        h, w = a.shape
        xa, ya = max(0, self.x0), max(0, self.y0)
        xb, yb = min(W, self.x0 + w), min(H, self.y0 + h)
        if xb <= xa or yb <= ya:
            return m
        m[ya:yb, xa:xb] = a[ya - self.y0:yb - self.y0, xa - self.x0:xb - self.x0]
        return m

    def paint_onto(self, canvas_rgb, canvas_mask, thresh=128):
        """Write this part's colour into a full-canvas HxWx3 array where its
        alpha >= thresh (later calls paint over earlier ones)."""
        H, W = canvas_mask.shape
        a = self.alpha >= thresh
        h, w = a.shape
        xa, ya = max(0, self.x0), max(0, self.y0)
        xb, yb = min(W, self.x0 + w), min(H, self.y0 + h)
        if xb <= xa or yb <= ya:
            return
        sub = a[ya - self.y0:yb - self.y0, xa - self.x0:xb - self.x0]
        rgb = self.rgba[ya - self.y0:yb - self.y0, xa - self.x0:xb - self.x0, :3]
        canvas_rgb[ya:yb, xa:xb][sub] = rgb[sub]
        canvas_mask[ya:yb, xa:xb] |= sub


def _read_transform(meta, folder):
    t = meta.get("crop_transform")
    if t:
        return t
    # schema 1: global file next to the pass folders
    for cand in (os.path.join(folder, "head_crop_transform.json"),
                 os.path.join(os.path.dirname(folder.rstrip("/\\")), "head_crop_transform.json")):
        if os.path.exists(cand):
            with open(cand, encoding="utf-8") as f:
                return json.load(f)
    return None


def load_pass(folder, facial=None):
    """Load one orchestrator pass (a folder holding parts_metadata.json)."""
    path = os.path.join(folder, "parts_metadata.json")
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    entries = meta if isinstance(meta, list) else meta.get("parts", [])
    if facial is None:
        base = os.path.basename(folder.rstrip("/\\")).lower()
        facial = (bool(meta.get("crop_transform")) or meta.get("pass") == "facial"
                  or os.path.exists(os.path.join(folder, "head_crop_transform.json"))
                  or "facial" in base or base.endswith("face"))
    t = _read_transform(meta, folder) if facial else None
    sc = float((t or {}).get("scale", 1.0) or 1.0)
    ox, oy = (t or {}).get("x_min", 0), (t or {}).get("y_min", 0)
    parts = []
    for e in entries:
        fp = os.path.join(folder, e.get("file", ""))
        if not os.path.exists(fp):
            continue
        im = Image.open(fp).convert("RGBA")
        if e.get("xyxy_full"):
            x0, y0, x1, y1 = e["xyxy_full"]
            if im.size != (x1 - x0, y1 - y0) and x1 > x0 and y1 > y0:
                im = im.resize((x1 - x0, y1 - y0), Image.LANCZOS)
        else:
            x0, y0 = e.get("xyxy", [0, 0])[:2]
            if facial and sc != 1.0:
                im = im.resize((max(1, round(im.width / sc)), max(1, round(im.height / sc))),
                               Image.LANCZOS)
                x0, y0 = ox + x0 / sc, oy + y0 / sc
        z = e.get("z_order", e.get("z_rank", 0))
        e = dict(e)
        e["_schema"] = meta.get("schema_version", 1) if isinstance(meta, dict) else 1
        parts.append(Part(e.get("tag", "part"), np.asarray(im).copy(), round(x0), round(y0),
                          z, facial, e))
    return parts


def back_to_front(parts):
    """Global draw order, back-most first.

    schema 2 runs: each pass is ordered by its own draw_index / z_order and the
    facial pass is inserted directly in front of the face (drawing.merge).
    schema 1 runs: reproduces the June exporter, which drew every facial part
    on top of every body part."""
    body = [p for p in parts if not p.facial]
    face = [p for p in parts if p.facial]

    def pass_btf(ps):
        if ps and all(p.draw_index is not None for p in ps):
            return sorted(ps, key=lambda p: p.draw_index)
        return sorted(ps, key=lambda p: -float(p.z_order))

    body_btf, face_btf = pass_btf(body), pass_btf(face)
    if any(p.meta.get("_schema", 1) >= 2 for p in parts):
        by_id = {id(p.meta): p for p in parts}
        merged = drawing.merge([p.meta for p in body_btf], [p.meta for p in face_btf])
        return [by_id[id(e)] for e in merged]
    return body_btf + face_btf


def composite(parts_btf, W, H, background=None):
    canvas = Image.new("RGBA", (W, H), background or (0, 0, 0, 0))
    for p in parts_btf:
        _paste_clipped(canvas, p)
    return canvas


def _paste_clipped(canvas, p):
    im = Image.fromarray(p.rgba)
    x0, y0 = p.x0, p.y0
    cx, cy = max(0, -x0), max(0, -y0)
    im = im.crop((cx, cy, im.width, im.height))
    canvas.alpha_composite(im, (max(0, x0), max(0, y0)))


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------
class GroundTruth:
    def __init__(self, folder):
        self.folder = folder
        with open(os.path.join(folder, "gt.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.W, self.H = self.meta["canvas"]
        self.layers = self.meta["layers"]
        self._alpha = {}

    def layer_mask(self, layer, thresh=128):
        key = (layer["index"], thresh)
        if key not in self._alpha:
            im = Image.open(os.path.join(self.folder, layer["file"])).convert("RGBA")
            a = np.asarray(im)[..., 3] >= thresh
            m = np.zeros((self.H, self.W), bool)
            x0, y0 = layer["bbox"][:2]
            h, w = a.shape
            m[y0:y0 + h, x0:x0 + w] = a[: self.H - y0, : self.W - x0]
            self._alpha[key] = m
        return self._alpha[key]

    def paint_layer(self, layer, canvas_rgb, thresh=128):
        """Write a layer's own colour (hidden areas included) into a
        full-canvas HxWx3 array where its alpha >= thresh."""
        im = np.asarray(Image.open(os.path.join(self.folder, layer["file"])).convert("RGBA"))
        x0, y0 = layer["bbox"][:2]
        h, w = im.shape[:2]
        h, w = min(h, self.H - y0), min(w, self.W - x0)
        sub = im[:h, :w, 3] >= thresh
        canvas_rgb[y0:y0 + h, x0:x0 + w][sub] = im[:h, :w, :3][sub]

    def top_index_map(self):
        """Per pixel: draw index of the front-most solid (non-overlay) layer,
        -1 where nothing is drawn."""
        if not hasattr(self, "_top"):
            top = np.full((self.H, self.W), -1, np.int32)
            for layer in self.layers:  # back to front
                if layer.get("is_overlay"):
                    continue
                top[self.layer_mask(layer)] = layer["index"]
            self._top = top
        return self._top

    def silhouette(self):
        return self.top_index_map() >= 0

    def composite(self):
        return np.asarray(Image.open(os.path.join(self.folder, "composite.png")).convert("RGBA"))
