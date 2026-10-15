"""
Shared loaders for AutoSplit output and ground truth (numpy + Pillow only).

Used by evaluate.py and wiggle.py. The run-folder loaders (Part, load_pass,
back_to_front, composite) live in autosplit_core/parts.py and are re-exported
here; this file adds the hand-cut ground truth.
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


# predicted parts live in the node package (the fill node uses them too)
from autosplit_core.parts import (Part, _paste_clipped, _read_transform, back_to_front,  # noqa: E402,F401
                                  composite, load_pass)


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
