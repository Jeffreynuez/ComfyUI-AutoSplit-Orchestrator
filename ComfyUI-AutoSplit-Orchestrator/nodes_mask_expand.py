"""
AsymmetricMaskExpand
====================
ComfyUI node that expands a mask's bounding box asymmetrically (different
amounts on top/bottom/left/right) so the inpaint context window captures
things that hang off in one direction — e.g. hats, horns, twintails above
the head.

Inspired by See-Through WebUI's head crop padding (pad_x=0.30, pad_y_up=0.60,
pad_y_down=0.30) which fixes hats/horns getting clipped from the head context.

Drop-in: feed your part mask into this node, then feed the output mask into
InpaintCropImproved (or whatever crop node you use). The expanded region is
relative to the *bounding box* of the input mask, not the whole image, so it
scales with the part size.

Inputs
------
mask           : MASK   — the manually-painted (or generated) inpaint mask
pad_left       : FLOAT  — fraction of bbox width to pad on the left
pad_right      : FLOAT  — fraction of bbox width to pad on the right
pad_up         : FLOAT  — fraction of bbox height to pad upward
pad_down       : FLOAT  — fraction of bbox height to pad downward
preset         : choice — overrides the four pad_* values when not "custom"
fill_value     : FLOAT  — what to fill the expanded region with (0..1).
                          1.0 = mask everything in the expanded box.
                          0.0 = leave expanded region empty (just enlarges
                                the bbox by adding a single pixel at the
                                corners so downstream crop nodes pick it up).

Presets
-------
custom    : use the four pad_* sliders
head      : 0.30 / 0.30 / 0.60 / 0.30  (See-Through head crop)
torso     : 0.20 / 0.20 / 0.15 / 0.20
arm       : 0.25 / 0.25 / 0.20 / 0.25
leg       : 0.20 / 0.20 / 0.15 / 0.30
hair      : 0.35 / 0.35 / 0.70 / 0.20
symmetric : 0.25 / 0.25 / 0.25 / 0.25
"""

import torch
import numpy as np


PRESETS = {
    "custom":    None,
    "head":      (0.30, 0.30, 0.60, 0.30),
    "torso":     (0.20, 0.20, 0.15, 0.20),
    "arm":       (0.25, 0.25, 0.20, 0.25),
    "leg":       (0.20, 0.20, 0.15, 0.30),
    "hair":      (0.35, 0.35, 0.70, 0.20),
    "symmetric": (0.25, 0.25, 0.25, 0.25),
}


class AsymmetricMaskExpand:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "preset": (list(PRESETS.keys()), {"default": "head"}),
                "pad_left":  ("FLOAT", {"default": 0.30, "min": 0.0, "max": 4.0, "step": 0.05}),
                "pad_right": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 4.0, "step": 0.05}),
                "pad_up":    ("FLOAT", {"default": 0.60, "min": 0.0, "max": 4.0, "step": 0.05}),
                "pad_down":  ("FLOAT", {"default": 0.30, "min": 0.0, "max": 4.0, "step": 0.05}),
                "fill_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                         "tooltip": "0 = just expand the bbox (corner pixels). 1 = fill the whole expanded box."}),
            },
        }

    RETURN_TYPES = ("MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("mask", "x1", "y1", "x2", "y2")
    FUNCTION = "expand"
    CATEGORY = "AutoSplit/mask"

    def expand(self, mask, preset, pad_left, pad_right, pad_up, pad_down, fill_value):
        # Resolve preset
        if preset != "custom" and PRESETS.get(preset) is not None:
            pad_left, pad_right, pad_up, pad_down = PRESETS[preset]

        # mask: torch.Tensor [B,H,W] or [H,W]
        if mask.dim() == 2:
            mask_t = mask.unsqueeze(0)
        else:
            mask_t = mask

        out_masks = []
        x1f = y1f = x2f = y2f = 0

        for b in range(mask_t.shape[0]):
            m = mask_t[b].detach().cpu().numpy()
            H, W = m.shape

            # Find existing bbox
            ys, xs = np.where(m > 1e-3)
            if len(xs) == 0:
                # Empty mask — passthrough
                out_masks.append(torch.from_numpy(m))
                continue

            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            # Compute expanded bbox, clamped to image
            ex1 = max(0,     int(round(x1 - pad_left  * bw)))
            ex2 = min(W - 1, int(round(x2 + pad_right * bw)))
            ey1 = max(0,     int(round(y1 - pad_up    * bh)))
            ey2 = min(H - 1, int(round(y2 + pad_down  * bh)))

            new_m = m.copy()
            if fill_value > 0:
                new_m[ey1:ey2 + 1, ex1:ex2 + 1] = np.maximum(
                    new_m[ey1:ey2 + 1, ex1:ex2 + 1], fill_value
                )
            else:
                # Just place 4 corner pixels so crop nodes pick up the bbox
                new_m[ey1, ex1] = max(new_m[ey1, ex1], 1.0)
                new_m[ey1, ex2] = max(new_m[ey1, ex2], 1.0)
                new_m[ey2, ex1] = max(new_m[ey2, ex1], 1.0)
                new_m[ey2, ex2] = max(new_m[ey2, ex2], 1.0)

            out_masks.append(torch.from_numpy(new_m))
            x1f, y1f, x2f, y2f = ex1, ey1, ex2, ey2

        out = torch.stack(out_masks, dim=0).to(mask.device).float()
        return (out, x1f, y1f, x2f, y2f)


NODE_CLASS_MAPPINGS = {
    "AsymmetricMaskExpand": AsymmetricMaskExpand,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AsymmetricMaskExpand": "Asymmetric Mask Expand (AutoSplit)",
}
