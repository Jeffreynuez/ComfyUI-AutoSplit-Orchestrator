"""
SplitLeftRight
==============
Splits a single mask containing two (or more) disconnected blobs into
separate left and right masks using OpenCV connected components, then
sorts by centroid X. Pure post-process — no model.

Ported from See-Through WebUI (common/utils/inference_utils.py
`label_lr_split` / `part_lr_split`). Useful when SAM3 returns a unified
"hands" / "ears" / "eyes" mask and you want them as -l / -r parts without
re-prompting the segmenter.

Inputs
------
mask         : MASK   — input mask (single channel, may contain multiple blobs)
image        : IMAGE  — (optional) source image, for producing per-side cropped images
min_area     : INT    — ignore connected components smaller than this many pixels
mode         : choice — "two_largest"  : take 2 biggest blobs, sort by centroid x
                        "all_to_sides" : assign every blob to L or R based on
                                         its centroid relative to the overall midline
                        "eye_pair"     : grab top 2 (eyes) AND next 2 (brows),
                                         returning L/R split for both pairs

Outputs
-------
mask_left, mask_right            : MASK
image_left, image_right          : IMAGE  (zeroed where not in that side's mask)
count                            : INT    (how many blobs were considered)
"""

import torch
import numpy as np
import cv2


def _to_np_mask(mask):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return mask[0].detach().cpu().numpy()


def _to_torch_mask(np_mask, ref):
    t = torch.from_numpy(np_mask.astype(np.float32))
    if ref.dim() == 3:
        t = t.unsqueeze(0)
    return t.to(ref.device)


def _split_two(labels, stats, id_a, id_b):
    """Return (left_mask, right_mask) for two component IDs, sorted by centroid x."""
    mask_a = (labels == id_a).astype(np.float32)
    mask_b = (labels == id_b).astype(np.float32)
    cx_a = stats[id_a][0] + stats[id_a][2] / 2.0
    cx_b = stats[id_b][0] + stats[id_b][2] / 2.0
    # Note: in screen coords, smaller x is the viewer's left. See-Through
    # treats viewer-left as "-r" (character's right). We follow the more
    # intuitive convention: smaller x = "left" output port.
    if cx_a <= cx_b:
        return mask_a, mask_b
    return mask_b, mask_a


class SplitLeftRight:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "mode": (["two_largest", "all_to_sides", "eye_pair"], {"default": "two_largest"}),
                "lr_convention": (["anatomical", "viewer"], {
                    "default": "anatomical",
                    "tooltip": "anatomical: character's own left/right (rigger convention — character's left hand exits on the viewer's right). viewer: smaller x = left output."
                }),
                "min_area": ("INT", {"default": 50, "min": 0, "max": 1_000_000, "step": 10}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("mask_left", "mask_right", "extra_left", "extra_right",
                    "image_left", "image_right", "count")
    FUNCTION = "split"
    CATEGORY = "AutoSplit/mask"

    def split(self, mask, mode, lr_convention, min_area, image=None):
        m_np = _to_np_mask(mask)
        H, W = m_np.shape

        binary = (m_np > 1e-3).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # Filter background (id 0) and tiny components
        valid_ids = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        # Sort by area, descending
        valid_ids.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])

        empty = np.zeros((H, W), dtype=np.float32)
        left_mask = empty.copy()
        right_mask = empty.copy()
        extra_left = empty.copy()
        extra_right = empty.copy()
        count = len(valid_ids)

        if mode == "two_largest":
            if len(valid_ids) >= 2:
                left_mask, right_mask = _split_two(labels, stats, valid_ids[0], valid_ids[1])
            elif len(valid_ids) == 1:
                # Only one blob — pass through to left
                left_mask = (labels == valid_ids[0]).astype(np.float32)

        elif mode == "all_to_sides":
            if len(valid_ids) >= 1:
                # Use overall mask centroid as midline
                ys, xs = np.where(binary > 0)
                midline = float(xs.mean()) if len(xs) else W / 2.0
                for i in valid_ids:
                    cx = stats[i][0] + stats[i][2] / 2.0
                    blob = (labels == i).astype(np.float32)
                    if cx <= midline:
                        left_mask = np.maximum(left_mask, blob)
                    else:
                        right_mask = np.maximum(right_mask, blob)

        elif mode == "eye_pair":
            # Top 2 areas → eyes (left, right). Next 2 → brows (left, right).
            if len(valid_ids) >= 2:
                left_mask, right_mask = _split_two(labels, stats, valid_ids[0], valid_ids[1])
            if len(valid_ids) >= 4:
                extra_left, extra_right = _split_two(labels, stats, valid_ids[2], valid_ids[3])

        # Apply L/R convention. Default internal logic is viewer-relative
        # (smaller x = left). Anatomical = character's own L/R, which is
        # the mirror: the character's left hand appears on the viewer's right.
        if lr_convention == "anatomical":
            left_mask, right_mask = right_mask, left_mask
            extra_left, extra_right = extra_right, extra_left

        # Build per-side images if an image was provided
        if image is not None:
            img = image  # [B,H,W,3]
            if img.dim() == 3:
                img = img.unsqueeze(0)
            B = img.shape[0]
            left_t = _to_torch_mask(left_mask, mask).unsqueeze(-1)   # [1,H,W,1]
            right_t = _to_torch_mask(right_mask, mask).unsqueeze(-1)
            if left_t.dim() == 3:
                left_t = left_t.unsqueeze(0)
            if right_t.dim() == 3:
                right_t = right_t.unsqueeze(0)
            image_left = (img * left_t).clamp(0, 1)
            image_right = (img * right_t).clamp(0, 1)
        else:
            image_left = torch.zeros((1, H, W, 3), dtype=torch.float32, device=mask.device)
            image_right = torch.zeros((1, H, W, 3), dtype=torch.float32, device=mask.device)

        return (
            _to_torch_mask(left_mask, mask),
            _to_torch_mask(right_mask, mask),
            _to_torch_mask(extra_left, mask),
            _to_torch_mask(extra_right, mask),
            image_left,
            image_right,
            count,
        )


NODE_CLASS_MAPPINGS = {
    "SplitLeftRight": SplitLeftRight,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SplitLeftRight": "Split Left/Right (AutoSplit)",
}
