"""
ComfyUI-AutoSplit-Orchestrator
==============================
Custom node package for 2D character auto-splitting.
Includes both SAM2 and SAM3 orchestrator variants.
"""

from .nodes import NODE_CLASS_MAPPINGS as SAM2_MAPPINGS
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as SAM2_DISPLAY

# Try to load SAM3 node — it's optional (requires comfyui-easy-sam3)
try:
    from .nodes_sam3 import NODE_CLASS_MAPPINGS as SAM3_MAPPINGS
    from .nodes_sam3 import NODE_DISPLAY_NAME_MAPPINGS as SAM3_DISPLAY
except Exception as e:
    print("[AutoSplit] SAM3 node not loaded (comfyui-easy-sam3 may not be installed): %s" % e)
    SAM3_MAPPINGS = {}
    SAM3_DISPLAY = {}

from .nodes_mask_expand import NODE_CLASS_MAPPINGS as MEXP_MAPPINGS
from .nodes_mask_expand import NODE_DISPLAY_NAME_MAPPINGS as MEXP_DISPLAY
from .nodes_split_lr import NODE_CLASS_MAPPINGS as LR_MAPPINGS
from .nodes_split_lr import NODE_DISPLAY_NAME_MAPPINGS as LR_DISPLAY

# Head crop + upscale for facial detail two-pass (requires SAM3)
try:
    from .nodes_head_crop import NODE_CLASS_MAPPINGS as HCROP_MAPPINGS
    from .nodes_head_crop import NODE_DISPLAY_NAME_MAPPINGS as HCROP_DISPLAY
except Exception as e:
    print("[AutoSplit] HeadCropUpscale node not loaded: %s" % e)
    HCROP_MAPPINGS = {}
    HCROP_DISPLAY = {}

NODE_CLASS_MAPPINGS = {**SAM2_MAPPINGS, **SAM3_MAPPINGS, **MEXP_MAPPINGS, **LR_MAPPINGS, **HCROP_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**SAM2_DISPLAY, **SAM3_DISPLAY, **MEXP_DISPLAY, **LR_DISPLAY, **HCROP_DISPLAY}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
