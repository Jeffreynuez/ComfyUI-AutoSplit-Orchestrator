"""
Smoke test for the SAM 3 orchestrator node with a fake SAM 3 processor.
Needs torch (CPU is fine); skipped when torch is missing.

Checks the June bugs stay fixed: both prompts of a left/right pair returning
the SAME instance, auto left/right on a plural prompt, hair that SAM 3 returns
as one blob, the image encoded once, the schema-2 sidecar, and the
low-confidence retry (recovers a real part, refuses a repeat of a found one).
"""
import json
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NODE_DIR = os.path.join(os.path.dirname(HERE), "ComfyUI-AutoSplit-Orchestrator")
sys.path.insert(0, NODE_DIR)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

H, W = 300, 300


def rect(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


EYE_VL = rect(60, 72, 120, 136)    # viewer-left eye
EYE_VR = rect(60, 72, 164, 180)    # viewer-right eye
FACE = rect(30, 120, 110, 190)
HAIR = rect(10, 150, 90, 210) & ~rect(30, 120, 110, 190) | rect(15, 45, 110, 190)
TORSO = rect(130, 290, 100, 200)
SASH = rect(200, 220, 100, 200)
HAND_VL = rect(240, 270, 60, 90)
HAND_VR = rect(240, 270, 210, 240)
BELT = rect(180, 192, 100, 200)

PROMPTS = {
    # SAM 3 being unreliable about "left": both prompts give the viewer-left eye
    "left eye": [(EYE_VL, 0.9)], "right eye": [(EYE_VL, 0.8)],
    "eye": [(EYE_VL, 0.9), (EYE_VR, 0.88)],
    "face": [(FACE, 0.95)], "hair": [(HAIR, 0.93)], "torso": [(TORSO, 0.9)],
    "sash": [(SASH, 0.7)], "hands": [(HAND_VL, 0.9), (HAND_VR, 0.85)],
    # right answer, low score: recovered by the retry and flagged
    "belt": [(BELT, 0.25)],
    # low score that only points back at the torso: the retry must refuse it
    "cape": [(TORSO, 0.22)],
}


class FakeProcessor:
    def __init__(self):
        self.encodes = 0
        self.threshold = 0.5

    def set_image(self, pil, state=None):
        self.encodes += 1
        return {"backbone_out": {}}

    def reset_all_prompts(self, state):
        for k in ("masks", "scores"):
            state.pop(k, None)

    def set_confidence_threshold(self, t, state=None):
        self.threshold = t
        return state

    def set_text_prompt(self, prompt, state):
        found = [(m, s) for m, s in PROMPTS.get(prompt, []) if s >= self.threshold]
        if found:
            state["masks"] = torch.stack([torch.from_numpy(m)[None] for m, _ in found])
            state["scores"] = torch.tensor([s for _, s in found])
        return state


def _install_fakes(outdir):
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: outdir
    sys.modules["folder_paths"] = fp
    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")

    class PB:
        def __init__(self, n):
            pass

        def update(self, k):
            pass
    utils.ProgressBar = PB
    comfy.utils = utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = utils


def test_orchestrator_end_to_end():
    if torch is None:
        print("torch missing - skipped")
        return
    out = tempfile.mkdtemp()
    _install_fakes(out)
    import nodes_sam3
    node = nodes_sam3.AutoSplitOrchestratorSAM3()

    img = np.full((H, W, 3), 0.3, np.float32)
    img[FACE] = (0.86, 0.66, 0.55)
    img[HAIR] = (0.1, 0.6, 0.4)
    img[TORSO] = (0.82, 0.58, 0.47)
    img[SASH] = (0.2, 0.7, 0.3)
    img[BELT] = (0.35, 0.2, 0.1)
    img[EYE_VL | EYE_VR] = (0.98, 0.98, 0.98)
    img[HAND_VL | HAND_VR] = (0.8, 0.55, 0.45)
    depth = np.zeros((H, W), np.float32)
    depth[HAIR & (np.arange(H)[:, None] < 50)] = 1.0      # bangs closer than back hair
    depth[FACE | TORSO] = 0.6
    proc = FakeProcessor()
    model = {"processor": proc, "model": None, "device": "cpu", "dtype": torch.float32}

    result = node.run_pipeline(
        torch.from_numpy(img)[None], torch.from_numpy(np.repeat(depth[..., None], 3, 2))[None],
        model, "face\nhair\ntorso\nsash\nbelt\ncape\nleft eye\nright eye\nhands", "fake", "smoke",
        padding=4, confidence_threshold=0.3, occlusion_threshold=0.15, mask_expand_pixels=4,
        output_mode="alpha_mask", run_id="run1", depth_near="bright")
    report, _ = result["result"]
    meta_path = result["ui"]["autosplit"][0]["metadata"]
    meta = json.load(open(meta_path))
    tags = {p["tag"]: p for p in meta["parts"]}

    assert proc.encodes == 1, "image encoded %d times" % proc.encodes
    assert meta["schema_version"] == 2 and meta["run_id"] == "run1"
    assert os.path.dirname(meta_path).endswith(os.path.join("smoke", "run1"))
    # pair resolved from geometry: anatomical left eye is on the viewer's right
    le, re_ = tags["left eye"], tags["right eye"]
    assert le["xyxy"][0] > re_["xyxy"][0], (le["xyxy"], re_["xyxy"])
    assert le["lr_source"] == "geometry"
    # plural auto split
    assert "hand_left" in tags and "hand_right" in tags
    assert tags["hand_left"]["xyxy"][0] > tags["hand_right"]["xyxy"][0]
    # hair split by depth although SAM 3 returned one blob
    assert "hair_front" in tags and "hair_back" in tags, list(tags)
    # draw order: sash in front of torso, eyes in front of face
    assert tags["sash"]["z_order"] < tags["torso"]["z_order"]
    assert tags["left eye"]["z_order"] < tags["face"]["z_order"]
    # second chance: belt recovered and flagged, cape refused as a repeat of torso
    assert "belt" in tags and "low_confidence" in tags["belt"]["flags"], tags.get("belt")
    assert "cape" not in tags
    assert "retry_duplicate" in report, report
    assert "Missing: 1 (cape)" in report, report


if __name__ == "__main__":
    test_orchestrator_end_to_end()
    print("smoke test passed")
