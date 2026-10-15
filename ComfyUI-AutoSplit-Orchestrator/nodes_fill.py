"""
Auto-Split Fill (Klein peels) -- the hidden-area fill as one ComfyUI node.

Runs after the SAM 3 orchestrator(s). From the body pass (and the facial pass,
when connected) it plans the "peels" with no per-character prompts
(autosplit_core/peel.py), makes each one with Flux.2 Klein inside this node
(a full-picture edit that takes the front layers away in the same pose),
splits every peel with the same SAM 3 orchestrator, and merges the peeled
shapes plus the rigging-convention rules (underlaps, back panels, deep-shadow
backs, skin under the face; autosplit_core/rigfill.py) into a new pass:

    <body pass>_filled/   parts_metadata.json + one PNG per part (and new
                          back parts such as "jacket back")
    <body pass>_fill/     the edited pictures and their splits (for checking)

Wire `report` from the body orchestrator into `body_report` (and the facial
orchestrator's into `face_report`) so this node runs after them and finds
their folders. It belongs in a bypassable group above the split.

Models (Jeffrey's install): Flux.2 Klein 4B fp8 + Qwen 3 4B, and 9B kv-fp8 +
Qwen 3 8B for jobs that remove whole arms or legs (4B ignores those), with
the Flux.2 VAE. Set the 9B model to "none" to run everything on 4B.
"""
import json
import math
import os
import re
import time

import numpy as np

try:
    from .autosplit_core import peel, rigfill
    from .autosplit_core.fillmerge import merge
    from .autosplit_core.parts import back_to_front, exploded, load_pass
except ImportError:  # pragma: no cover - loaded outside the package (tests)
    from autosplit_core import peel, rigfill
    from autosplit_core.fillmerge import merge
    from autosplit_core.parts import back_to_front, exploded, load_pass


def _pick(options, *keys):
    """The first option whose lower-case name holds every key."""
    for o in options:
        if all(k in o.lower() for k in keys):
            return o
    return None


def run_dir_from_report(report):
    """The orchestrator report ends with '  Output: <folder>'."""
    if not report:
        return None
    m = re.findall(r"^\s*Output:\s*(.+?)\s*$", report, flags=re.M)
    return m[-1] if m else None


def _has_alpha_parts(folder):
    """True when the pass's part PNGs carry their shape in the alpha channel
    (output_mode alpha_mask), False when they are opaque rectangles
    (solid_background), None when the pass saved no parts."""
    from PIL import Image
    with open(os.path.join(folder, "parts_metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    entries = meta if isinstance(meta, list) else meta.get("parts", [])
    seen = False
    for e in entries[:6]:
        fp = os.path.join(folder, e.get("file", ""))
        if os.path.isfile(fp):
            seen = True
            im = Image.open(fp)
            if im.mode in ("RGBA", "LA") and np.asarray(im.convert("RGBA"))[..., 3].min() < 128:
                return True
    return False if seen else None


def opaque_passes(runs):
    """Names of the passes (runs[0] body, runs[1] facial) whose parts were
    saved without transparency, so the fill cannot read their shapes."""
    names = ("body", "facial")
    return [names[min(i, 1)] for i, d in enumerate(runs) if _has_alpha_parts(d) is False]


def _contact_sheet(view, thumbs, silhouette, height=1024, pad=0.08):
    """The exploded view and the edits side by side, cropped to the
    character (padded for the parts pushed outward), `height` tall."""
    from PIL import Image
    H, W = silhouette.shape
    ys, xs = np.nonzero(silhouette)
    if len(xs):
        px, py = int(pad * W), int(pad * H)
        box = (max(0, xs.min() - px), max(0, ys.min() - py), min(W, xs.max() + px), min(H, ys.max() + py))
    else:
        box = (0, 0, W, H)
    tiles = [view.crop(box)]
    for t in thumbs:
        im = Image.fromarray(t)
        sx, sy = im.width / float(W), im.height / float(H)
        tiles.append(im.crop((int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy))))
    tiles = [t.resize((max(1, round(t.width * height / t.height)), height), Image.LANCZOS) for t in tiles]
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 8 * (len(tiles) - 1), height), (40, 40, 40))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 8
    return np.asarray(sheet)


def _to_tensor(rgb):
    import torch
    return torch.from_numpy(rgb.astype(np.float32) / 255.0)[None]


def _to_rgb(t):
    return (t[0, :, :, :3].clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)


class KleinEditor:
    """Flux.2 Klein full-picture edits (what tools/klein_workflows.py's
    edit_full graph does: ReferenceLatent on the scaled picture, cfg 1,
    euler/simple), with the 4B / 9B models loaded only when a job needs them."""

    def __init__(self, models, vae_name):
        self.models = models          # {"4b": (unet, text encoder), "9b": ...}
        self.vae_name = vae_name
        self.loaded = {}
        self.vae = None

    def _load(self, size):
        import nodes
        if size not in self.loaded:
            unet_name, te_name = self.models[size]
            model = nodes.UNETLoader().load_unet(unet_name, "default")[0]
            clip = nodes.CLIPLoader().load_clip(te_name, type="flux2", device="default")[0]
            self.loaded = {size: (model, clip)}   # one model set at a time
        if self.vae is None:
            self.vae = nodes.VAELoader().load_vae(self.vae_name)[0]
        return self.loaded[size]

    def edit(self, image, prompt, size="4b", seed=7, steps=4, megapixels=1.5):
        """image: IMAGE tensor [1, H, W, C]. Returns (edit at working size,
        edit scaled back to the picture's size)."""
        import comfy.model_management
        import comfy.utils
        import node_helpers
        import nodes
        import torch

        model, clip = self._load(size)
        H, W = image.shape[1], image.shape[2]
        s = math.sqrt(megapixels * 1024 * 1024 / (W * H))
        w, h = max(16, round(W * s / 16) * 16), max(16, round(H * s / 16) * 16)
        small = comfy.utils.common_upscale(image.movedim(-1, 1), w, h, "lanczos", "disabled").movedim(1, -1)
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        latent = self.vae.encode(small[:, :, :, :3])
        pos = node_helpers.conditioning_set_values(cond, {"reference_latents": [latent]}, append=True)
        neg = nodes.ConditioningZeroOut().zero_out(cond)[0]
        neg = node_helpers.conditioning_set_values(neg, {"reference_latents": [latent]}, append=True)
        empty = torch.zeros([1, 128, h // 16, w // 16], device=comfy.model_management.intermediate_device())
        out = nodes.common_ksampler(model, seed, steps, 1.0, "euler", "simple", pos, neg,
                                    {"samples": empty}, denoise=1.0)[0]
        img = self.vae.decode(out["samples"])
        if len(img.shape) == 5:
            img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
        up = comfy.utils.common_upscale(img.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)
        return img, up


class AutoSplitFillKlein:
    """Hidden-area fill for an Auto-Split run: Klein peels + rigging rules."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        dm = folder_paths.get_filename_list("diffusion_models")
        te = folder_paths.get_filename_list("text_encoders")
        va = folder_paths.get_filename_list("vae")
        d4 = _pick(dm, "klein", "4b") or (dm[0] if dm else "")
        d9 = _pick(dm, "klein", "9b", "fp8") or _pick(dm, "klein", "9b") or "none"
        t4 = _pick(te, "qwen_3_4b") or _pick(te, "qwen", "4b") or (te[0] if te else "")
        t8 = _pick(te, "qwen_3_8b") or _pick(te, "qwen", "8b") or "none"
        vf = _pick(va, "flux2") or (va[0] if va else "")
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The same picture the orchestrator split."}),
                "depth_map": ("IMAGE", {"tooltip": "The same depth map the body orchestrator used."}),
                "sam3_model": ("EASY_SAM3_MODEL",),
                "body_report": ("STRING", {"forceInput": True,
                                           "tooltip": "'report' from the body orchestrator."}),
                "klein_4b": (dm, {"default": d4}),
                "klein_4b_text_encoder": (te, {"default": t4}),
                "klein_9b": (["none"] + dm, {"default": d9,
                                             "tooltip": "Used for jobs that remove whole arms or legs. "
                                                        "'none' runs everything on 4B."}),
                "klein_9b_text_encoder": (["none"] + te, {"default": t8}),
                "klein_vae": (va, {"default": vf}),
                "base_layer": ("STRING", {"default": peel.BASE_LAYER,
                                          "tooltip": "What the base body wears (the rig's convention)."}),
                "edit_seed": ("INT", {"default": 7, "min": 0, "max": 2 ** 31 - 1,
                                      "tooltip": "Seed for every edit (kept fixed so a run repeats)."}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "megapixels": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 4.0, "step": 0.1,
                                         "tooltip": "Working size of the edits; they are scaled back."}),
            },
            "optional": {
                "face_report": ("STRING", {"forceInput": True,
                                           "tooltip": "'report' from the facial orchestrator."}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("report", "preview")
    FUNCTION = "run"
    CATEGORY = "2D Character Split"
    OUTPUT_NODE = True

    def run(self, image, depth_map, sam3_model, body_report, klein_4b, klein_4b_text_encoder,
            klein_9b, klein_9b_text_encoder, klein_vae, base_layer, edit_seed, steps, megapixels,
            face_report=None):
        import folder_paths
        from comfy.utils import ProgressBar
        from PIL import Image

        try:
            from .nodes_sam3 import AutoSplitOrchestratorSAM3
        except ImportError:  # pragma: no cover
            from nodes_sam3 import AutoSplitOrchestratorSAM3

        t0 = time.time()
        lines = []

        def log(msg):
            print("[AutoSplit-Fill] " + msg)
            lines.append(msg)

        body_dir = run_dir_from_report(body_report)
        if not body_dir or not os.path.isfile(os.path.join(body_dir, "parts_metadata.json")):
            raise RuntimeError("Auto-Split Fill: no body pass found in body_report (connect the body "
                               "orchestrator's 'report' output). Got: %r" % body_dir)
        face_dir = run_dir_from_report(face_report)
        runs = [body_dir] + ([face_dir] if face_dir and os.path.isfile(
            os.path.join(face_dir, "parts_metadata.json")) else [])
        with open(os.path.join(body_dir, "parts_metadata.json"), encoding="utf-8") as f:
            body_meta = json.load(f)

        opaque = opaque_passes(runs)
        if opaque:
            raise RuntimeError(
                "Auto-Split Fill needs parts cut out with transparency: set output_mode to "
                "'alpha_mask' on the %s orchestrator%s (this run saved %s on a solid background) "
                "and run again." % (" and ".join(opaque), "s" if len(opaque) > 1 else "",
                                    "both passes" if len(opaque) > 1 else "its parts"))
        src = _to_rgb(image)
        H, W = src.shape[:2]
        jobs = peel.plan_runs(runs, src, base_layer=base_layer or peel.BASE_LAYER)
        edits = [j for j in jobs if not peel.is_rule(j)]
        rules = [j for j in jobs if peel.is_rule(j)]
        have_9b = klein_9b != "none" and klein_9b_text_encoder != "none"
        for j in edits:
            if j.get("model") == "9b" and not have_9b:
                j["model"] = "4b"
        edits.sort(key=lambda j: j.get("model") == "9b")   # one model swap at most
        log("body pass: %s" % body_dir)
        log("planned %d edits + %d rules" % (len(edits), len(rules)))

        work = body_dir.rstrip("/\\") + "_fill"
        out = body_dir.rstrip("/\\") + "_filled"
        os.makedirs(work, exist_ok=True)
        out_root = folder_paths.get_output_directory()
        work_rel = os.path.relpath(work, out_root)

        editor = KleinEditor({"4b": (klein_4b, klein_4b_text_encoder),
                              "9b": (klein_9b, klein_9b_text_encoder)}, klein_vae)
        splitter = AutoSplitOrchestratorSAM3()
        pbar = ProgressBar(max(1, len(edits)) + 1)
        plan, thumbs = [], []
        for j in edits:
            t = time.time()
            small, up = editor.edit(image, j["prompt"], size=j.get("model", "4b"), seed=edit_seed,
                                    steps=steps, megapixels=megapixels)
            up_rgb = _to_rgb(up)
            img_path = os.path.join(work, j["name"] + "_up.png")
            Image.fromarray(up_rgb).save(img_path)
            Image.fromarray(_to_rgb(small)).save(os.path.join(work, j["name"] + "_edit.png"))
            thumbs.append(_to_rgb(small))
            splitter.run_pipeline(
                image=up, depth_map=depth_map, sam3_model=sam3_model,
                part_labels="\n".join(j["sam_labels"]), character_name=j["name"],
                output_directory=work_rel, padding=20, confidence_threshold=0.3,
                occlusion_threshold=0.15, mask_expand_pixels=15, mask_blur=1,
                enable_inpainting=False, output_mode="alpha_mask", mask_output_mode="none",
                auto_lr_split_parts="", lr_convention=body_meta.get("lr_convention", "anatomical"),
                cluster_hair_front_back=False, passthrough_parts="", write_metadata_sidecar=True,
                run_id=j["name"], depth_near=body_meta.get("depth_near", "bright"),
                character_facing=body_meta.get("character_facing", "front"), order_method="evidence")
            plan.append(peel.peel_step(j, img_path, os.path.join(work, j["name"])))
            log("%-22s %s  %4.0fs  %s" % (j["name"], j.get("model", "4b"), time.time() - t, j["prompt"][:90]))
            pbar.update(1)
        plan += rules
        with open(os.path.join(work, "plan.json"), "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=1)

        merge(runs, src, plan, out, plan_name="plan.json", log=log)
        pbar.update(1)
        log("done in %.0fs" % (time.time() - t0))
        log("  Output: %s" % out)

        # preview: the filled split pulled apart, beside the edits, all cropped
        # to the character
        filled = load_pass(out) + ([p for p in load_pass(runs[1])] if len(runs) > 1 else [])
        view = exploded(back_to_front(filled), W, H, 0.12).convert("RGB")
        sheet = _contact_sheet(view, thumbs, rigfill.picture_silhouette(src))
        report = "\n".join(lines)
        return {"ui": {"autosplit_fill": [{"folder": out, "work": work}]},
                "result": (report, _to_tensor(sheet))}


NODE_CLASS_MAPPINGS = {"AutoSplitFillKlein": AutoSplitFillKlein}
NODE_DISPLAY_NAME_MAPPINGS = {"AutoSplitFillKlein": "Auto-Split Fill (Klein peels)"}
