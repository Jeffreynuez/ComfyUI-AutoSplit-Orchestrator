"""
2D Character Auto-Split: SAM3 Orchestrator Node
================================================

Splits a flat character illustration into named, draw-ordered, left/right
labelled part PNGs plus a parts_metadata.json sidecar (schema 2).

  Phase 1    SAM 3 text-grounded segmentation, every instance kept, the image
             encoded once per run
  Phase 1.5  bilateral pairs named from geometry under an explicit convention
             (never from the word "left" in a prompt); hair split by depth
  Phase 2    mask cleanup, draw order from evidence in the picture
             (autosplit_core.ordering), crop + save, metadata

REQUIREMENTS:
  - comfyui-easy-sam3 (EASY_SAM3_MODEL)
"""

import os
import json
import traceback

try:
    from .autosplit_core import masks as core_masks
    from .autosplit_core import ordering as core_ordering
    from .autosplit_core import sides as core_sides
except ImportError:  # imported outside the package (tests, tools)
    from autosplit_core import masks as core_masks
    from autosplit_core import ordering as core_ordering
    from autosplit_core import sides as core_sides


class AutoSplitOrchestratorSAM3:
    """
    SAM3-based orchestrator for 2D character part splitting.

    Processes each body part one at a time using SAM3's text-grounded segmentation:
      text-prompt segment -> depth check -> (optional inpaint) -> crop -> save
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "sam3_model": ("EASY_SAM3_MODEL",),
                "part_labels": ("STRING", {
                    "default": "head\nhair\ntorso\nleft arm\nright arm\nleft hand\nright hand\nleft leg\nright leg\nleft foot\nright foot",
                    "multiline": True,
                    "placeholder": "Body parts to segment (one per line)",
                    "tooltip": "One body part per line. These are sent to SAM3 as text prompts for segmentation."
                }),
                "character_name": ("STRING", {
                    "default": "character",
                    "placeholder": "Name prefix for output files",
                    "tooltip": "Used as the filename prefix for all saved parts (e.g. 'character_head.png')."
                }),
                "output_directory": ("STRING", {
                    "default": "split_parts_sam3",
                    "placeholder": "Output folder name (inside ComfyUI output/)",
                    "tooltip": "Folder name under ComfyUI's output directory where part PNGs are saved."
                }),
                "padding": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 200,
                    "step": 5,
                    "display": "slider",
                    "tooltip": "Extra pixels around each cropped part bounding box."
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.30,
                    "min": 0.05,
                    "max": 0.95,
                    "step": 0.05,
                    "tooltip": "SAM3 confidence threshold for text-grounded detection. Lower = more detections, higher = stricter."
                }),
                "occlusion_threshold": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Depth difference to consider a region occluded by another part."
                }),
                "mask_expand_pixels": ("INT", {
                    "default": 15,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "How many pixels to expand the mask when inpainting occluded areas."
                }),
                "mask_blur": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Gaussian blur radius applied to segmented part mask edges. 0 = no blur (sharp edges)."
                }),
                "enable_inpainting": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Legacy OpenCV (Telea) fill of occluded regions. Off by default: it leaves grey smears. Hidden-part fill belongs in the Klein fill stage."
                }),
                "output_mode": (["alpha_mask", "solid_background"], {
                    "default": "solid_background",
                    "tooltip": "alpha_mask: RGBA with mask as alpha. solid_background: part on solid color + separate blank mask PNG."
                }),
                "background_color": ("COLORCODE", {
                    "default": "#FFFFFF",
                    "tooltip": "Background color when output_mode is solid_background. Click to open color picker. Ignored for alpha_mask mode."
                }),
                "mask_output_mode": (["blank", "part_silhouette", "none"], {
                    "default": "blank",
                    "tooltip": "blank: all-black mask for manual painting. part_silhouette: white where the part is, black elsewhere (the segmentation mask). none: no mask file generated."
                }),
                "auto_lr_split_parts": ("STRING", {
                    "default": "hands\nfeet\nears",
                    "multiline": True,
                    "placeholder": "Parts to auto-split into Left/Right (one per line)",
                    "tooltip": "Parts that should be auto-split into left/right via connected components. Use the form that matches your part_labels (e.g. 'hands'). Each becomes <part>_left and <part>_right."
                }),
                "lr_convention": (["anatomical", "viewer"], {
                    "default": "anatomical",
                    "tooltip": "anatomical = character's own L/R (rigger convention). viewer = camera-relative L/R."
                }),
                "cluster_hair_front_back": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Split detected 'hair' into hair_front (closer to camera) and hair_back (farther) by depth clustering."
                }),
                "passthrough_parts": ("STRING", {
                    "default": "nose\nmouth",
                    "multiline": True,
                    "placeholder": "Parts to keep as source pixels (one per line)",
                    "tooltip": "Parts whose pixels are kept verbatim — no inpainting, no compositing. Critical for tiny features the inpainter destroys."
                }),
                "write_metadata_sidecar": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Write parts_metadata.json with bbox, depth_median, and front-to-back z-order for rigging."
                }),
            },
            "optional": {
                "florence2_model": ("FL2MODEL",),
                "run_id": ("STRING", {
                    "default": "",
                    "tooltip": "When set, parts go to <output_directory>/<run_id>/ so runs never mix. The Krita plugin sets this for every split."
                }),
                "depth_near": (["bright", "dark"], {
                    "default": "bright",
                    "tooltip": "Which end of the depth map is closer to the viewer. Depth Anything V2 = bright. Marigold = dark."
                }),
                "character_facing": (["front", "back"], {
                    "default": "front",
                    "tooltip": "Back views flip anatomical left/right (the character's left is then on the viewer's left)."
                }),
                "order_method": (["evidence", "name_table"], {
                    "default": "evidence",
                    "tooltip": "evidence: decide draw order from colour, containment and depth in the picture, with the rigger name table as tie-breaker. name_table: the June behaviour."
                }),
                "retry_confidence": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Second chance for labels that found nothing: retry once at this lower confidence. Recovered parts are flagged low_confidence; a result that repeats a part already found is refused. 0 turns the retry off."
                }),
                "crop_transform": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Wire from Head Crop + Upscale 'transform' on the facial pass so part positions are written in full-image coordinates."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("report", "preview_grid",)
    FUNCTION = "run_pipeline"
    CATEGORY = "2D Character Split"
    OUTPUT_NODE = True

    # Prompt variations to help SAM3 find specific parts
    PROMPT_VARIATIONS = {
        'head': ['face', 'head'],
        'face': ['face', 'head'],
        'eyes': ['eyes', 'eye', 'both eyes'],
        'eye': ['eye', 'eyes'],
        'iris': ['iris', 'eye pupil', 'eyeball', 'eye iris'],
        'irides': ['iris', 'eye pupil', 'eyeball'],
        'left eye': ['left eye', 'left eyeball', 'eye on the left side'],
        'right eye': ['right eye', 'right eyeball', 'eye on the right side'],
        'left iris': ['left iris', 'left eye pupil', 'left eyeball', 'iris on the left side'],
        'right iris': ['right iris', 'right eye pupil', 'right eyeball', 'iris on the right side'],
        'nose': ['nose', 'character nose'],
        'mouth': ['mouth', 'lips', 'character mouth'],
        'eyebrow': ['eyebrow', 'eyebrows', 'brow'],
        'ear': ['ear', 'ears'],
        'ears': ['ears', 'ear'],
        'left hand': ['left hand', 'hand on the left side'],
        'right hand': ['right hand', 'hand on the right side'],
        'left foot': ['left foot', 'foot on the left side'],
        'right foot': ['right foot', 'foot on the right side'],
        'left leg': ['left leg', 'left thigh and shin'],
        'right leg': ['right leg', 'right thigh and shin'],
        'left arm': ['left arm', 'left sleeve'],
        'right arm': ['right arm', 'right sleeve'],
    }

    # Max allowed mask area as % of image — anything larger is a full-body misdetection
    MAX_PART_AREA_PCT = 25.0
    # Min mask area as % of image — anything smaller is noise.
    # Must be very low (0.01%) to allow facial features (eyes, nose, mouth)
    # on full-body character sheets where they are < 0.1% of total pixels.
    MIN_PART_AREA_PCT = 0.01

    # -------------------------------------------------
    # SAM3 text-grounded segmentation for a single part
    # -------------------------------------------------
    def _sam3_state(self, image_tensor, sam3_model, cache):
        """Encode the picture ONCE per run. SAM 3's image backbone is the slow
        part; the June node re-encoded it for every prompt variation (about 50
        times per character). Text prompts then reuse the cached features."""
        import numpy as np
        from PIL import Image as PILImage
        key = (id(image_tensor), tuple(image_tensor.shape))
        if cache.get("key") == key and "state" in cache:
            return cache["state"]
        img = image_tensor[0, :, :, :3] if image_tensor.shape[-1] > 3 else image_tensor[0]
        img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        state = sam3_model["processor"].set_image(PILImage.fromarray(img_np))
        cache.clear()
        cache.update(key=key, state=state, rgb=img_np)
        return state

    def _segment_part_sam3(self, image_tensor, sam3_model, part_name, confidence,
                           cache=None, extra_prompts=None):
        """
        Text-grounded segmentation with prompt variations.

        Returns (best_mask, best_score, best_prompt, instances):
          best_mask   torch float [H,W] - the top instance, or None
          instances   every instance the winning prompt found that passed the
                      area filter, as (numpy bool mask, score), best first.
                      SAM 3 is an all-instances model: "eyes" comes back as two
                      instances, and the June node threw the second one away.
        """
        import torch
        import numpy as np

        processor = sam3_model['processor']
        if cache is None:
            cache = {}
        state = self._sam3_state(image_tensor, sam3_model, cache)
        B, H, W, C = image_tensor.shape
        image_area = H * W
        processor.set_confidence_threshold(confidence)

        prompt_key = part_name.lower().strip()
        variations = list(self.PROMPT_VARIATIONS.get(prompt_key, [part_name]))
        if part_name not in variations:
            variations.append(part_name)
        for p in (extra_prompts or []):
            if p not in variations:
                variations.append(p)

        best = (None, 0.0, None, [])
        for var_idx, prompt_text in enumerate(variations):
            try:
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(prompt_text, state)
            except Exception as e:
                print("[SAM3] Error with prompt '%s': %s" % (prompt_text, e))
                traceback.print_exc()
                continue
            masks = state.get('masks', None)
            scores = state.get('scores', None)
            if masks is None or len(masks) == 0:
                print("[SAM3] '%s' (prompt '%s'): no masks" % (part_name, prompt_text))
                continue
            found = []
            for m_idx in range(len(masks)):
                md = masks[m_idx]
                score = float(scores[m_idx].item()) if scores is not None else 1.0
                m_np = md.squeeze().cpu().numpy() if isinstance(md, torch.Tensor) else np.array(md).squeeze()
                m_bin = m_np > 0.5
                if m_bin.shape != (H, W):
                    t = torch.from_numpy(m_bin.astype(np.float32))[None, None]
                    m_bin = torch.nn.functional.interpolate(
                        t, size=(H, W), mode='bilinear', align_corners=False)[0, 0].numpy() > 0.5
                pct = m_bin.sum() / image_area * 100
                if pct > self.MAX_PART_AREA_PCT or pct < self.MIN_PART_AREA_PCT:
                    continue
                found.append((m_bin, score))
            found.sort(key=lambda t: -t[1])
            print("[SAM3] '%s' (prompt '%s'): %d usable instance(s) %s" % (
                part_name, prompt_text, len(found), [round(s, 3) for _, s in found]))
            if found and found[0][1] > best[1]:
                best = (torch.from_numpy(found[0][0].astype(np.float32)), found[0][1],
                        prompt_text, found)
            if best[1] >= 0.85:
                break  # good enough; skip the remaining phrasings

        if best[0] is not None:
            print("[SAM3] ACCEPTED '%s' via '%s' (score=%.3f, %d instance(s))" % (
                part_name, best[2], best[1], len(best[3])))
        else:
            print("[SAM3] FAILED: no valid mask for '%s'" % part_name)
        return best

    # -------------------------------------------------
    # Florence-2 detection + SAM3 bbox segmentation (fallback)
    # -------------------------------------------------
    def _detect_and_segment_with_florence(self, image_tensor, florence2_model, sam3_model, part_name, confidence):
        """
        Fallback: Use Florence-2 for detection, then SAM3 with bbox hint.
        Returns: (mask_tensor [H,W], score) or (None, 0.0)
        """
        import torch
        import numpy as np
        import torchvision.transforms.functional as TF
        import comfy.model_management as mm

        processor_f = florence2_model['processor']
        model_f = florence2_model['model']
        dtype_f = florence2_model['dtype']
        device_f = mm.get_torch_device()

        model_f.to(device_f)

        B, H, W, C = image_tensor.shape
        img_permuted = image_tensor.permute(0, 3, 1, 2)
        if img_permuted.shape[1] > 3:
            img_permuted = img_permuted[:, :3, :, :]
        image_pil = TF.to_pil_image(img_permuted[0])
        image_area = W * H

        # Try to get a bbox from Florence-2
        prompt_key = part_name.lower().strip()
        # copy: appending to the class table would leak labels between runs
        variations = list(self.PROMPT_VARIATIONS.get(prompt_key, [part_name]))
        if part_name not in variations:
            variations.append(part_name)

        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'

        for prompt_text in variations:
            prompt = task_prompt + " " + prompt_text

            inputs = processor_f(
                text=prompt, images=image_pil,
                return_tensors="pt", do_rescale=False
            ).to(dtype_f).to(device_f)

            generated_ids = model_f.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
                use_cache=False,
            )

            results = processor_f.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor_f.post_process_generation(results, task=task_prompt, image_size=(W, H))

            if task_prompt not in parsed:
                continue

            bboxes = parsed[task_prompt].get('bboxes', [])
            if not bboxes:
                continue

            # Filter bboxes — pick smallest valid one
            scored = []
            for bbox in bboxes:
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                area_pct = (bw * bh) / image_area * 100
                scored.append((area_pct, bbox))
            scored.sort(key=lambda x: x[0])

            best_bbox = None
            for area_pct, bbox in scored:
                if 0.5 <= area_pct < self.MAX_PART_AREA_PCT:
                    best_bbox = bbox
                    break

            if best_bbox is None:
                continue

            # Convert Florence bbox [x1,y1,x2,y2] pixel coords to SAM3 format [cx,cy,w,h] normalized 0-1
            x1, y1, x2, y2 = best_bbox
            cx = ((x1 + x2) / 2.0) / W
            cy = ((y1 + y2) / 2.0) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            sam3_box = [cx, cy, bw, bh]

            print("[Florence+SAM3] Using bbox from Florence-2: %s -> SAM3 box: %s" % (
                str(best_bbox), str(sam3_box)))

            # Now use SAM3 with this bbox
            processor_s = sam3_model['processor']
            processor_s.set_confidence_threshold(confidence)

            img_rgb = image_tensor[:, :, :, :3] if C > 3 else image_tensor
            img_np = (img_rgb[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img_np)

            state = processor_s.set_image(pil_img)
            state = processor_s.add_boxes_prompts([sam3_box], [True], state)

            masks = state.get('masks', None)
            scores = state.get('scores', None)

            if masks is not None and len(masks) > 0:
                mask_data = masks[0]
                score = scores[0].item() if scores is not None else 1.0
                if isinstance(mask_data, torch.Tensor):
                    m_np = mask_data.squeeze().cpu().numpy()
                else:
                    m_np = np.array(mask_data).squeeze()

                m_binary = (m_np > 0.5).astype(np.float32)
                if m_binary.shape != (H, W):
                    m_tensor = torch.from_numpy(m_binary).unsqueeze(0).unsqueeze(0)
                    m_tensor = torch.nn.functional.interpolate(
                        m_tensor, size=(H, W), mode='bilinear', align_corners=False
                    ).squeeze()
                    m_binary_final = (m_tensor > 0.5).float()
                else:
                    m_binary_final = torch.from_numpy(m_binary)

                mask_pct = m_binary_final.sum().item() / (H * W) * 100
                if mask_pct <= self.MAX_PART_AREA_PCT:
                    print("[Florence+SAM3] ACCEPTED mask: %.1f%% of image, score=%.3f" % (mask_pct, score))
                    return m_binary_final, score
                else:
                    print("[Florence+SAM3] Rejected: mask too large (%.1f%%)" % mask_pct)

        return None, 0.0

    # -------------------------------------------------
    # Overlap-aware occlusion detection (same as SAM2 version)
    # -------------------------------------------------
    def _check_occlusion(self, depth_map, current_mask, all_masks, current_idx, threshold):
        """Downsampled overlap-aware occlusion check."""
        import torch

        ANALYSIS_SIZE = 256

        if depth_map.dim() == 4:
            depth = depth_map[0, :, :, 0]
        elif depth_map.dim() == 3:
            depth = depth_map[:, :, 0]
        else:
            depth = depth_map
        while depth.dim() > 2:
            depth = depth.squeeze(0)

        mask_2d = current_mask.clone()
        while mask_2d.dim() > 2:
            mask_2d = mask_2d.squeeze(0)
        while mask_2d.dim() < 2:
            mask_2d = mask_2d.unsqueeze(0)

        H, W = mask_2d.shape
        device = mask_2d.device

        def downsample(t, size=ANALYSIS_SIZE):
            return torch.nn.functional.interpolate(
                t.unsqueeze(0).unsqueeze(0).float(),
                size=(size, size), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)

        depth_small = downsample(depth.to(device))
        mask_small = downsample(mask_2d)

        active_small = mask_small > 0.5
        if active_small.sum() == 0:
            return False, torch.zeros_like(mask_2d), 0.0

        current_median_depth = depth_small[active_small].median()
        occlusion_small = torch.zeros(ANALYSIS_SIZE, ANALYSIS_SIZE, device=device)

        for i, other_mask in enumerate(all_masks):
            if i == current_idx or other_mask is None:
                continue

            other_2d = other_mask.to(device).clone()
            while other_2d.dim() > 2:
                other_2d = other_2d.squeeze(0)
            while other_2d.dim() < 2:
                other_2d = other_2d.unsqueeze(0)

            other_small = downsample(other_2d)
            overlap = (mask_small > 0.5) & (other_small > 0.5)
            if overlap.sum() == 0:
                continue

            other_active = other_small > 0.5
            if other_active.sum() == 0:
                continue

            other_median_depth = depth_small[other_active].median()
            if other_median_depth < (current_median_depth - threshold):
                occlusion_small = torch.logical_or(occlusion_small.bool(), overlap).float()

        occlusion_pct = 0.0
        if active_small.sum() > 0:
            occlusion_pct = (occlusion_small.sum().float() / active_small.sum().float() * 100).item()

        is_occluded = occlusion_pct > 2.0

        if is_occluded:
            occlusion_mask = torch.nn.functional.interpolate(
                occlusion_small.unsqueeze(0).unsqueeze(0),
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)
            occlusion_mask = (occlusion_mask > 0.5).float()
        else:
            occlusion_mask = torch.zeros_like(mask_2d)

        return is_occluded, occlusion_mask, occlusion_pct

    # -------------------------------------------------
    # Mask expansion
    # -------------------------------------------------
    def _expand_mask(self, mask, expand_px):
        import torch
        if expand_px <= 0:
            return mask
        kernel_size = expand_px * 2 + 1
        expanded = torch.nn.functional.max_pool2d(
            mask.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=kernel_size, stride=1, padding=expand_px
        ).squeeze()
        return (expanded > 0.5).float()

    # -------------------------------------------------
    # Bilateral L/R auto-split
    # -------------------------------------------------
    def _split_lr_mask(self, mask_2d, anatomical=True, min_area=50):
        """
        Split a single mask containing two blobs into left/right via
        cv2.connectedComponentsWithStats. Returns (left_mask, right_mask)
        as torch tensors, or (None, None) if fewer than 2 blobs found.
        Convention: smaller centroid x = viewer-left. If anatomical=True,
        the outputs are swapped so 'left' = character's own left (mirror).
        """
        import torch
        import numpy as np
        try:
            import cv2
        except ImportError:
            print("[AutoSplit-SAM3] cv2 not available, skipping L/R split")
            return None, None

        m_np = mask_2d.detach().cpu().numpy()
        binary = (m_np > 0.5).astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        valid = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        if len(valid) < 2:
            return None, None
        valid.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
        a, b = valid[0], valid[1]
        cx_a = stats[a][0] + stats[a][2] / 2.0
        cx_b = stats[b][0] + stats[b][2] / 2.0
        # viewer-left = smaller cx
        if cx_a <= cx_b:
            left_id, right_id = a, b
        else:
            left_id, right_id = b, a
        left = torch.from_numpy((labels == left_id).astype(np.float32))
        right = torch.from_numpy((labels == right_id).astype(np.float32))
        if anatomical:
            left, right = right, left  # mirror to character's own L/R
        return left, right

    # -------------------------------------------------
    # Hair front/back clustering by depth
    # -------------------------------------------------
    def _cluster_hair_by_depth(self, mask_2d, depth_map, min_area=80):
        """
        Connected-components on the hair mask, take the two largest, sort
        by median depth — front = lower depth value (closer), back = higher.
        Returns (front_mask, back_mask) torch tensors, or (None, None).
        """
        import torch
        import numpy as np
        try:
            import cv2
        except ImportError:
            return None, None

        m_np = mask_2d.detach().cpu().numpy()
        binary = (m_np > 0.5).astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        valid = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        if len(valid) < 2:
            return None, None
        valid.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])

        # Extract a 2D depth array from an IMAGE tensor [B,H,W,C] or [H,W,C] or [H,W]
        d_np = self._depth_to_2d_np(depth_map)
        # Resize depth to mask shape if needed
        if d_np.shape != m_np.shape:
            d_np = cv2.resize(d_np, (m_np.shape[1], m_np.shape[0]), interpolation=cv2.INTER_LINEAR)

        scored = []
        for cid in valid[:4]:  # consider top 4 just in case hair has stragglers
            blob = (labels == cid)
            if blob.sum() == 0:
                continue
            med_depth = float(np.median(d_np[blob]))
            scored.append((med_depth, cid, blob))
        if len(scored) < 2:
            return None, None
        scored.sort(key=lambda x: x[0])  # ascending → front first (lower depth = closer)
        front_blob = scored[0][2]
        back_blob = scored[-1][2]
        return (
            torch.from_numpy(front_blob.astype(np.float32)),
            torch.from_numpy(back_blob.astype(np.float32)),
        )

    # -------------------------------------------------
    # Per-part depth median (uses workflow's depth_map)
    # -------------------------------------------------
    def _depth_to_2d_np(self, depth_map):
        """Convert any incoming depth representation to a 2D numpy array.
        Handles IMAGE tensors [B,H,W,C], [H,W,C], [H,W], MASK tensors, raw numpy."""
        import numpy as np
        try:
            import torch as _torch
            is_tensor = isinstance(depth_map, _torch.Tensor)
        except ImportError:
            is_tensor = False

        if is_tensor:
            d = depth_map.detach().cpu().numpy()
        else:
            d = np.array(depth_map)

        # Drop batch dim if present
        if d.ndim == 4:
            d = d[0]  # [H,W,C]
        # Drop channel dim by taking first channel (depth is grayscale)
        if d.ndim == 3:
            d = d[..., 0]
        # Now d should be 2D; if not, bail with a warning
        if d.ndim != 2:
            print("[AutoSplit-SAM3] WARN: unexpected depth shape %s, flattening" % (d.shape,))
            d = d.reshape(d.shape[0], -1)
        return d.astype(np.float32)

    def _depth_median(self, mask_2d, depth_map):
        import numpy as np
        try:
            import cv2
        except ImportError:
            return None
        m_np = mask_2d.detach().cpu().numpy() > 0.5
        if not m_np.any():
            return None
        d_np = self._depth_to_2d_np(depth_map)
        if d_np.shape != m_np.shape:
            d_np = cv2.resize(d_np, (m_np.shape[1], m_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        return float(np.median(d_np[m_np]))

    # -------------------------------------------------
    # Default z-order ranks (front to back, lower = in front)
    # -------------------------------------------------
    DEFAULT_Z_ORDER = [
        'nose', 'mouth', 'eyes', 'eye', 'eyewhite', 'irides', 'eyelash', 'eyebrow',
        'face',
        'eyewear', 'earwear', 'ears', 'ear',
        'hair_front', 'hairf', 'front hair',
        'headwear', 'hat',
        'neck', 'neckwear',
        'topwear', 'shirt', 'jacket',
        'arms', 'arm', 'left arm', 'right arm',
        'hands', 'hand', 'handwear', 'left hand', 'right hand',
        'bottomwear', 'pants', 'skirt',
        'legs', 'leg', 'left leg', 'right leg', 'legwear',
        'feet', 'foot', 'left foot', 'right foot', 'footwear',
        'torso', 'body',
        'tail',
        'hair_back', 'hairb', 'back hair', 'hair',
        'wings',
        'objects', 'background',
    ]

    def _z_rank(self, tag):
        """Return a z-rank for sorting parts front-to-back (kept for old
        callers; the orchestrator now uses autosplit_core.ordering)."""
        return core_ordering.prior_rank(tag)

    def _z_rank_legacy(self, tag):
        t = tag.lower().strip()
        if t in self.DEFAULT_Z_ORDER:
            return self.DEFAULT_Z_ORDER.index(t)
        # Check left/right variants
        for prefix in ('left_', 'right_', 'left ', 'right '):
            if t.startswith(prefix):
                base = t[len(prefix):]
                if base in self.DEFAULT_Z_ORDER:
                    return self.DEFAULT_Z_ORDER.index(base)
        return len(self.DEFAULT_Z_ORDER) + 100  # unknown → push to back

    # -------------------------------------------------
    # Crop + pad + save (BG removal happens here)
    # -------------------------------------------------
    @staticmethod
    def _hex_to_rgb_float(hex_color):
        """Convert a hex color string like '#FF8800' to (r, g, b) floats in 0-1."""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join(c * 2 for c in hex_color)
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)

    def _clean_mask(self, mask_2d, close_ksize=5, island_min_frac=0.02):
        """
        Clean a single part's binary mask:
          1) morphological close — bridge small gaps, smooth ragged edges
          2) fill fully-enclosed holes (via border flood fill)
          3) drop tiny disconnected islands (noise) smaller than
             island_min_frac of the largest connected component
        Returns a torch tensor on the same device/dtype. Falls back to the
        input mask unchanged if cv2 is unavailable or anything fails.
        Note: runs per-part AFTER L/R split, so each input is a single
        intended blob — keeping the largest comps is safe for clothing too.
        """
        import torch
        try:
            import numpy as np
            import cv2
            device = mask_2d.device
            dtype = mask_2d.dtype
            arr = (mask_2d.detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
            if arr.max() == 0:
                return mask_2d

            # 1) Morphological close
            if close_ksize and close_ksize > 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
                arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)

            # 2) Fill enclosed holes: flood the border-connected background,
            #    invert to isolate holes, OR back into the mask.
            h, w = arr.shape
            ff = arr.copy()
            ff_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(ff, ff_mask, (0, 0), 255)
            holes = cv2.bitwise_not(ff)
            arr = cv2.bitwise_or(arr, holes)

            # 3) Drop tiny islands relative to the largest component
            num, labels, stats, _ = cv2.connectedComponentsWithStats(
                (arr > 0).astype(np.uint8), connectivity=8
            )
            if num > 2:  # label 0 is background; >2 means multiple fg comps
                areas = stats[1:, cv2.CC_STAT_AREA]
                max_area = float(areas.max())
                keep = np.zeros_like(arr)
                for lbl in range(1, num):
                    if stats[lbl, cv2.CC_STAT_AREA] >= island_min_frac * max_area:
                        keep[labels == lbl] = 255
                arr = keep

            cleaned = torch.from_numpy((arr > 0).astype(np.float32)).to(device=device, dtype=dtype)
            return cleaned
        except Exception as e:
            print("[AutoSplit-SAM3] _clean_mask skipped (%s)" % e)
            return mask_2d

    def _crop_and_save(self, image_tensor, mask, part_name, character_name,
                       output_dir, padding, output_mode="solid_background",
                       background_color="#FFFFFF", mask_blur=0,
                       mask_output_mode="blank", passthrough=False):
        """
        Crop the masked region from the image and save as PNG.

        Two output modes:
          - alpha_mask: RGBA with mask as alpha (original behavior)
          - solid_background: RGB part composited on solid color background,
            plus a separate blank mask PNG for manual painting in ComfyUI
        """
        import torch
        import numpy as np
        from PIL import Image

        img = image_tensor[0].cpu().numpy()  # (H, W, C)
        mask_np = mask.cpu().numpy()

        # Strip alpha if present — use only RGB
        if img.shape[2] > 3:
            img = img[:, :, :3]

        mask_h, mask_w = mask_np.shape[:2]
        img_h, img_w = img.shape[:2]
        if img_h != mask_h or img_w != mask_w:
            pil_img = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
            pil_img = pil_img.resize((mask_w, mask_h), Image.LANCZOS)
            img = np.array(pil_img).astype(np.float32) / 255.0

        ys, xs = np.where(mask_np > 0.5)
        if len(ys) == 0:
            return None, None, None

        y_min = max(0, int(ys.min()) - padding)
        y_max = min(mask_h - 1, int(ys.max()) + padding)
        x_min = max(0, int(xs.min()) - padding)
        x_max = min(mask_w - 1, int(xs.max()) + padding)
        bbox_xyxy = [x_min, y_min, x_max, y_max]

        img_crop = img[y_min:y_max+1, x_min:x_max+1, :]
        mask_crop = mask_np[y_min:y_max+1, x_min:x_max+1]

        safe_name = part_name.replace(" ", "_").replace("/", "_")
        filename = "{0}_{1}.png".format(character_name, safe_name)
        filepath = os.path.join(output_dir, filename)

        # Apply mask blur if requested
        if mask_blur > 0 and not passthrough:
            from PIL import ImageFilter
            mask_pil_blur = Image.fromarray(
                (mask_crop * 255).clip(0, 255).astype(np.uint8), 'L'
            )
            mask_pil_blur = mask_pil_blur.filter(
                ImageFilter.GaussianBlur(radius=mask_blur)
            )
            mask_crop = np.array(mask_pil_blur).astype(np.float32) / 255.0

        if output_mode == "solid_background":
            if passthrough:
                # Source-pixel passthrough: keep the entire bbox as source RGB,
                # no compositing onto a background color. Critical for tiny
                # features (nose/mouth) the inpainter would destroy.
                composited = img_crop.copy()
                print("[AutoSplit-SAM3] Passthrough mode for '%s' — source pixels preserved" % part_name)
            else:
                # Composite the part onto a solid color background (hex color)
                bg_r, bg_g, bg_b = self._hex_to_rgb_float(background_color)
                bg = np.zeros_like(img_crop, dtype=np.float32)
                bg[:, :, 0] = bg_r
                bg[:, :, 1] = bg_g
                bg[:, :, 2] = bg_b

                # Blend: where mask is 1 show the part, where 0 show the background
                mask_3d = np.expand_dims(mask_crop, axis=-1)
                composited = img_crop * mask_3d + bg * (1.0 - mask_3d)

            # Save as RGB (no alpha)
            rgb_8bit = (composited * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(rgb_8bit, 'RGB')
            pil_img.save(filepath)

            # Save companion mask based on mask_output_mode
            if mask_output_mode == "blank":
                # All-black mask for manual painting in ComfyUI
                mask_filename = "{0}_{1}_mask.png".format(character_name, safe_name)
                mask_filepath = os.path.join(output_dir, mask_filename)
                blank_mask = np.zeros((*img_crop.shape[:2],), dtype=np.uint8)
                mask_pil = Image.fromarray(blank_mask, 'L')
                mask_pil.save(mask_filepath)
                print("[AutoSplit-SAM3] Saved blank mask: %s" % mask_filepath)
            elif mask_output_mode == "part_silhouette":
                # Actual segmentation mask: white = part, black = background
                mask_filename = "{0}_{1}_mask.png".format(character_name, safe_name)
                mask_filepath = os.path.join(output_dir, mask_filename)
                silhouette = (mask_crop * 255).clip(0, 255).astype(np.uint8)
                mask_pil = Image.fromarray(silhouette, 'L')
                mask_pil.save(mask_filepath)
                print("[AutoSplit-SAM3] Saved silhouette mask: %s" % mask_filepath)
            # else: mask_output_mode == "none" — skip mask generation entirely

            crop_tensor = torch.from_numpy(composited).unsqueeze(0)
        else:
            # Original behavior: RGBA with mask as alpha channel
            rgba = np.zeros((*img_crop.shape[:2], 4), dtype=np.float32)
            rgba[:, :, :3] = img_crop
            rgba[:, :, 3] = mask_crop

            rgba_8bit = (rgba * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(rgba_8bit, 'RGBA')
            pil_img.save(filepath)

            crop_tensor = torch.from_numpy(img_crop).unsqueeze(0)

        return filepath, crop_tensor, bbox_xyxy

    # -------------------------------------------------
    # Simple inpainting
    # -------------------------------------------------
    def _simple_inpaint(self, image_tensor, mask, expand_px):
        import torch
        import numpy as np

        # Ensure we work with RGB only
        img = image_tensor[0].cpu().numpy()
        if img.shape[2] > 3:
            img = img[:, :, :3]
        img_np = (img * 255).clip(0, 255).astype(np.uint8)

        mask_expanded = self._expand_mask(mask, expand_px)
        mask_np = (mask_expanded.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        try:
            import cv2
            result = cv2.inpaint(img_np, mask_np.astype(np.uint8), inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
            return result_tensor
        except ImportError:
            from PIL import ImageFilter, Image
            pil_img = Image.fromarray(img_np)
            blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=15))
            blurred_np = np.array(blurred).astype(np.float32) / 255.0
            img_float = img_np.astype(np.float32) / 255.0
            mask_3d = np.expand_dims(mask_np.astype(np.float32) / 255.0, -1)
            result = img_float * (1 - mask_3d) + blurred_np * mask_3d
            return torch.from_numpy(result).unsqueeze(0)

    # -------------------------------------------------
    # Preview grid builder
    # -------------------------------------------------
    def _build_preview_grid(self, crops):
        import torch
        max_h = max(c.shape[1] for c in crops)
        max_w = max(c.shape[2] for c in crops)

        padded = []
        for crop in crops:
            h, w = crop.shape[1], crop.shape[2]
            pad_h = max_h - h
            pad_w = max_w - w
            if pad_h > 0 or pad_w > 0:
                crop = torch.nn.functional.pad(
                    crop.permute(0, 3, 1, 2),
                    (0, pad_w, 0, pad_h),
                    mode='constant', value=0
                ).permute(0, 2, 3, 1)
            padded.append(crop)

        cols = min(4, len(padded))
        rows = (len(padded) + cols - 1) // cols

        grid_rows = []
        for r in range(rows):
            row_imgs = []
            for c in range(cols):
                idx = r * cols + c
                if idx < len(padded):
                    row_imgs.append(padded[idx][0])
                else:
                    row_imgs.append(torch.zeros(max_h, max_w, 3))
            grid_rows.append(torch.cat(row_imgs, dim=1))

        grid = torch.cat(grid_rows, dim=0)
        return grid.unsqueeze(0)

    # -------------------------------------------------
    # Main pipeline
    # -------------------------------------------------
    def run_pipeline(self, image, depth_map, sam3_model,
                     part_labels, character_name, output_directory, padding,
                     confidence_threshold, occlusion_threshold,
                     mask_expand_pixels, mask_blur=0, enable_inpainting=False,
                     output_mode="solid_background", background_color="#FFFFFF",
                     mask_output_mode="blank",
                     auto_lr_split_parts="hands\nfeet\nears",
                     lr_convention="anatomical",
                     cluster_hair_front_back=True,
                     passthrough_parts="nose\nmouth",
                     write_metadata_sidecar=True,
                     florence2_model=None, run_id="", depth_near="bright",
                     character_facing="front", order_method="evidence",
                     crop_transform="", retry_confidence=0.2):
        """
        Phase 1    segment every label (SAM 3, all instances kept)
        Phase 1.5  name bilateral pairs from geometry, split hair by depth
        Phase 2    clean masks, decide draw order from evidence, crop + save,
                   write parts_metadata.json (schema 2)
        """
        import torch
        import numpy as np
        import folder_paths
        from comfy.utils import ProgressBar

        run_id = (run_id or "").strip()
        output_dir = os.path.join(folder_paths.get_output_directory(), output_directory)
        if run_id:
            output_dir = os.path.join(output_dir, run_id)
        os.makedirs(output_dir, exist_ok=True)

        transform = None
        if crop_transform:
            try:
                transform = json.loads(crop_transform) if isinstance(crop_transform, str) else dict(crop_transform)
            except Exception as e:
                print("[AutoSplit-SAM3] crop_transform ignored (%s)" % e)
        pass_name = "facial" if transform else "body"

        parts = [p.strip() for p in part_labels.strip().split("\n") if p.strip()]
        if not parts:
            return ("No parts specified.", image)

        B, H_img, W_img, C = image.shape
        print("\n" + "=" * 60)
        print("[AutoSplit-SAM3] %s pass for '%s' -> %s" % (pass_name, character_name, output_dir))
        print("[AutoSplit-SAM3] parts: %s" % parts)
        print("=" * 60)

        report_lines = [
            "Auto-Split Report (SAM3) for '%s' (%s pass)" % (character_name, pass_name),
            "=" * 50,
            "Parts requested: %d" % len(parts),
            "",
        ]

        lr_targets = set(p.strip().lower() for p in auto_lr_split_parts.strip().split("\n") if p.strip())
        passthrough_set = set(p.strip().lower() for p in passthrough_parts.strip().split("\n") if p.strip())
        near_map = core_masks.nearness(depth_map, depth_near)
        nm = near_map
        if nm.shape != (H_img, W_img):
            nm = core_masks.resize_to(nm, (H_img, W_img))
        lo, hi = float(nm.min()), float(nm.max())
        near01 = (nm - lo) / (hi - lo) if hi > lo else np.zeros_like(nm)
        # the June helpers (occlusion check, depth_median) expect LOWER = nearer
        legacy_depth = torch.from_numpy((1.0 - near01).astype(np.float32))[None, :, :, None]

        # pairs given explicitly as "left X" / "right X"
        lowered = {p.lower(): p for p in parts}
        pair_base = {}
        for p in parts:
            pl = p.lower()
            for side_word, other in (("left ", "right "), ("right ", "left ")):
                if pl.startswith(side_word) and (other + pl[len(side_word):]) in lowered:
                    pair_base[p] = pl[len(side_word):]

        # ------------------------------------------------------------------
        # Phase 1: segment
        # ------------------------------------------------------------------
        cache = {}
        seg = {}
        pbar = ProgressBar(len(parts) + 2)
        for i, part in enumerate(parts):
            print("\n--- Segmenting %d/%d: '%s' ---" % (i + 1, len(parts), part))
            extra = [pair_base[part]] if part in pair_base else None
            mask, score, prompt, inst = None, 0.0, None, []
            try:
                mask, score, prompt, inst = self._segment_part_sam3(
                    image, sam3_model, part, confidence_threshold, cache, extra)
            except Exception as e:
                print("[AutoSplit-SAM3] SAM3 failed for '%s': %s" % (part, e))
                traceback.print_exc()
            if mask is None and florence2_model is not None:
                try:
                    mask, score = self._detect_and_segment_with_florence(
                        image, florence2_model, sam3_model, part, confidence_threshold)
                    if mask is not None:
                        prompt = "florence2:" + part
                        inst = [((mask > 0.5).cpu().numpy(), score)]
                except Exception as e:
                    print("[AutoSplit-SAM3] Florence fallback failed for '%s': %s" % (part, e))
            seg[part] = {"mask": None if mask is None else (mask > 0.5).cpu().numpy(),
                         "score": float(score), "prompt": prompt, "instances": inst}
            pbar.update(1)

        # Phase 1b: second chance. SAM 3 scores some correct answers below the
        # main threshold: on the Salena sheet "green sash" comes back at 0.27
        # with IoU 0.98 against the hand-cut layer. Retry each label that found
        # nothing once at retry_confidence, flag what comes back, and refuse a
        # result that only repeats a part already found (a low score is often
        # SAM 3 pointing at the nearest thing it does know).
        if retry_confidence and 0.0 < float(retry_confidence) < float(confidence_threshold):
            for part in parts:
                if seg[part]["mask"] is not None:
                    continue
                extra = [pair_base[part]] if part in pair_base else None
                try:
                    mask, score, prompt, inst = self._segment_part_sam3(
                        image, sam3_model, part, float(retry_confidence), cache, extra)
                except Exception as e:
                    print("[AutoSplit-SAM3] retry failed for '%s': %s" % (part, e))
                    continue
                if mask is None:
                    continue
                taken = [s["mask"] for p, s in seg.items()
                         if p != part and s["mask"] is not None and not s.get("low_confidence")]
                inst = [(m, sc) for m, sc in inst
                        if all(core_masks.iou(m, t) <= 0.7 for t in taken)]
                if not inst:
                    print("[AutoSplit-SAM3] retry for '%s' only repeated a found part; refused" % part)
                    seg[part]["retry_refused"] = True
                    continue
                print("[AutoSplit-SAM3] retry recovered '%s' via '%s' (score=%.3f)" % (
                    part, prompt, inst[0][1]))
                seg[part] = {"mask": inst[0][0], "score": float(inst[0][1]), "prompt": prompt,
                             "instances": inst, "low_confidence": True}

        # ------------------------------------------------------------------
        # Phase 1.5: name bilateral pairs, split hair
        # ------------------------------------------------------------------
        items = []   # dicts: name, mask, source, score, prompt, n_inst, lr_source, flags
        done = set()
        all_found = [s["mask"] for s in seg.values() if s["mask"] is not None]
        centre_x = None
        if all_found:
            u = np.zeros_like(all_found[0])
            for m in all_found:
                u |= m
            c = core_masks.centroid(u)
            centre_x = c[0] if c else None

        def add(name, mask, source, lr_source=None, flags=None):
            s = seg.get(source, {})
            flags = list(flags or [])
            if mask is not None and s.get("low_confidence"):
                flags.append("low_confidence")
            if mask is None and s.get("retry_refused"):
                flags.append("retry_duplicate")
            items.append({"name": name, "mask": mask, "source": source,
                          "score": s.get("score", 0.0), "prompt": s.get("prompt"),
                          "n_inst": len(s.get("instances") or []),
                          "lr_source": lr_source, "flags": flags})

        for part in parts:
            if part in done:
                continue
            pl = part.lower()
            s = seg[part]
            if part in pair_base:
                side_word = "left" if pl.startswith("left ") else "right"
                other = lowered[("right " if side_word == "left" else "left ") + pair_base[part]]
                left_lbl = part if side_word == "left" else other
                right_lbl = other if side_word == "left" else part
                done.update([left_lbl, right_lbl])
                pool = list(seg[left_lbl]["instances"]) + list(seg[right_lbl]["instances"])
                pair = core_sides.pick_pair(pool)
                if len(pair) == 2:
                    lm, rm, info = core_sides.assign(pair[0][0], pair[1][0], lr_convention, character_facing)
                    add(left_lbl, lm, left_lbl, "geometry")
                    add(right_lbl, rm, right_lbl, "geometry")
                elif len(pair) == 1 and centre_x is not None:
                    m = pair[0][0]
                    cx = core_masks.centroid(m)[0]
                    viewer_left = cx < centre_x
                    mirrored = lr_convention == "anatomical" and character_facing != "back"
                    is_left = (not viewer_left) if mirrored else viewer_left
                    keep, miss = (left_lbl, right_lbl) if is_left else (right_lbl, left_lbl)
                    add(keep, m, keep, "geometry-single", ["pair_partner_missing"])
                    add(miss, None, miss, None, ["not_detected"])
                    report_lines.append("  WARN  '%s'/'%s' -- SAM 3 found only one; kept as '%s'"
                                        % (left_lbl, right_lbl, keep))
                else:
                    add(left_lbl, None, left_lbl, None, ["not_detected"])
                    add(right_lbl, None, right_lbl, None, ["not_detected"])
                continue

            done.add(part)
            if s["mask"] is None:
                add(part, None, part, None, ["not_detected"])
                continue

            if pl in lr_targets:
                safe_base = "iris" if pl in ("iris", "irides") else pl.rstrip("s")
                pair = core_sides.pick_pair(s["instances"])
                res = None
                if len(pair) == 2:
                    res = core_sides.assign(pair[0][0], pair[1][0], lr_convention, character_facing)
                else:
                    res = core_sides.split_union(s["mask"], lr_convention, character_facing)
                if res:
                    lm, rm, info = res
                    add("%s_left" % safe_base, lm, part, info["lr_source"])
                    add("%s_right" % safe_base, rm, part, info["lr_source"])
                    continue
                report_lines.append("  WARN  '%s' -- could not find two sides" % part)

            if cluster_hair_front_back and pl == "hair":
                comps = core_masks.components(s["mask"], min_area=80)
                split = None
                if len(comps) >= 2:
                    a, b = comps[0], comps[1]
                    na, nb = float(np.median(nm[a])), float(np.median(nm[b]))
                    split = (a, b) if na >= nb else (b, a)
                else:
                    split = core_masks.split_by_depth(s["mask"], nm)
                if split:
                    add("hair_front", split[0], part, None, ["depth_split"])
                    add("hair_back", split[1], part, None, ["depth_split"])
                    continue
            add(part, s["mask"], part)

        # left/right sanity: the two sides must not be the same pixels
        by_name = {it["name"]: it for it in items}
        for it in items:
            n = it["name"].lower()
            mate = None
            if n.startswith("left "):
                mate = by_name.get("right " + it["name"][5:]) or by_name.get("Right " + it["name"][5:])
            elif n.endswith("_left"):
                mate = by_name.get(it["name"][:-5] + "_right")
            if mate and it["mask"] is not None and mate["mask"] is not None:
                if core_masks.iou(it["mask"], mate["mask"]) > 0.3:
                    it["flags"].append("lr_overlap")
                    mate["flags"].append("lr_overlap")

        # ------------------------------------------------------------------
        # Phase 2: clean, order, save
        # ------------------------------------------------------------------
        for it in items:
            if it["mask"] is not None:
                it["mask"] = core_masks.clean(it["mask"])
                if not it["mask"].any():
                    it["mask"] = None
                    it["flags"].append("empty_after_clean")
        live = [it for it in items if it["mask"] is not None]

        rgb = cache.get("rgb")
        if rgb is None:
            rgb = (image[0, :, :, :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        order = {"back_to_front": [], "z_order": {}, "edges": [], "dropped": []}
        if live:
            if order_method == "evidence":
                order = core_ordering.compute_order(rgb, {it["name"]: it["mask"] for it in live},
                                                    near=nm)
            else:
                ranked = sorted(live, key=lambda it: (core_ordering.prior_rank(it["name"]),
                                                      -float(np.median(nm[it["mask"]]))))
                names = [it["name"] for it in ranked]
                order = {"back_to_front": list(reversed(names)),
                         "z_order": {n: i for i, n in enumerate(names)}, "edges": [], "dropped": []}
        pbar.update(1)

        proc_masks_for_occ = [torch.from_numpy(it["mask"].astype(np.float32)) for it in live]
        saved_count = inpainted_count = 0
        preview_crops, metadata_entries = [], []
        sc = float((transform or {}).get("scale", 1.0) or 1.0)
        ox, oy = (transform or {}).get("x_min", 0), (transform or {}).get("y_min", 0)

        for i, it in enumerate(live):
            part = it["name"]
            mask_t = torch.from_numpy(it["mask"].astype(np.float32))
            is_passthrough = it["source"].lower() in passthrough_set or part.lower() in passthrough_set
            work_image = image
            if not is_passthrough and enable_inpainting:
                is_occ, occ_mask, occ_pct = self._check_occlusion(
                    legacy_depth, mask_t, proc_masks_for_occ, i, occlusion_threshold)
                if is_occ:
                    try:
                        work_image = self._simple_inpaint(image, occ_mask, mask_expand_pixels)
                        inpainted_count += 1
                        it["flags"].append("telea_inpainted")
                    except Exception as e:
                        print("[AutoSplit-SAM3] Inpainting failed: %s" % e)
            try:
                filepath, crop_tensor, bbox_xyxy = self._crop_and_save(
                    work_image, mask_t, part, character_name, output_dir, padding,
                    output_mode=output_mode, background_color=background_color,
                    mask_blur=mask_blur, mask_output_mode=mask_output_mode,
                    passthrough=is_passthrough)
            except Exception as e:
                print("[AutoSplit-SAM3] Save failed for '%s': %s" % (part, e))
                traceback.print_exc()
                it["flags"].append("save_failed")
                continue
            if not filepath:
                it["flags"].append("empty_crop")
                continue
            saved_count += 1
            if crop_tensor is not None:
                preview_crops.append(crop_tensor)
            safe_name = part.replace(" ", "_").replace("/", "_")
            mask_file_val = ("%s_%s_mask.png" % (character_name, safe_name)
                             if output_mode == "solid_background" and mask_output_mode != "none" else None)
            x0, y0, x1, y1 = bbox_xyxy
            entry = {
                "tag": part,
                "source_part": it["source"],
                "file": os.path.basename(filepath),
                "mask_file": mask_file_val,
                "mask_type": mask_output_mode if mask_file_val else None,
                "xyxy": bbox_xyxy,
                "xyxy_full": ([int(round(ox + x0 / sc)), int(round(oy + y0 / sc)),
                               int(round(ox + (x1 + 1) / sc)), int(round(oy + (y1 + 1) / sc))]
                              if transform else [x0, y0, x1 + 1, y1 + 1]),
                "z_order": order["z_order"].get(part, 0),
                "draw_index": (order["back_to_front"].index(part)
                               if part in order["back_to_front"] else None),
                "z_rank": core_ordering.prior_rank(part),
                "depth_median": self._depth_median(mask_t, legacy_depth),
                "passthrough": is_passthrough,
                "score": round(it["score"], 4),
                "prompt": it["prompt"],
                "instances": it["n_inst"],
                "area_px": int(it["mask"].sum()),
                "lr_source": it["lr_source"],
                "flags": it["flags"],
                "in_front_of": sorted(e["back"] for e in order["edges"] if e["front"] == part),
            }
            metadata_entries.append(entry)
            pbar.update(0)

        missing = [it["name"] for it in items if it["mask"] is None]
        for it in items:
            flag_txt = (" [%s]" % ", ".join(it["flags"])) if it["flags"] else ""
            if it["mask"] is None:
                report_lines.append("  SKIP  '%s' -- not detected%s" % (it["name"], flag_txt))
            else:
                report_lines.append("  OK    '%s' -- score=%.3f%s%s" % (
                    it["name"], it["score"],
                    (" lr=%s" % it["lr_source"]) if it["lr_source"] else "", flag_txt))

        if write_metadata_sidecar and metadata_entries:
            metadata_entries.sort(key=lambda e: e["z_order"])
            sidecar_path = os.path.join(output_dir, "parts_metadata.json")
            sidecar = {
                "schema_version": 2,
                "character": character_name,
                "pass": pass_name,
                "run_id": run_id or None,
                "image_size": [int(W_img), int(H_img)],
                "lr_convention": lr_convention,
                "character_facing": character_facing,
                "depth_near": depth_near,
                "order_method": order_method,
                "crop_transform": transform,
                "missing": missing,
                "parts": metadata_entries,
                "order_edges": [{"front": e["front"], "back": e["back"],
                                 "confidence": e["confidence"], "cues": e["cues"]}
                                for e in order["edges"]],
            }
            try:
                with open(sidecar_path, "w", encoding="utf-8") as f:
                    json.dump(sidecar, f, indent=2)
                report_lines += ["", "DRAW ORDER (front to back, %s):" % order_method]
                for e in metadata_entries:
                    report_lines.append("  %2d. %s" % (e["z_order"], e["tag"]))
            except Exception as e:
                print("[AutoSplit-SAM3] Failed to write metadata sidecar: %s" % e)
                sidecar_path = None
        else:
            sidecar_path = None
        pbar.update(1)

        report_lines += ["", "=" * 50, "SUMMARY:",
                         "  Saved: %d parts" % saved_count,
                         "  Missing: %d (%s)" % (len(missing), ", ".join(missing) or "-"),
                         "  Telea inpainted: %d" % inpainted_count,
                         "  Output: %s" % output_dir]
        report = "\n".join(report_lines)
        print("\n" + report)

        preview = self._build_preview_grid(preview_crops) if preview_crops else image
        return {"ui": {"autosplit": [{"metadata": sidecar_path, "folder": output_dir,
                                      "pass": pass_name, "run_id": run_id or None}]},
                "result": (report, preview)}


# ----------------------------------------------------------
# ComfyUI Node Registration
# ----------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "AutoSplitOrchestratorSAM3": AutoSplitOrchestratorSAM3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoSplitOrchestratorSAM3": "Auto-Split Orchestrator (SAM3)",
}
