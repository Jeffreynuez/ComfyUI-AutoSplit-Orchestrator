"""
2D Character Auto-Split: Orchestrator Node
===========================================

Custom ComfyUI node that loops through each body part sequentially:
  1. Detects the part with Florence-2
  2. Segments the part with SAM2
  3. Checks the depth map for occlusion (overlap-aware)
  4. If NOT occluded -> crop + pad + export
  5. If occluded -> expand mask, inpaint, crop + pad + export
  6. Moves to the next part and repeats

INSTALLATION:
  1. Copy this file to: ComfyUI/custom_nodes/auto_split_orchestrator.py
  2. Restart ComfyUI
  3. The node will appear under "2D Character Split" category

REQUIREMENTS (must be installed as custom nodes):
  - comfyui-florence2
  - ComfyUI-segment-anything-2
"""

import os
import json
import traceback


class AutoSplitOrchestrator:
    """
    Master orchestrator node for 2D character part splitting.

    Processes each body part one at a time:
      detect -> segment -> depth check -> (optional inpaint) -> crop -> save
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bg_removed_image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "florence2_model": ("FL2MODEL",),
                "sam2_model": ("SAM2MODEL",),
                "part_labels": ("STRING", {
                    "default": "head\nhair\ntorso\nleft arm\nright arm\nleft hand\nright hand\nleft leg\nright leg\nleft foot\nright foot",
                    "multiline": True,
                    "tooltip": "One body part per line. These are sent to Florence-2 one at a time."
                }),
                "character_name": ("STRING", {
                    "default": "character",
                }),
                "output_directory": ("STRING", {
                    "default": "split_parts",
                }),
                "padding": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 200,
                    "step": 5,
                    "display": "slider",
                }),
                "occlusion_threshold": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Depth difference to consider a region occluded"
                }),
                "mask_expand_pixels": ("INT", {
                    "default": 15,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "How many pixels to expand the mask when inpainting occluded areas"
                }),
                "enable_inpainting": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If True, occluded parts get inpainted. If False, exported as-is."
                }),
            },
            "optional": {
                "flux_inpaint_model": ("MODEL",),
                "flux_clip": ("CLIP",),
                "flux_vae": ("VAE",),
                "inpaint_steps": ("INT", {"default": 20, "min": 5, "max": 60, "step": 1}),
                "inpaint_guidance": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 30.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("report", "preview_grid",)
    FUNCTION = "run_pipeline"
    CATEGORY = "2D Character Split"
    OUTPUT_NODE = True

    # -------------------------------------------------
    # Prompt variations for better Florence-2 detection
    # -------------------------------------------------
    PROMPT_VARIATIONS = {
        'head': ['face', 'head', 'character face'],
        'face': ['face', 'head', 'character face'],
        'left hand': ['left hand', 'left wrist and hand', 'hand on the left'],
        'right hand': ['right hand', 'right wrist and hand', 'hand on the right'],
        'left foot': ['left foot', 'left ankle and foot', 'foot on the left'],
        'right foot': ['right foot', 'right ankle and foot', 'foot on the right'],
        'left leg': ['left leg', 'left thigh and leg'],
        'right leg': ['right leg', 'right thigh and leg'],
        'left arm': ['left arm', 'left sleeve and arm'],
        'right arm': ['right arm', 'right sleeve and arm'],
    }

    # Max allowed mask area as % of image — anything larger is a full-body misdetection
    MAX_PART_AREA_PCT = 20.0

    # -------------------------------------------------
    # Florence-2 detection for a single part
    # -------------------------------------------------
    def _detect_part(self, image_tensor, florence2_model, part_name):
        """
        Run Florence-2 caption_to_phrase_grounding for a single part.
        Uses the ORIGINAL image (with background) for best detection.
        Tries multiple prompt variations if the first one fails or returns
        only the full-character bbox.
        Returns: dict with bboxes/polygons, or None if not found.
        """
        import torch
        import torchvision.transforms.functional as TF
        import comfy.model_management as mm

        processor = florence2_model['processor']
        model = florence2_model['model']
        dtype = florence2_model['dtype']
        device = mm.get_torch_device()

        model.to(device)

        # Prepare image - ensure RGB (3 channels)
        img_permuted = image_tensor.permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W
        if img_permuted.shape[1] > 3:
            img_permuted = img_permuted[:, :3, :, :]
        image_pil = TF.to_pil_image(img_permuted[0])
        W, H = image_pil.size
        image_area = W * H

        print("[_detect_part] Image size: %dx%d" % (W, H))

        # Build list of prompts to try
        prompt_key = part_name.lower().strip()
        variations = self.PROMPT_VARIATIONS.get(prompt_key, [part_name])
        # Always include the original part_name if not already there
        if part_name not in variations:
            variations.append(part_name)

        # Try each prompt variation with caption_to_phrase_grounding
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        grounding_key = '<CAPTION_TO_PHRASE_GROUNDING>'

        for var_idx, prompt_text in enumerate(variations):
            prompt = task_prompt + " " + prompt_text

            print("[_detect_part] Attempt %d/%d: grounding for '%s' (prompt: '%s')" % (
                var_idx + 1, len(variations), part_name, prompt_text))

            inputs = processor(
                text=prompt, images=image_pil,
                return_tensors="pt", do_rescale=False
            ).to(dtype).to(device)

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
                use_cache=False,
            )

            results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            print("[_detect_part] Raw result: %s" % results[:300])

            parsed = processor.post_process_generation(
                results, task=task_prompt, image_size=(W, H)
            )

            if grounding_key not in parsed:
                continue

            data = parsed[grounding_key]
            bboxes = data.get('bboxes', []) if isinstance(data, dict) else []
            print("[_detect_part] Found %d raw bboxes: %s" % (len(bboxes), str(bboxes)))

            if not bboxes:
                continue

            # Score and filter bboxes
            scored = []
            for bbox in bboxes:
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                area = bw * bh
                area_pct = area / image_area * 100
                scored.append((area, area_pct, bbox))

            scored.sort(key=lambda x: x[0])  # smallest first

            # Pick smallest bbox between 0.5% and MAX_PART_AREA_PCT of image
            best_bbox = None
            for area, area_pct, bbox in scored:
                if area_pct >= 0.5 and area_pct < self.MAX_PART_AREA_PCT:
                    best_bbox = bbox
                    break

            if best_bbox is not None:
                sel_pct = (best_bbox[2]-best_bbox[0])*(best_bbox[3]-best_bbox[1])/image_area*100
                print("[_detect_part] ACCEPTED bbox (%.1f%% of image): %s" % (sel_pct, str(best_bbox)))
                return {
                    'bboxes': [best_bbox],
                    'labels': [part_name],
                    'width': W,
                    'height': H,
                }
            else:
                print("[_detect_part] All bboxes too large (smallest: %.1f%%), trying next prompt..." % scored[0][1])

        # FALLBACK: referring_expression_segmentation with original part_name
        task_prompt2 = '<REFERRING_EXPRESSION_SEGMENTATION>'
        for prompt_text in variations:
            prompt2 = task_prompt2 + " " + prompt_text

            print("[_detect_part] Fallback RES for '%s' (prompt: '%s')" % (part_name, prompt_text))

            inputs2 = processor(
                text=prompt2, images=image_pil,
                return_tensors="pt", do_rescale=False
            ).to(dtype).to(device)

            generated_ids2 = model.generate(
                input_ids=inputs2["input_ids"],
                pixel_values=inputs2["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
                use_cache=False,
            )

            results2 = processor.batch_decode(generated_ids2, skip_special_tokens=False)[0]
            print("[_detect_part] Fallback raw result: %s" % results2[:300])

            parsed2 = processor.post_process_generation(
                results2, task=task_prompt2, image_size=(W, H)
            )

            seg_key = '<REFERRING_EXPRESSION_SEGMENTATION>'
            if seg_key in parsed2 and parsed2[seg_key].get('polygons'):
                polygons = parsed2[seg_key]['polygons']
                print("[_detect_part] Found %d polygon groups via RES" % len(polygons))
                return {
                    'polygons': polygons,
                    'labels': parsed2[seg_key].get('labels', [part_name]),
                    'width': W,
                    'height': H,
                }

        print("[_detect_part] FAILED: No valid detection for '%s' after all attempts" % part_name)
        return None

    # -------------------------------------------------
    # SAM2 segmentation from detection results
    # -------------------------------------------------
    def _segment_part(self, image_tensor, sam2_model, detection):
        """
        Run SAM2 segmentation using bounding boxes or polygons from Florence-2.
        Returns: mask tensor (H, W) or None.
        """
        import torch
        import numpy as np
        import comfy.model_management as mm
        from contextlib import nullcontext

        model = sam2_model["model"]
        device = sam2_model["device"]
        dtype = sam2_model["dtype"]

        B, H, W, C = image_tensor.shape
        # SAM2 expects RGB (3 channels) - strip alpha if present
        if C > 3:
            image_rgb = image_tensor[:, :, :, :3]
        else:
            image_rgb = image_tensor
        image_np = (image_rgb.contiguous() * 255).byte().numpy()

        autocast_condition = not mm.is_device_mps(device)

        try:
            model.to(device)
        except Exception:
            model.model.to(device)

        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            model.set_image(image_np[0])

            # Use bboxes if available
            if 'bboxes' in detection and detection['bboxes']:
                bboxes = np.array(detection['bboxes'])
                out_masks, scores, logits = model.predict(
                    point_coords=None,
                    point_labels=None,
                    box=bboxes,
                    multimask_output=True,
                )
            elif 'polygons' in detection and detection['polygons']:
                # Convert polygons to bounding boxes for SAM2
                all_points = []
                for polygon_group in detection['polygons']:
                    for polygon in polygon_group:
                        points = np.array(polygon).reshape(-1, 2)
                        all_points.append(points)

                if all_points:
                    all_pts = np.concatenate(all_points, axis=0)
                    x_min, y_min = all_pts.min(axis=0)
                    x_max, y_max = all_pts.max(axis=0)
                    bbox = np.array([[x_min, y_min, x_max, y_max]])

                    out_masks, scores, logits = model.predict(
                        point_coords=None,
                        point_labels=None,
                        box=bbox,
                        multimask_output=True,
                    )
                else:
                    return None
            else:
                return None

            # Select best mask
            if out_masks.ndim == 3:
                sorted_ind = np.argsort(scores)[::-1]
                best_mask = out_masks[sorted_ind][0]
            elif out_masks.ndim == 4:
                # Multiple objects - combine
                combined = np.zeros((H, W), dtype=bool)
                for m in out_masks:
                    if m.ndim == 3:
                        sorted_ind = np.argsort(scores)[::-1]
                        combined = np.logical_or(combined, m[sorted_ind[0]])
                    else:
                        combined = np.logical_or(combined, m)
                best_mask = combined.astype(np.uint8)
            else:
                best_mask = out_masks

            mask_tensor = torch.from_numpy(best_mask.astype(np.float32))

            # Resize mask to original image dimensions if needed
            if mask_tensor.shape[-2:] != (H, W):
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                    size=(H, W), mode='bilinear', align_corners=False
                ).squeeze(0).squeeze(0)
                mask_tensor = (mask_tensor > 0.5).float()

            return mask_tensor

    # -------------------------------------------------
    # Overlap-aware occlusion detection
    # -------------------------------------------------
    def _check_occlusion(self, depth_map, current_mask, all_masks, current_idx, threshold):
        """
        For the current part, check if other parts' masks overlap AND
        have lower depth values (are in front of it).

        OPTIMIZED: Downsamples masks to 256x256 for fast comparison,
        then upscales the occlusion mask back to full resolution.

        Returns: (is_occluded, occlusion_mask, occlusion_pct)
        """
        import torch

        ANALYSIS_SIZE = 256  # Downsample to this for speed

        # Convert depth map to single-channel 2D tensor
        if depth_map.dim() == 4:
            depth = depth_map[0, :, :, 0]
        elif depth_map.dim() == 3:
            depth = depth_map[:, :, 0]
        else:
            depth = depth_map
        while depth.dim() > 2:
            depth = depth.squeeze(0)

        # Ensure current_mask is exactly 2D
        mask_2d = current_mask.clone()
        while mask_2d.dim() > 2:
            mask_2d = mask_2d.squeeze(0)
        while mask_2d.dim() < 2:
            mask_2d = mask_2d.unsqueeze(0)

        H, W = mask_2d.shape
        device = mask_2d.device

        # Downsample everything to ANALYSIS_SIZE for fast comparison
        def downsample(t, size=ANALYSIS_SIZE):
            return torch.nn.functional.interpolate(
                t.unsqueeze(0).unsqueeze(0).float(),
                size=(size, size), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)

        depth_small = downsample(depth.to(device))
        mask_small = downsample(mask_2d)

        print("[_check_occlusion] full res: %dx%d, analysis res: %dx%d" % (H, W, ANALYSIS_SIZE, ANALYSIS_SIZE))

        # Get depth values within current part (on small mask)
        active_small = mask_small > 0.5
        if active_small.sum() == 0:
            return False, torch.zeros_like(mask_2d), 0.0

        current_median_depth = depth_small[active_small].median()

        # Check overlap with every other part (all at low res)
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

            # Find overlap region
            overlap = (mask_small > 0.5) & (other_small > 0.5)
            if overlap.sum() == 0:
                continue

            # Check if the other part is in front (lower depth = closer)
            other_active = other_small > 0.5
            if other_active.sum() == 0:
                continue

            other_median_depth = depth_small[other_active].median()

            # If the other part is in front of the current part
            if other_median_depth < (current_median_depth - threshold):
                occlusion_small = torch.logical_or(occlusion_small.bool(), overlap).float()
                print("[_check_occlusion] Part %d overlaps and is in front (depth %.3f vs %.3f)" % (
                    i, other_median_depth.item(), current_median_depth.item()))

        occlusion_pct = 0.0
        if active_small.sum() > 0:
            occlusion_pct = (occlusion_small.sum().float() / active_small.sum().float() * 100).item()

        is_occluded = occlusion_pct > 2.0

        # Upscale occlusion mask back to full resolution only if occluded
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
    # Mask expansion for inpainting
    # -------------------------------------------------
    def _expand_mask(self, mask, expand_px):
        """Dilate mask by expand_px pixels."""
        import torch

        if expand_px <= 0:
            return mask

        kernel_size = expand_px * 2 + 1
        expanded = torch.nn.functional.max_pool2d(
            mask.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=kernel_size,
            stride=1,
            padding=expand_px
        ).squeeze()
        return (expanded > 0.5).float()

    # -------------------------------------------------
    # Crop + pad + save a single part
    # -------------------------------------------------
    def _crop_and_save(self, image_tensor, mask, part_name, character_name,
                       output_dir, padding):
        """
        Crop the masked region from the image, add padding, save as PNG with alpha.
        Returns: (filepath, crop_tensor) or (None, None) on failure.
        """
        import torch
        import numpy as np
        from PIL import Image

        # Work on CPU for saving
        img = image_tensor[0].cpu().numpy()  # (H, W, C)
        mask_np = mask.cpu().numpy()

        # Strip alpha channel if present — we only need RGB for the color data
        if img.shape[2] > 3:
            img = img[:, :, :3]

        # Resize image to match mask if needed
        mask_h, mask_w = mask_np.shape[:2]
        img_h, img_w = img.shape[:2]
        if img_h != mask_h or img_w != mask_w:
            pil_img = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
            pil_img = pil_img.resize((mask_w, mask_h), Image.LANCZOS)
            img = np.array(pil_img).astype(np.float32) / 255.0

        # Find bounding box
        ys, xs = np.where(mask_np > 0.5)
        if len(ys) == 0:
            return None, None

        y_min = max(0, int(ys.min()) - padding)
        y_max = min(mask_h - 1, int(ys.max()) + padding)
        x_min = max(0, int(xs.min()) - padding)
        x_max = min(mask_w - 1, int(xs.max()) + padding)

        # Crop
        img_crop = img[y_min:y_max+1, x_min:x_max+1, :]
        mask_crop = mask_np[y_min:y_max+1, x_min:x_max+1]

        # Create RGBA
        rgba = np.zeros((*img_crop.shape[:2], 4), dtype=np.float32)
        rgba[:, :, :3] = img_crop
        rgba[:, :, 3] = mask_crop

        # Convert to 8-bit and save
        rgba_8bit = (rgba * 255).clip(0, 255).astype(np.uint8)

        # Clean the part name for filename
        safe_name = part_name.replace(" ", "_").replace("/", "_")
        filename = "{0}_{1}.png".format(character_name, safe_name)
        filepath = os.path.join(output_dir, filename)

        pil_img = Image.fromarray(rgba_8bit, 'RGBA')
        pil_img.save(filepath)

        # Return crop as tensor for preview
        crop_tensor = torch.from_numpy(img_crop).unsqueeze(0)
        return filepath, crop_tensor

    # -------------------------------------------------
    # Simple inpainting (content-aware fill)
    # -------------------------------------------------
    def _simple_inpaint(self, image_tensor, mask, expand_px):
        """
        Simple content-aware fill for occluded regions.
        Uses OpenCV inpainting if available, otherwise a basic blur fill.
        """
        import torch
        import numpy as np
        from PIL import Image

        img_np = (image_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        mask_expanded = self._expand_mask(mask, expand_px)
        mask_np = (mask_expanded.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        try:
            import cv2
            inpaint_mask = mask_np.astype(np.uint8)
            result = cv2.inpaint(img_np, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
            return result_tensor
        except ImportError:
            from PIL import ImageFilter
            pil_img = Image.fromarray(img_np)
            blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=15))
            blurred_np = np.array(blurred).astype(np.float32) / 255.0
            img_float = img_np.astype(np.float32) / 255.0
            mask_3d = np.expand_dims(mask_np.astype(np.float32) / 255.0, -1)
            result = img_float * (1 - mask_3d) + blurred_np * mask_3d
            return torch.from_numpy(result).unsqueeze(0)

    # -------------------------------------------------
    # Main pipeline
    # -------------------------------------------------
    def run_pipeline(self, image, bg_removed_image, depth_map,
                     florence2_model, sam2_model, part_labels,
                     character_name, output_directory, padding,
                     occlusion_threshold, mask_expand_pixels,
                     enable_inpainting,
                     flux_inpaint_model=None, flux_clip=None,
                     flux_vae=None, inpaint_steps=20,
                     inpaint_guidance=30.0):
        """
        Main orchestration loop. Processes each part sequentially.
        """
        import torch
        import numpy as np
        import folder_paths
        from comfy.utils import ProgressBar

        # Setup output directory
        output_dir = os.path.join(folder_paths.get_output_directory(), output_directory)
        os.makedirs(output_dir, exist_ok=True)

        # Parse part labels
        parts = [p.strip() for p in part_labels.strip().split("\n") if p.strip()]
        if not parts:
            return ("No parts specified.", image)

        print("\n" + "=" * 60)
        print("[AutoSplitOrchestrator] Starting pipeline for '%s'" % character_name)
        print("[AutoSplitOrchestrator] Parts to process: %s" % str(parts))
        print("[AutoSplitOrchestrator] Output: %s" % output_dir)
        print("=" * 60 + "\n")

        report_lines = [
            "Auto-Split Report for '%s'" % character_name,
            "=" * 50,
            "Parts requested: %d" % len(parts),
            "",
        ]

        # Phase 1: Detect and segment ALL parts first (needed for overlap analysis)
        all_masks = []
        all_detections = []
        pbar = ProgressBar(len(parts) * 2)  # detect + save phases

        print("[AutoSplitOrchestrator] Phase 1: Detecting and segmenting all parts...")
        for i, part in enumerate(parts):
            print("\n--- Detecting part %d/%d: '%s' ---" % (i + 1, len(parts), part))

            try:
                # Use ORIGINAL image (with background) for detection - Florence-2 needs context
                detection = self._detect_part(image, florence2_model, part)
            except Exception as e:
                print("[AutoSplitOrchestrator] Detection failed for '%s': %s" % (part, e))
                traceback.print_exc()
                detection = None

            if detection is None:
                print("[AutoSplitOrchestrator] Could not detect '%s' -- skipping" % part)
                all_masks.append(None)
                all_detections.append(None)
                report_lines.append("  SKIP  '%s' -- not detected by Florence-2" % part)
                pbar.update(1)
                continue

            try:
                # Use ORIGINAL image for SAM2 too - it needs RGB context, not RGBA bg-removed
                mask = self._segment_part(image, sam2_model, detection)
            except Exception as e:
                print("[AutoSplitOrchestrator] Segmentation failed for '%s': %s" % (part, e))
                traceback.print_exc()
                mask = None

            if mask is None:
                print("[AutoSplitOrchestrator] Could not segment '%s' -- skipping" % part)
                all_masks.append(None)
                all_detections.append(detection)
                report_lines.append("  SKIP  '%s' -- SAM2 segmentation failed" % part)
            else:
                nonzero = (mask > 0.5).sum().item()
                B, H_img, W_img, C = image.shape
                total_pixels = H_img * W_img
                mask_pct = nonzero / total_pixels * 100

                print("[AutoSplitOrchestrator] '%s' segmented: %d pixels (%.1f%% of image)" % (
                    part, nonzero, mask_pct))

                # Reject masks that are too large — likely a full-body misdetection
                if mask_pct > self.MAX_PART_AREA_PCT:
                    print("[AutoSplitOrchestrator] WARNING: '%s' mask covers %.1f%% of image (max %.1f%%) -- rejected as full-body" % (
                        part, mask_pct, self.MAX_PART_AREA_PCT))
                    all_masks.append(None)
                    all_detections.append(detection)
                    report_lines.append("  SKIP  '%s' -- mask too large (%.1f%%), likely full-body detection" % (part, mask_pct))
                else:
                    all_masks.append(mask)
                    all_detections.append(detection)

            pbar.update(1)

        # Phase 2: Occlusion analysis + crop + save
        print("\n[AutoSplitOrchestrator] Phase 2: Occlusion analysis + export...")
        saved_count = 0
        inpainted_count = 0
        skipped_count = 0
        preview_crops = []

        for i, part in enumerate(parts):
            mask = all_masks[i]
            if mask is None:
                skipped_count += 1
                pbar.update(1)
                print("[Phase 2] Skipping '%s' (no mask)" % part)
                continue

            # Ensure mask is 2D for all downstream operations
            while mask.dim() > 2:
                mask = mask.squeeze(0)

            print("\n--- Phase 2: part %d/%d: '%s' (mask pixels: %d) ---" % (
                i + 1, len(parts), part, (mask > 0.5).sum().item()))

            # Occlusion check
            print("[Phase 2] Running occlusion check for '%s'..." % part)
            is_occluded, occlusion_mask, occlusion_pct = self._check_occlusion(
                depth_map, mask, all_masks, i, occlusion_threshold
            )
            print("[Phase 2] Occlusion check done for '%s'" % part)

            if is_occluded:
                status_str = "occluded %.1f%%" % occlusion_pct
            else:
                status_str = "clear"
            print("[AutoSplitOrchestrator] '%s': %s" % (part, status_str))

            # Determine which image to use for this part
            work_image = bg_removed_image

            if is_occluded and enable_inpainting:
                print("[AutoSplitOrchestrator] '%s': inpainting occluded regions..." % part)
                try:
                    inpaint_mask = occlusion_mask
                    work_image = self._simple_inpaint(
                        bg_removed_image, inpaint_mask, mask_expand_pixels
                    )
                    inpainted_count += 1
                    report_lines.append(
                        "  INPAINT  '%s' -- %.1f%% occluded, inpainted" % (part, occlusion_pct)
                    )
                except Exception as e:
                    print("[AutoSplitOrchestrator] Inpainting failed for '%s': %s" % (part, e))
                    traceback.print_exc()
                    report_lines.append(
                        "  WARN  '%s' -- inpainting failed, exported as-is" % part
                    )
            elif is_occluded:
                report_lines.append(
                    "  WARN  '%s' -- %.1f%% occluded (inpainting disabled)" % (part, occlusion_pct)
                )
            else:
                report_lines.append("  OK  '%s' -- clear, no occlusion" % part)

            # Crop + save
            try:
                filepath, crop_tensor = self._crop_and_save(
                    work_image, mask, part, character_name, output_dir, padding
                )
                if filepath:
                    saved_count += 1
                    print("[AutoSplitOrchestrator] Saved: %s" % filepath)
                    if crop_tensor is not None:
                        preview_crops.append(crop_tensor)
                else:
                    skipped_count += 1
                    report_lines.append("  SKIP  '%s' -- empty mask after crop" % part)
            except Exception as e:
                print("[AutoSplitOrchestrator] Save failed for '%s': %s" % (part, e))
                traceback.print_exc()
                skipped_count += 1

            pbar.update(1)

        # Summary
        report_lines.append("")
        report_lines.append("=" * 50)
        report_lines.append("SUMMARY:")
        report_lines.append("  Saved: %d parts" % saved_count)
        report_lines.append("  Inpainted: %d parts" % inpainted_count)
        report_lines.append("  Skipped: %d parts" % skipped_count)
        report_lines.append("  Output: %s" % output_dir)

        report = "\n".join(report_lines)
        print("\n" + report)

        # Build preview grid (or return original image if no crops)
        if preview_crops:
            preview = self._build_preview_grid(preview_crops)
        else:
            preview = image

        return (report, preview,)

    def _build_preview_grid(self, crops):
        """Arrange cropped parts into a grid for preview."""
        import torch

        # Find max dimensions
        max_h = max(c.shape[1] for c in crops)
        max_w = max(c.shape[2] for c in crops)

        # Pad all crops to same size
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

        # Arrange in grid
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


# ----------------------------------------------------------
# ComfyUI Node Registration
# ----------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "AutoSplitOrchestrator": AutoSplitOrchestrator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoSplitOrchestrator": "Auto-Split Orchestrator",
}
