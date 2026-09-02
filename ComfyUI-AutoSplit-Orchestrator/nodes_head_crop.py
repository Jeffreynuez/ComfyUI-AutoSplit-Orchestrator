"""
HeadCropUpscale Node
====================
Auto-detects a target region (head/face) via SAM3, crops it with padding,
and upscales the result so a second orchestrator can segment facial features
(eyes, nose, mouth, iris, eyebrows) at much higher resolution.

Also crops + scales the depth map to match, so the downstream orchestrator
gets consistent depth data without needing a second Marigold pass.

Usage:
  LoadImage ──┬──> HeadCropUpscale ──> 2nd Orchestrator (facial parts only)
              └──> Main Orchestrator (body parts)
"""

import numpy as np
import torch


class HeadCropUpscale:
    """
    Two-pass helper: detects a region (default: head) via SAM3,
    crops with configurable padding, and upscales both the image
    and depth map. Feed the outputs into a second orchestrator
    configured with facial-only parts for high-res segmentation.
    """

    CROP_PROMPT_VARIATIONS = {
        'head': ['head', 'face', 'head and face'],
        'face': ['face', 'head'],
        'upper body': ['upper body', 'torso and head'],
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "sam3_model": ("EASY_SAM3_MODEL",),
                "crop_target": ("STRING", {
                    "default": "head",
                    "placeholder": "Region to detect and crop (e.g. head)",
                    "tooltip": "SAM3 text prompt for the region to crop. 'head' works best for the facial detail pass."
                }),
                "scale_factor": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.5,
                    "tooltip": "Upscale multiplier for the cropped region. 2.0 = double resolution, giving SAM3 4x the pixels to work with."
                }),
                "padding_pct": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "tooltip": "Extra padding around the detected region as a fraction of crop size. 0.25 = 25% extra on each side. Helps capture hair and ears."
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.30,
                    "min": 0.05,
                    "max": 0.95,
                    "step": 0.05,
                    "tooltip": "SAM3 confidence threshold for detecting the crop target region."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("cropped_image", "cropped_depth",)
    FUNCTION = "crop_and_upscale"
    CATEGORY = "2D Character Split"

    def _detect_region(self, image_tensor, sam3_model, target, confidence):
        """
        Run SAM3 text-grounded segmentation to find the target region.
        Tries multiple prompt variations for robustness.
        Returns: (mask_2d_numpy, score) or (None, 0.0)
        """
        from PIL import Image as PILImage

        processor = sam3_model['processor']

        B, H, W, C = image_tensor.shape
        img_rgb = image_tensor[:, :, :, :3] if C > 3 else image_tensor
        img_np = (img_rgb[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img_np)

        image_area = H * W

        # Build prompt list
        target_lower = target.lower().strip()
        variations = list(self.CROP_PROMPT_VARIATIONS.get(target_lower, [target]))
        if target not in variations:
            variations.append(target)

        best_mask = None
        best_score = 0.0

        for prompt_text in variations:
            try:
                print("[HeadCrop] Trying prompt: '%s'" % prompt_text)
                state = processor.set_image(pil_img)
                processor.reset_all_prompts(state)
                processor.set_confidence_threshold(confidence, state)
                state = processor.set_text_prompt(prompt_text, state)

                masks = state.get('masks')
                logits = state.get('masks_logits')

                if masks is None or (hasattr(masks, 'numel') and masks.numel() == 0):
                    print("[HeadCrop] No masks for '%s'" % prompt_text)
                    continue

                masks_f = masks.float()
                if masks_f.ndim == 4:
                    masks_f = masks_f.squeeze(1)

                scores = None
                if logits is not None:
                    logits_f = logits.float()
                    if logits_f.ndim == 4:
                        logits_f = logits_f.squeeze(1)
                    scores = logits_f.mean(dim=(-2, -1))

                for i in range(masks_f.shape[0]):
                    m = masks_f[i]
                    score = scores[i].item() if scores is not None else 1.0
                    m_np = m.squeeze().cpu().numpy()
                    m_binary = (m_np > 0.5).astype(np.float32)
                    nonzero = m_binary.sum()
                    mask_pct = nonzero / image_area * 100

                    print("[HeadCrop]   Mask %d: score=%.3f, pixels=%d (%.1f%%)" % (
                        i, score, int(nonzero), mask_pct))

                    # Skip noise (<0.01%) and full-body misdetections (>30%)
                    if mask_pct < 0.01 or mask_pct > 30.0:
                        continue

                    if score > best_score:
                        best_score = score
                        if m_binary.shape != (H, W):
                            m_t = torch.from_numpy(m_binary).unsqueeze(0).unsqueeze(0)
                            m_t = torch.nn.functional.interpolate(
                                m_t, size=(H, W), mode='bilinear', align_corners=False
                            ).squeeze()
                            best_mask = (m_t > 0.5).float().numpy()
                        else:
                            best_mask = m_binary

            except Exception as e:
                print("[HeadCrop] Error with prompt '%s': %s" % (prompt_text, e))
                continue

        return best_mask, best_score

    def _write_transform(self, x_min, y_min, scale):
        """Record the crop offset + upscale factor so downstream tools can map
        facial-pass part coordinates back to the full image:
            full_x = x_min + facial_x / scale  (and the part is downscaled by 1/scale)."""
        try:
            import os
            import json
            import folder_paths
            out = folder_paths.get_output_directory()
            path = os.path.join(out, "head_crop_transform.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"x_min": int(x_min), "y_min": int(y_min),
                           "scale": float(scale)}, f)
            print("[HeadCrop] transform sidecar -> x=%d y=%d scale=%.2f"
                  % (int(x_min), int(y_min), float(scale)))
        except Exception as e:
            print("[HeadCrop] could not write transform sidecar: %s" % e)

    def crop_and_upscale(self, image, depth_map, sam3_model, crop_target="head",
                         scale_factor=2.0, padding_pct=0.25, confidence_threshold=0.30):
        """
        Main entry point:
        1. Detect target region via SAM3
        2. Compute padded bounding box
        3. Crop image + depth map
        4. Upscale both
        5. Return as IMAGE tensors
        """
        B, H, W, C = image.shape

        print("=" * 60)
        print("[HeadCrop] Detecting '%s' for facial detail crop..." % crop_target)

        mask, score = self._detect_region(image, sam3_model, crop_target, confidence_threshold)

        if mask is None:
            print("[HeadCrop] WARNING: Could not detect '%s' — returning full image unchanged" % crop_target)
            print("[HeadCrop] The facial detail orchestrator will run on the full image.")
            self._write_transform(0, 0, 1.0)
            return (image, depth_map)

        print("[HeadCrop] ACCEPTED '%s' (score=%.3f)" % (crop_target, score))

        # ---- Bounding box with padding ----
        ys, xs = np.where(mask > 0.5)
        y_min_raw, y_max_raw = int(ys.min()), int(ys.max())
        x_min_raw, x_max_raw = int(xs.min()), int(xs.max())

        crop_h = y_max_raw - y_min_raw
        crop_w = x_max_raw - x_min_raw
        pad_h = int(crop_h * padding_pct)
        pad_w = int(crop_w * padding_pct)

        y_min = max(0, y_min_raw - pad_h)
        y_max = min(H - 1, y_max_raw + pad_h)
        x_min = max(0, x_min_raw - pad_w)
        x_max = min(W - 1, x_max_raw + pad_w)

        final_h = y_max - y_min + 1
        final_w = x_max - x_min + 1
        pct_of_image = final_h * final_w / (H * W) * 100

        print("[HeadCrop] Crop box: x=[%d,%d] y=[%d,%d]  %dx%d (%.1f%% of image)" % (
            x_min, x_max, y_min, y_max, final_w, final_h, pct_of_image))

        # ---- Crop image ----
        img_crop = image[:, y_min:y_max + 1, x_min:x_max + 1, :]

        # ---- Crop depth map (may be different resolution) ----
        dB, dH, dW, dC = depth_map.shape
        if dH != H or dW != W:
            sy, sx = dH / H, dW / W
            dy_min = max(0, int(y_min * sy))
            dy_max = min(dH - 1, int(y_max * sy))
            dx_min = max(0, int(x_min * sx))
            dx_max = min(dW - 1, int(x_max * sx))
            depth_crop = depth_map[:, dy_min:dy_max + 1, dx_min:dx_max + 1, :]
        else:
            depth_crop = depth_map[:, y_min:y_max + 1, x_min:x_max + 1, :]

        # ---- Upscale ----
        if scale_factor > 1.0:
            new_h = int(img_crop.shape[1] * scale_factor)
            new_w = int(img_crop.shape[2] * scale_factor)

            # Image: [B,H,W,C] -> [B,C,H,W] for interpolate, then back
            img_bchw = img_crop.permute(0, 3, 1, 2).float()
            img_scaled = torch.nn.functional.interpolate(
                img_bchw, size=(new_h, new_w), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)

            depth_bchw = depth_crop.permute(0, 3, 1, 2).float()
            depth_scaled = torch.nn.functional.interpolate(
                depth_bchw, size=(new_h, new_w), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)

            print("[HeadCrop] Upscaled %dx%d -> %dx%d (%.1fx)" % (
                final_w, final_h, new_w, new_h, scale_factor))
        else:
            img_scaled = img_crop
            depth_scaled = depth_crop

        print("[HeadCrop] Done — output image: %dx%d" % (img_scaled.shape[2], img_scaled.shape[1]))
        print("=" * 60)

        eff_scale = scale_factor if scale_factor > 1.0 else 1.0
        self._write_transform(x_min, y_min, eff_scale)
        return (img_scaled, depth_scaled)


NODE_CLASS_MAPPINGS = {
    "HeadCropUpscale": HeadCropUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeadCropUpscale": "Head Crop + Upscale (Facial Detail)",
}
