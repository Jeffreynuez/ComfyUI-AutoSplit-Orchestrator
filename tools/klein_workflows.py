"""
Flux.2 Klein image-edit workflows (ComfyUI API format) for the hidden-area
fill prototype ("peeling"): the same character with some front layers taken
away, in the same pose.

    python tools/klein_workflows.py full  out.json '{"image": "...", "prompt": "...", "prefix": "..."}'
    python tools/klein_workflows.py masked out.json '{"image": "...", "mask_image": "...", ...}'

  full    edits the whole picture (ReferenceLatent on the input, 4 steps,
          cfg 1). Used for every peel so far.
  masked  edits only inside a mask (Pixaroma crop + stitch), for local fixes.

Models (Jeffrey's install): Flux.2 Klein 4B fp8 + Qwen 3 4B, or 9B kv-fp8 +
Qwen 3 8B, with the Flux.2 VAE. 4B is the default; 9B followed the jacket
instruction better but drifted more in style.
"""
import json
import sys

MODELS = {
    "4b": ("Flux2\\flux-2-klein-4b-fp8.safetensors", "qwen_3_4b_fp8_mixed.safetensors"),
    "9b": ("Flux2\\flux-2-klein-9b-kv-fp8.safetensors", "qwen_3_8b_fp8mixed.safetensors"),
}
VAE = "Flux2\\flux2-vae.safetensors"


def _loaders(size):
    unet, te = MODELS[size]
    return {
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "flux2", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    }


def edit_full(image, prompt, prefix, size="4b", mp=1.5, seed=7, steps=4):
    wf = _loaders(size)
    wf.update({
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["1", 0], "upscale_method": "lanczos",
                                                                   "megapixels": mp, "resolution_steps": 16}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": prompt}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["5", 0]}},
        "8": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["6", 0], "latent": ["7", 0]}},
        "9": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "10": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["9", 0], "latent": ["7", 0]}},
        "11": {"class_type": "GetImageSize", "inputs": {"image": ["2", 0]}},
        "12": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["11", 0], "height": ["11", 1], "batch_size": 1}},
        "13": {"class_type": "KSampler", "inputs": {"model": ["3", 0], "positive": ["8", 0], "negative": ["10", 0],
                                                    "latent_image": ["12", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                                                    "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["5", 0]}},
        "15": {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix}},
    })
    return wf


def edit_masked(image, mask_image, prompt, prefix, size="4b", target=1024, context_px=48, seed=7,
                steps=4, grow=4, blur=4, softness=12):
    wf = _loaders(size)
    wf.update({
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "20": {"class_type": "LoadImageMask", "inputs": {"image": mask_image, "channel": "red"}},
        "21": {"class_type": "PixaromaInpaintCrop", "inputs": {
            "image": ["1", 0], "mask": ["20", 0], "size_mode": "keep shape (long side)", "target": target,
            "multiple": 16, "context_px": context_px, "mask_grow": grow, "mask_blur": blur,
            "softness": softness, "blend_mode": "mask", "invert_mask": False}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": prompt}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": ["5", 0]}},
        "8": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["6", 0], "latent": ["7", 0]}},
        "9": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "10": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["9", 0], "latent": ["7", 0]}},
        "12": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["21", 3], "height": ["21", 4], "batch_size": 1}},
        "13": {"class_type": "KSampler", "inputs": {"model": ["3", 0], "positive": ["8", 0], "negative": ["10", 0],
                                                    "latent_image": ["12", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                                                    "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["5", 0]}},
        "22": {"class_type": "PixaromaInpaintStitch", "inputs": {"image": ["14", 0], "crop_info": ["21", 2], "softness": -1,
                                                                 "blend_mode": "from crop", "color_match": "off"}},
        "15": {"class_type": "SaveImage", "inputs": {"images": ["22", 0], "filename_prefix": prefix}},
    })
    return wf


if __name__ == "__main__":
    kind, out = sys.argv[1], sys.argv[2]
    kwargs = json.loads(sys.argv[3])
    wf = edit_full(**kwargs) if kind == "full" else edit_masked(**kwargs)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=1)
    print("wrote", out)
