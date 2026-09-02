"""
Skin generation (img2img restyle) for the AutoSplit Studio Krita plugin.

Same validated logic / settings as the standalone restyle_test.py, ported to use
the plugin's urllib-based comfy client. Restyles the full character via an anime
Illustrious model, returning the path of the restyled PNG in ComfyUI's output.
"""
import os
import time
import uuid

from . import comfy_client

# ------------------------------ settings ------------------------------
CHECKPOINT_HINT = "novaAnimeXL"   # substring to find your anime checkpoint
DENOISE = 0.6      # 0.5 keeps more of the original, 0.75 changes more
STEPS = 28
CFG = 5.0
SAMPLER = "euler_ancestral"   # Euler a
SCHEDULER = "normal"
CLIP_SKIP = -2     # Illustrious models need clip skip 2
SIZE = 1024
QUALITY = "masterpiece, best quality, very aesthetic, absurdres, "
NEG = ("worst quality, low quality, blurry, bad anatomy, bad hands, extra limbs, "
       "deformed, watermark, signature, text, photorealistic, 3d render")
# ----------------------------------------------------------------------


def resolve_checkpoint(comfy_url, hint=CHECKPOINT_HINT):
    info = comfy_client._get_json(comfy_url + "/object_info/CheckpointLoaderSimple")
    opts = info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    for o in opts:
        if hint.lower() in o.lower():
            return o
    raise RuntimeError("No checkpoint matching '%s'. Some available: %s"
                       % (hint, opts[:12]))


def _graph(ckpt, image_ref, prompt, seed):
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "12": {"class_type": "CLIPSetLastLayer",
               "inputs": {"clip": ["4", 1], "stop_at_clip_layer": CLIP_SKIP}},
        "10": {"class_type": "LoadImage", "inputs": {"image": image_ref}},
        "11": {"class_type": "ImageScale", "inputs": {
            "image": ["10", 0], "upscale_method": "lanczos",
            "width": SIZE, "height": SIZE, "crop": "disabled"}},
        "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["4", 2]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["12", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["12", 0]}},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": STEPS, "cfg": CFG,
            "sampler_name": SAMPLER, "scheduler": SCHEDULER, "denoise": DENOISE,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "restyle"}},
    }


def run_restyle(comfy_url, image_path, user_prompt, output_dir, progress=None, timeout_s=600):
    """Restyle the character; return the restyled PNG path in ComfyUI's output."""
    def say(m):
        if progress:
            progress(m)

    ckpt = resolve_checkpoint(comfy_url)
    say("model: %s" % ckpt)
    ref = comfy_client.upload_image(comfy_url, image_path)
    say("uploaded %s" % os.path.basename(image_path))
    prompt = QUALITY + (user_prompt.strip() if user_prompt and user_prompt.strip()
                        else "same pose and proportions")
    seed = uuid.uuid4().int % (2 ** 31)
    graph = _graph(ckpt, ref, prompt, seed)
    res = comfy_client._post_json(comfy_url + "/prompt",
                                  {"prompt": graph, "client_id": uuid.uuid4().hex})
    pid = res["prompt_id"]
    say("queued %s; restyling (denoise %.2f)..." % (pid[:8], DENOISE))

    start = time.time()
    last = 0
    while time.time() - start < timeout_s:
        h = comfy_client._get_json(comfy_url + "/history/" + pid)
        if pid in h:
            entry = h[pid]
            imgs = entry.get("outputs", {}).get("9", {}).get("images", [])
            if imgs:
                im = imgs[0]
                sub, fn = im.get("subfolder", ""), im["filename"]
                p = os.path.join(output_dir, sub, fn) if sub else os.path.join(output_dir, fn)
                say("restyle done in %.0fs" % (time.time() - start))
                return p
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError("ComfyUI restyle errored - see ComfyUI console.")
        elapsed = time.time() - start
        if elapsed - last >= 3:
            last = elapsed
            say("...restyling (%.0fs)" % elapsed)
        time.sleep(1.0)
    raise TimeoutError("Restyle timed out after %ds." % timeout_s)
