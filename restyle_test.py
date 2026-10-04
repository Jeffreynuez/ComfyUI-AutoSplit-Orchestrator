"""
2D AutoSplit Studio - Phase 3: img2img restyle (standalone test).

Restyles a full character via an anime SDXL/Illustrious model (img2img): changes
outfit/colours per your prompt while keeping pose, proportions, and your flat-2D
style (because img2img modifies the existing image rather than inventing one).
Proves the restyle before we wire it into the plugin and feed it back into the
splitter to make a new Spine skin.

RUN:
  pip install requests          (once, if needed)
  python restyle_test.py "change the outfit to a black and gold dress"

(no prompt arg -> uses a default). The restyled PNG path is printed; open it to
check the look. Tune DENOISE / CHECKPOINT_HINT below.
"""
import os
import sys
import time
import uuid

try:
    import requests
except ImportError:
    print("Missing dependency:  pip install requests")
    sys.exit(1)

# ------------------------------ CONFIG ------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMFY_URL = os.environ.get("AUTOSPLIT_COMFY_URL", "http://127.0.0.1:8188")
INPUT_IMAGE = os.environ.get(
    "AUTOSPLIT_INPUT", os.path.join(SCRIPT_DIR, "input", "character.png"))
COMFY_ROOT = os.environ.get(
    "AUTOSPLIT_COMFY_ROOT",
    os.path.join(os.path.dirname(SCRIPT_DIR), "ComfyUI-Easy-Install", "ComfyUI"))
OUTPUT_DIR = os.environ.get(
    "AUTOSPLIT_COMFY_OUTPUT", os.path.join(COMFY_ROOT, "output"))
CHECKPOINT_HINT = "novaAnimeXL"   # substring used to find your anime checkpoint
DENOISE = 0.6      # 0.5 subtle, 0.75 bolder (Illustrious img2img sweet range 0.6-0.8)
STEPS = 28         # Illustrious: 20-30
CFG = 5.0          # Illustrious sweet spot ~4.5-5
SAMPLER = "euler_ancestral"   # "Euler a" - recommended for Illustrious / NovaAnimeXL
SCHEDULER = "normal"
CLIP_SKIP = -2     # Illustrious models require clip skip 2
SIZE = 1024        # SDXL works best near 1MP; Salena is ~square
QUALITY = "masterpiece, best quality, very aesthetic, absurdres, "
NEG = ("worst quality, low quality, blurry, bad anatomy, bad hands, extra limbs, "
       "deformed, watermark, signature, text, photorealistic, 3d render")
# --------------------------------------------------------------------

CLIENT = uuid.uuid4().hex


def log(m):
    print("[restyle] " + str(m), flush=True)


def get(url, t=30):
    with requests.get(url, timeout=t) as r:
        r.raise_for_status()
        return r.json()


def post(url, obj, t=180):
    r = requests.post(url, json=obj, timeout=t)
    if r.status_code != 200:
        raise RuntimeError("ComfyUI HTTP %d: %s" % (r.status_code, r.text[:1200]))
    return r.json()


def resolve_checkpoint():
    info = get(COMFY_URL + "/object_info/CheckpointLoaderSimple")
    opts = info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    for o in opts:
        if CHECKPOINT_HINT.lower() in o.lower():
            return o
    raise RuntimeError("No checkpoint matching '%s'. Some available: %s"
                       % (CHECKPOINT_HINT, opts[:12]))


def upload(path):
    name = os.path.basename(path)
    with open(path, "rb") as f:
        r = requests.post(COMFY_URL + "/upload/image",
                          files={"image": (name, f, "image/png")},
                          data={"overwrite": "true"}, timeout=180)
    r.raise_for_status()
    info = r.json()
    sub, n = info.get("subfolder", ""), info.get("name", name)
    return (sub + "/" + n) if sub else n


def build_graph(ckpt, image_ref, prompt, seed):
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "12": {"class_type": "CLIPSetLastLayer",
               "inputs": {"clip": ["4", 1], "stop_at_clip_layer": CLIP_SKIP}},
        "10": {"class_type": "LoadImage", "inputs": {"image": image_ref}},
        "11": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": ["10", 0], "upscale_method": "lanczos",
            "megapixels": round((SIZE * SIZE) / 1e6, 2), "resolution_steps": 8}},
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


def main():
    user_prompt = (sys.argv[1] if len(sys.argv) > 1 else
                   "black and gold dress outfit, same pose and proportions")
    prompt = QUALITY + user_prompt
    try:
        get(COMFY_URL + "/system_stats", 10)
    except Exception:
        log("Can't reach ComfyUI at %s - is it running on 8188?" % COMFY_URL)
        return
    if not os.path.exists(INPUT_IMAGE):
        log("Input image not found: %s" % INPUT_IMAGE)
        return

    ckpt = resolve_checkpoint()
    log("model: %s" % ckpt)
    log("prompt: %s" % prompt)
    ref = upload(INPUT_IMAGE)
    log("uploaded %s" % os.path.basename(INPUT_IMAGE))
    seed = uuid.uuid4().int % (2 ** 31)
    graph = build_graph(ckpt, ref, prompt, seed)
    pid = post(COMFY_URL + "/prompt", {"prompt": graph, "client_id": CLIENT})["prompt_id"]
    log("queued %s; restyling (denoise %.2f)..." % (pid[:8], DENOISE))

    start = time.time()
    while time.time() - start < 600:
        h = get(COMFY_URL + "/history/" + pid)
        if pid in h:
            entry = h[pid]
            imgs = entry.get("outputs", {}).get("9", {}).get("images", [])
            if imgs:
                im = imgs[0]
                sub, fn = im.get("subfolder", ""), im["filename"]
                p = os.path.join(OUTPUT_DIR, sub, fn) if sub else os.path.join(OUTPUT_DIR, fn)
                log("done in %.0fs. Restyled image:\n  %s" % (time.time() - start, p))
                return
            if entry.get("status", {}).get("status_str") == "error":
                log("ComfyUI reported an error - check the ComfyUI console.")
                return
        time.sleep(1.5)
    log("timed out after 600s.")


if __name__ == "__main__":
    main()
