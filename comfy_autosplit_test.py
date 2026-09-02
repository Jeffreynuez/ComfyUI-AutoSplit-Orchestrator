"""
2D AutoSplit Studio - Phase 1, Increment 1
ComfyUI API driver (standalone test). Run this from a normal terminal.

GOAL: prove we can drive your SAM3 AutoSplit workflow from code -
  upload a character image -> run the workflow in ComfyUI -> produce the split parts.
Once this works, the exact same logic gets wrapped in a Krita plugin.

SETUP (one time):
  1. In ComfyUI: Settings (gear) -> turn ON "Enable Dev mode Options".
  2. Open your 2D_Character_AutoSplit_SAM3 workflow -> Workflow menu ->
     "Export (API)" / "Save (API Format)". Save the .json anywhere inside this
     folder (any filename - it's auto-detected).
  3. Make sure ComfyUI is running (port 8188).

RUN:
  pip install requests          (one time, if needed)
  python comfy_autosplit_test.py
"""

import json
import os
import sys
import time
import uuid

try:
    import requests
except ImportError:
    print("Missing dependency. Run:  pip install requests")
    sys.exit(1)

# ----------------------------- CONFIG -----------------------------
# Every path below can be overridden with an environment variable, so this
# repo runs on any machine without editing source. Defaults assume a standard
# ComfyUI-Easy-Install layout next to this project.
#
#   AUTOSPLIT_COMFY_URL     ComfyUI server            (default http://127.0.0.1:8188)
#   AUTOSPLIT_COMFY_ROOT    ComfyUI install root      (used to find output/)
#   AUTOSPLIT_COMFY_OUTPUT  ComfyUI output dir        (overrides COMFY_ROOT)
#   AUTOSPLIT_INPUT         character image to split
#   AUTOSPLIT_OUT           where Spine/part output is written
COMFY_URL = os.environ.get("AUTOSPLIT_COMFY_URL", "http://127.0.0.1:8188")

# Folder this script lives in. Put your exported API-format workflow JSON here
# (any filename - it's auto-detected).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# The character image to split:
INPUT_IMAGE = os.environ.get(
    "AUTOSPLIT_INPUT", os.path.join(SCRIPT_DIR, "input", "character.png"))

# Where the SAM3 orchestrator writes parts (ComfyUI output dir):
COMFY_ROOT = os.environ.get(
    "AUTOSPLIT_COMFY_ROOT",
    os.path.join(os.path.dirname(SCRIPT_DIR), "ComfyUI-Easy-Install", "ComfyUI"))
COMFY_OUTPUT_DIR = os.environ.get(
    "AUTOSPLIT_COMFY_OUTPUT", os.path.join(COMFY_ROOT, "output"))
PART_SUBDIRS = ["split_parts_sam3", "split_parts_sam3_facial"]
# ------------------------------------------------------------------

CLIENT_ID = str(uuid.uuid4())


def log(msg):
    print("[autosplit] " + str(msg), flush=True)


def is_api_workflow(data):
    """True if 'data' looks like a ComfyUI API-format prompt graph."""
    return (isinstance(data, dict) and len(data) > 0
            and all(isinstance(v, dict) and "class_type" in v for v in data.values()))


def find_api_workflow():
    """Find an exported API-format workflow JSON in this folder (any filename).
    Prefers one that contains the SAM3 AutoSplit orchestrator."""
    found = []
    for fn in os.listdir(SCRIPT_DIR):
        if not fn.lower().endswith(".json"):
            continue
        path = os.path.join(SCRIPT_DIR, fn)
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if is_api_workflow(data):
            has_autosplit = any(v.get("class_type") == "AutoSplitOrchestratorSAM3"
                                for v in data.values())
            found.append((0 if has_autosplit else 1, fn, path))
    if not found:
        return None
    found.sort()  # autosplit-containing files first
    return found[0][2]


def upload_image(path):
    fname = os.path.basename(path)
    with open(path, "rb") as f:
        files = {"image": (fname, f, "image/png")}
        data = {"overwrite": "true"}
        r = requests.post(COMFY_URL + "/upload/image", files=files, data=data, timeout=120)
    r.raise_for_status()
    info = r.json()
    name = info.get("name", fname)
    sub = info.get("subfolder", "")
    ref = (sub + "/" + name) if sub else name
    log("uploaded '%s' -> ComfyUI input as '%s'" % (path, ref))
    return ref


def inject_input_image(api_wf, image_ref):
    count = 0
    for _node_id, node in api_wf.items():
        if isinstance(node, dict) and node.get("class_type") == "LoadImage":
            node.setdefault("inputs", {})["image"] = image_ref
            count += 1
    log("set input image on %d LoadImage node(s)" % count)
    if count == 0:
        log("WARNING: no LoadImage node found in the workflow.")
    return api_wf


def queue_prompt(api_wf):
    payload = {"prompt": api_wf, "client_id": CLIENT_ID}
    r = requests.post(COMFY_URL + "/prompt", json=payload, timeout=120)
    if r.status_code != 200:
        log("ComfyUI rejected the prompt (HTTP %d):" % r.status_code)
        log(r.text)
        r.raise_for_status()
    pid = r.json()["prompt_id"]
    log("queued prompt_id = %s" % pid)
    return pid


def wait_for_completion(pid, timeout_s=1200):
    log("waiting for the workflow to finish (first run can take a while)...")
    start = time.time()
    while time.time() - start < timeout_s:
        r = requests.get(COMFY_URL + "/history/" + pid, timeout=30)
        if r.status_code == 200:
            hist = r.json()
            if pid in hist:
                entry = hist[pid]
                status = entry.get("status", {})
                if status.get("completed") or entry.get("outputs"):
                    log("workflow completed in %.0fs." % (time.time() - start))
                    return entry
                if status.get("status_str") == "error":
                    log("workflow reported an ERROR:")
                    log(json.dumps(status, indent=2)[:2000])
                    raise RuntimeError("ComfyUI workflow errored - see ComfyUI console.")
        time.sleep(2)
    raise TimeoutError("Workflow did not complete within %ds." % timeout_s)


def report_parts():
    log("--- results ---")
    total = 0
    for sub in PART_SUBDIRS:
        d = os.path.join(COMFY_OUTPUT_DIR, sub)
        if not os.path.isdir(d):
            log("%-24s (folder not found)" % sub)
            continue
        pngs = [f for f in os.listdir(d) if f.lower().endswith(".png")
                and not f.lower().endswith("_mask.png")]
        total += len(pngs)
        log("%-24s %d part PNG(s)" % (sub, len(pngs)))
        meta = os.path.join(d, "parts_metadata.json")
        if os.path.exists(meta):
            try:
                data = json.load(open(meta, encoding="utf-8"))
                parts = data if isinstance(data, list) else data.get("parts", [])
                names = [p.get("tag", "?") for p in parts][:40]
                log("    metadata: %d entries -> %s" % (len(parts), ", ".join(names)))
            except Exception as e:
                log("    (could not read parts_metadata.json: %s)" % e)
    log("TOTAL parts produced: %d" % total)


def sanitize_workflow(api_wf):
    """Make the workflow robust for headless/API runs.

    The orchestrator's Florence-2 fallback is OPTIONAL and SAM3 doesn't need it.
    But if the Florence-2 model isn't installed, its loader fails ComfyUI's
    prompt validation - and because both orchestrators take it as an input, that
    failure cascades and silently invalidates the whole split (0 parts). So we
    drop the Florence-2 loader node(s) and any links to them, leaving a clean
    SAM3-only graph that validates regardless of Florence-2's state."""
    drop = [nid for nid, n in api_wf.items()
            if isinstance(n, dict) and n.get("class_type") == "Florence2ModelLoader"]
    for nid in drop:
        api_wf.pop(nid, None)
    if drop:
        log("dropped optional Florence-2 loader node(s) %s -> SAM3-only" % ", ".join(drop))
    remaining = set(api_wf.keys())
    for n in api_wf.values():
        inputs = n.get("inputs", {}) if isinstance(n, dict) else {}
        for name, val in list(inputs.items()):
            if isinstance(val, list) and len(val) == 2 and str(val[0]) not in remaining:
                inputs.pop(name, None)
    return api_wf


def main():
    workflow_api = find_api_workflow()
    if not workflow_api:
        log("ERROR: no API-format workflow JSON found in:\n  %s" % SCRIPT_DIR)
        log("Export it from ComfyUI: Settings -> Enable Dev mode Options -> 'Export (API)'.")
        sys.exit(1)
    if not os.path.exists(INPUT_IMAGE):
        log("ERROR: input image not found:\n  %s" % INPUT_IMAGE)
        sys.exit(1)

    log("ComfyUI: " + COMFY_URL)
    try:
        requests.get(COMFY_URL + "/system_stats", timeout=10)
    except Exception:
        log("ERROR: can't reach ComfyUI at %s - is it running on port 8188?" % COMFY_URL)
        sys.exit(1)

    log("using workflow: %s" % os.path.basename(workflow_api))
    api_wf = json.load(open(workflow_api, encoding="utf-8"))
    api_wf = sanitize_workflow(api_wf)
    ref = upload_image(INPUT_IMAGE)
    inject_input_image(api_wf, ref)
    pid = queue_prompt(api_wf)
    wait_for_completion(pid)
    report_parts()
    log("done. If you see parts above, the ComfyUI integration works - "
        "next we wrap this in Krita.")


if __name__ == "__main__":
    main()
