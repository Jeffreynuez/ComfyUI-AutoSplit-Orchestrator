"""
2D AutoSplit Studio - standalone driver.

Runs the SAM 3 AutoSplit workflow from a terminal, without Krita, through the
same client the Krita plugin uses (krita_plugin/autosplit_studio/comfy_client.py,
pure standard library). Useful for debugging and for scripted batches.

    python comfy_autosplit_test.py [image.png] [--workflow wf.json] [--run-id ID]

Each run writes into <output_directory>/<run id>/ for both passes, prints the
parts it produced and the folders, and never reads an older run's files.

Paths come from the environment (see README "Configuration"):
  AUTOSPLIT_COMFY_URL     ComfyUI server            (default http://127.0.0.1:8188)
  AUTOSPLIT_COMFY_ROOT    ComfyUI install root      (used to find output/)
  AUTOSPLIT_COMFY_OUTPUT  ComfyUI output dir        (overrides COMFY_ROOT)
  AUTOSPLIT_INPUT         character image to split  (or pass it as an argument)
"""
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "krita_plugin", "autosplit_studio"))
import comfy_client  # noqa: E402  (stdlib only; no Krita needed)


def find_comfy_root():
    env = os.environ.get("AUTOSPLIT_COMFY_ROOT")
    if env:
        return env
    base = os.path.dirname(SCRIPT_DIR)
    for cand in (os.path.join(base, "AI Work", "ComfyUI-Easy-Install", "ComfyUI"),
                 os.path.join(base, "ComfyUI-Easy-Install", "ComfyUI")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(base, "AI Work", "ComfyUI-Easy-Install", "ComfyUI")


COMFY_URL = os.environ.get("AUTOSPLIT_COMFY_URL", "http://127.0.0.1:8188")
COMFY_OUTPUT_DIR = os.environ.get("AUTOSPLIT_COMFY_OUTPUT",
                                  os.path.join(find_comfy_root(), "output"))


def log(msg):
    print("[autosplit] " + str(msg), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?",
                    default=os.environ.get("AUTOSPLIT_INPUT",
                                           os.path.join(SCRIPT_DIR, "input", "character.png")))
    ap.add_argument("--workflow", default=os.path.join(SCRIPT_DIR, "2D_Character_AutoSplit_SAM3.json"))
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.image):
        log("ERROR: input image not found: %s" % args.image)
        sys.exit(1)
    try:
        comfy_client.check_server(COMFY_URL)
    except Exception as e:
        log("ERROR: %s" % e)
        sys.exit(1)

    log("ComfyUI %s, workflow %s" % (COMFY_URL, os.path.basename(args.workflow)))
    res = comfy_client.run_split(COMFY_URL, args.workflow, args.image, progress=log,
                                 run_id=args.run_id)
    entries = comfy_client.read_parts(COMFY_OUTPUT_DIR, passes=res["passes"])
    log("run %s: %d parts" % (res["run_id"], len(entries)))
    for p in res["passes"]:
        log("  %-6s %s" % (p.get("pass"), p.get("folder")))
    order = [e.get("tag") for e in comfy_client.draw_order(entries)]
    log("draw order (back -> front): " + " | ".join(order))
    flagged = [(e.get("tag"), e.get("flags")) for e in entries if e.get("flags")]
    if flagged:
        log("flags: " + json.dumps(flagged))


if __name__ == "__main__":
    main()
