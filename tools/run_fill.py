"""
Fill the hidden areas of a split, end to end, on a running ComfyUI.

    python tools/run_fill.py --run <body pass> --run <facial pass> \
        --source <the picture that was split> [--name salena] \
        [--base-layer "a light green sports bra and briefs"] [--model auto|4b|9b]

1. plans the peels from the split (tools/peel_plan.py, no per-character
   prompts), 2. uploads the picture and runs one ComfyUI workflow per peel
   (Flux.2 Klein edit -> scale back -> SAM 3 split), 3. merges the peeled parts
   into a copy of the body pass (tools/peel_merge.py) and prints where it is.

Uses the same environment variables as the rest of the repo
(AUTOSPLIT_COMFY_URL, AUTOSPLIT_COMFY_ROOT / AUTOSPLIT_COMFY_OUTPUT).
Standard library + numpy, Pillow, OpenCV.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "krita_plugin", "autosplit_studio"))
import comfy_client  # noqa: E402
import peel_plan  # noqa: E402
sys.path.insert(0, ROOT)
from comfy_autosplit_test import COMFY_OUTPUT_DIR, COMFY_URL  # noqa: E402


def log(m):
    print("[fill] " + str(m), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="body pass, then facial pass")
    ap.add_argument("--source", required=True)
    ap.add_argument("--name", default=None, help="label for this fill (default: a timestamp)")
    ap.add_argument("--base-layer", default=None,
                    help="what the base body wears (default: a simple plain sports top and briefs)")
    ap.add_argument("--model", choices=["auto", "4b", "9b"], default="auto")
    ap.add_argument("--out", default=None, help="filled body pass folder (default: next to the body pass)")
    ap.add_argument("--plan-only", action="store_true", help="write the workflows and stop")
    args = ap.parse_args()

    name = args.name or time.strftime("%Y%m%d-%H%M%S")
    subdir = "autosplit_fill/" + name
    work = os.path.join(COMFY_OUTPUT_DIR, *subdir.split("/"), "_workflows")
    body = os.path.normpath(args.run[0])
    out = args.out or body + "_filled"

    if not args.plan_only:
        comfy_client.check_server(COMFY_URL)
        ref = comfy_client.upload_image(COMFY_URL, args.source)
        log("uploaded %s" % ref)
    else:
        ref = os.path.basename(args.source)
    workflows, plan_path = peel_plan.build(args.run, args.source, ref, COMFY_OUTPUT_DIR, subdir, work,
                                           base_layer=args.base_layer or peel_plan.peel.BASE_LAYER,
                                           model=args.model)
    if args.plan_only:
        return
    for wf_path in workflows:
        with open(wf_path, encoding="utf-8") as f:
            wf = json.load(f)
        comfy_client.queue_and_wait(COMFY_URL, wf, progress=log,
                                    label=os.path.splitext(os.path.basename(wf_path))[0])
    cmd = [sys.executable, os.path.join(HERE, "peel_merge.py"), "--source", args.source,
           "--plan", plan_path, "--out", out]
    for r in args.run:
        cmd += ["--run", r]
    log("merging -> %s" % out)
    subprocess.run(cmd, check=True)
    log("done: filled body pass in %s (facial pass unchanged)" % out)


if __name__ == "__main__":
    main()
