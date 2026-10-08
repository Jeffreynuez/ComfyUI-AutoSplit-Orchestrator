"""
Plan the hidden-area fill for a split, with no per-character prompts.

Reads a split (body pass + facial pass), decides the peels with
autosplit_core.peel (base body, covered garments, head, back-hair rule) and
writes, into --out:

  jobs.json        what each peel removes, its prompt, model and targets
  <job>.json       one ComfyUI API workflow per peel: Flux.2 Klein edit of
                   the picture -> scaled back to the picture's size -> saved
                   -> Depth Anything V2 -> SAM 3 orchestrator on the labels
                   the merge needs
  plan.json        the tools/peel_merge.py plan pointing at those outputs

Run the workflows (ComfyUI, or tools/run_fill.py which also merges), then:

    python tools/peel_merge.py --run <body> --run <face> --source <picture> \
        --plan <out>/plan.json --out <filled body pass>

    python tools/peel_plan.py --run <body pass> --run <facial pass> \
        --source <picture file> --comfy-image <picture name in ComfyUI input> \
        --comfy-output <ComfyUI output folder> --subdir autosplit_fill/<name> \
        --out <folder for the workflows> [--base-layer "a light green sports bra and briefs"]
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ComfyUI-AutoSplit-Orchestrator"))
from autosplit_core import peel  # noqa: E402
from klein_workflows import edit_full  # noqa: E402
from partsio import back_to_front, load_pass  # noqa: E402

DEFAULT_TEMPLATE = os.path.join(ROOT, "2D_Character_AutoSplit_SAM3.json")


def _template_nodes(path):
    """The SAM 3 loader, depth and body orchestrator settings from the
    pipeline's own workflow, so a peel is split exactly like the picture."""
    with open(path, encoding="utf-8") as f:
        wf = json.load(f)
    loader = depth = orch = None
    for node in wf.values():
        ct = node.get("class_type")
        if ct == "easy sam3ModelLoader" and loader is None:
            loader = node
        elif ct == "DepthAnythingV2Preprocessor" and depth is None:
            depth = node
        elif ct == "AutoSplitOrchestratorSAM3" and "crop_transform" not in node["inputs"] and orch is None:
            orch = node
    if not (loader and depth and orch):
        raise SystemExit("template %s lacks the SAM 3 loader, depth or body orchestrator" % path)
    return loader, depth, orch


def peel_workflow(job, comfy_image, subdir, template):
    loader, depth, orch = template
    prefix = "%s/%s" % (subdir, job["name"])
    wf = edit_full(comfy_image, job["prompt"], prefix + "_edit", size=job.get("model", "4b"))
    wf["40"] = {"class_type": "GetImageSize", "inputs": {"image": ["1", 0]}}
    wf["41"] = {"class_type": "ImageScale", "inputs": {"image": ["14", 0], "upscale_method": "lanczos",
                                                       "width": ["40", 0], "height": ["40", 1],
                                                       "crop": "disabled"}}
    wf["42"] = {"class_type": "SaveImage", "inputs": {"images": ["41", 0], "filename_prefix": prefix + "_up"}}
    d = copy.deepcopy(depth)
    d["inputs"]["image"] = ["41", 0]
    wf["43"] = d
    wf["44"] = copy.deepcopy(loader)
    o = copy.deepcopy(orch)
    o["inputs"].update({
        "image": ["41", 0], "depth_map": ["43", 0], "sam3_model": ["44", 0],
        "part_labels": "\n".join(job["sam_labels"]), "character_name": job["name"],
        "output_directory": subdir, "run_id": job["name"], "cluster_hair_front_back": False,
        "auto_lr_split_parts": "", "passthrough_parts": "", "output_mode": "alpha_mask",
        "mask_output_mode": "none", "write_metadata_sidecar": True, "enable_inpainting": False,
    })
    wf["45"] = o
    for n in wf.values():
        n.pop("_meta", None)
    return wf


def build(runs, source, comfy_image, comfy_output, subdir, out, template=DEFAULT_TEMPLATE,
          base_layer=peel.BASE_LAYER, model="auto", verbose=True):
    """Write the peel workflows, jobs.json and plan.json into `out`.
    Returns (workflow paths in run order, plan path)."""
    rgb = np.asarray(Image.open(source).convert("RGB"))
    H, W = rgb.shape[:2]
    parts = []
    for f in runs:
        parts += load_pass(f)
    order = back_to_front(parts)
    btf = [(p.tag, p.full_mask(W, H)) for p in order]
    facial = {p.tag for p in parts if p.facial}
    jobs = peel.plan(btf, rgb, facial=facial, base_layer=base_layer)

    tmpl = _template_nodes(template)
    os.makedirs(out, exist_ok=True)
    plan, summary, workflows = [], [], []
    out_base = os.path.join(comfy_output, *subdir.split("/"))
    for job in jobs:
        rule = next((k for k in ("hull", "underlap", "back_panel", "skin_under") if k in job), None)
        if rule:
            plan.append(job)
            summary.append({"rule": {rule: job[rule]}})
            continue
        if model != "auto":
            job["model"] = model
        wf = peel_workflow(job, comfy_image, subdir, tmpl)
        wf_path = os.path.join(out, job["name"] + ".json")
        with open(wf_path, "w", encoding="utf-8") as f:
            json.dump(wf, f, indent=1)
        workflows.append(wf_path)
        step = {"image": os.path.join(out_base, job["name"] + "_up_00001_.png"),
                "run": os.path.join(out_base, job["name"]),
                "parts": job["parts"], "job": job["name"]}
        for k in ("anchors", "colour", "exclude", "claim_unowned"):
            if k in job:
                step[k] = job[k]
        plan.append(step)
        summary.append({k: job[k] for k in ("name", "kind", "model", "targets", "removes", "prompt",
                                            "sam_labels") if k in job})
    plan_path = os.path.join(out, "plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=1)
    with open(os.path.join(out, "jobs.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    if verbose:
        for s in summary:
            if "rule" in s:
                print("rule   %s" % json.dumps(s["rule"]))
            else:
                print("%-22s %s  targets %s\n    %s" % (s["name"], s["model"], ", ".join(s["targets"]),
                                                        s["prompt"]))
        print("\n%d workflows + plan.json -> %s" % (len(workflows), out))
    return workflows, plan_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="body pass, then facial pass")
    ap.add_argument("--source", required=True, help="the picture that was split (file)")
    ap.add_argument("--comfy-image", required=True, help="the same picture's name in ComfyUI input")
    ap.add_argument("--comfy-output", required=True, help="ComfyUI output folder (for the plan's paths)")
    ap.add_argument("--subdir", default="autosplit_fill/run", help="output subfolder for this plan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--base-layer", default=peel.BASE_LAYER,
                    help="what the base body wears (the rig's convention); default: %(default)s")
    ap.add_argument("--model", choices=["auto", "4b", "9b"], default="auto")
    args = ap.parse_args()
    build(args.run, args.source, args.comfy_image, args.comfy_output, args.subdir, args.out,
          args.template, args.base_layer, args.model)


if __name__ == "__main__":
    main()
