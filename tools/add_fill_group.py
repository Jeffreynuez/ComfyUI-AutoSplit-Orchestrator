"""
Add the hidden-area fill to a saved ComfyUI workflow (UI format), as its own
group above the existing ones, wired to the split that is already there.

    python tools/add_fill_group.py <workflow.json> [--out <new.json>] [--bypass]

It finds the body orchestrator (Auto-Split Orchestrator SAM3 fed by the
picture) and the facial one (fed by Head Crop + Upscale), then adds:

  Auto-Split Fill (Klein peels)   image, depth map and SAM 3 model taken from
                                  the body orchestrator's inputs, its report
                                  (and the facial one's) wired in
  Preview Image                   the filled split pulled apart + the edits
  Show Any (easy-use)             the fill report, when that node pack is there

in a group "Hidden-Area Fill (Klein peels + rig rules)". Right-click the
group title (or use a group muter) to bypass it. --bypass saves it bypassed.
Without --out the file is rewritten after a .bak copy is made.
"""
import argparse
import json
import os
import shutil
import sys

FILL = "AutoSplitFillKlein"
GROUP_TITLE = "Hidden-Area Fill (Klein peels + rig rules) - right-click title to Bypass/Enable"

WIDGETS = [  # in INPUT_TYPES order (widgets only)
    "Flux2\\flux-2-klein-4b-fp8.safetensors",
    "qwen_3_4b_fp8_mixed.safetensors",
    "Flux2\\flux-2-klein-9b-kv-fp8.safetensors",
    "qwen_3_8b_fp8mixed.safetensors",
    "Flux2\\flux2-vae.safetensors",
    "a simple plain sports top and simple plain briefs",
    7,      # edit_seed
    4,      # steps
    1.5,    # megapixels
]


def add_fill_group(wf, bypass=False):
    nodes = {n["id"]: n for n in wf["nodes"]}
    links = {l[0]: l for l in wf["links"]}
    if any(n["type"] == FILL for n in wf["nodes"]):
        raise SystemExit("this workflow already has an Auto-Split Fill node")

    def origin(node, name):
        inp = next((i for i in node.get("inputs", []) if i["name"] == name), None)
        lk = links.get(inp.get("link")) if inp else None
        return (lk[1], lk[2], lk[5]) if lk else None

    orchs = [n for n in wf["nodes"] if n["type"] == "AutoSplitOrchestratorSAM3"]
    body = face = None
    for o in orchs:
        src = origin(o, "image")
        if src and nodes[src[0]]["type"] == "HeadCropUpscale":
            face = face or o
        elif src:
            body = body or o
    if body is None:
        raise SystemExit("no body Auto-Split Orchestrator (SAM3) fed by a picture in this workflow")
    need = {k: origin(body, k) for k in ("image", "depth_map", "sam3_model")}
    if not all(need.values()):
        raise SystemExit("the body orchestrator's image / depth_map / sam3_model are not all connected")

    next_node = wf.get("last_node_id", max(nodes)) + 1
    next_link = wf.get("last_link_id", max(links) if links else 0) + 1

    def new_link(o_id, o_slot, t_id, t_slot, typ):
        nonlocal next_link
        lid = next_link
        next_link += 1
        wf["links"].append([lid, o_id, o_slot, t_id, t_slot, typ])
        out = nodes[o_id]["outputs"][o_slot]
        out["links"] = (out.get("links") or []) + [lid]
        return lid

    groups = wf.setdefault("groups", [])
    top = min([g["bounding"][1] for g in groups] + [min(n["pos"][1] for n in wf["nodes"])])
    x = body["pos"][0] - 100
    y = top - 900
    fid, pid, rid = next_node, next_node + 1, next_node + 2
    mode = 4 if bypass else 0
    fill = {
        "id": fid, "type": FILL, "pos": [x, y], "size": [440, 420], "flags": {}, "order": 99,
        "mode": mode, "title": "Auto-Split Fill (Klein peels)",
        "inputs": [
            {"name": "image", "type": "IMAGE", "link": None},
            {"name": "depth_map", "type": "IMAGE", "link": None},
            {"name": "sam3_model", "type": "EASY_SAM3_MODEL", "link": None},
            {"name": "body_report", "type": "STRING", "link": None},
            {"name": "face_report", "type": "STRING", "shape": 7, "link": None},
        ],
        "outputs": [{"name": "report", "type": "STRING", "links": [], "slot_index": 0},
                    {"name": "preview", "type": "IMAGE", "links": [], "slot_index": 1}],
        "properties": {"Node name for S&R": FILL},
        "widgets_values": list(WIDGETS),
    }
    preview = {
        "id": pid, "type": "PreviewImage", "pos": [x + 470, y], "size": [640, 420], "flags": {},
        "order": 100, "mode": mode, "title": "Filled split (pulled apart) + edits",
        "inputs": [{"name": "images", "type": "IMAGE", "link": None}], "outputs": [],
        "properties": {"Node name for S&R": "PreviewImage"}, "widgets_values": [],
    }
    wf["nodes"] += [fill, preview]
    nodes[fid], nodes[pid] = fill, preview
    fill["inputs"][0]["link"] = new_link(need["image"][0], need["image"][1], fid, 0, "IMAGE")
    fill["inputs"][1]["link"] = new_link(need["depth_map"][0], need["depth_map"][1], fid, 1, "IMAGE")
    fill["inputs"][2]["link"] = new_link(need["sam3_model"][0], need["sam3_model"][1], fid, 2,
                                         "EASY_SAM3_MODEL")
    fill["inputs"][3]["link"] = new_link(body["id"], 0, fid, 3, "STRING")
    if face is not None:
        fill["inputs"][4]["link"] = new_link(face["id"], 0, fid, 4, "STRING")
    preview["inputs"][0]["link"] = new_link(fid, 1, pid, 0, "IMAGE")
    if any(n["type"] == "easy showAnything" for n in wf["nodes"]):
        report = {
            "id": rid, "type": "easy showAnything", "pos": [x, y + 450], "size": [1110, 300],
            "flags": {}, "order": 101, "mode": mode, "title": "Fill Report",
            "inputs": [{"name": "anything", "type": "*", "shape": 7, "link": None}],
            "outputs": [{"name": "output", "type": "*", "links": None}],
            "properties": {"Node name for S&R": "easy showAnything"}, "widgets_values": [""],
        }
        wf["nodes"].append(report)
        nodes[rid] = report
        report["inputs"][0]["link"] = new_link(fid, 0, rid, 0, "STRING")
        next_node += 3
    else:
        next_node += 2
    wf["last_node_id"] = next_node - 1
    wf["last_link_id"] = next_link - 1
    gid = max([g.get("id", 0) for g in groups] + [0]) + 1
    groups.append({"id": gid, "title": GROUP_TITLE, "bounding": [x - 40, y - 80, 1190, 880],
                   "color": "#3f7e6a", "font_size": 24, "flags": {}})
    return fid


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workflow")
    ap.add_argument("--out")
    ap.add_argument("--bypass", action="store_true", help="save the new group bypassed")
    args = ap.parse_args()
    with open(args.workflow, encoding="utf-8") as f:
        wf = json.load(f)
    if "nodes" not in wf:
        sys.exit("not a UI-format workflow (no 'nodes'); API-format files have no groups")
    fid = add_fill_group(wf, args.bypass)
    out = args.out or args.workflow
    if out == args.workflow:
        shutil.copy2(args.workflow, args.workflow + ".bak")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=2)
    print("added node %d (Auto-Split Fill) in its own group -> %s" % (fid, os.path.abspath(out)))


if __name__ == "__main__":
    main()
