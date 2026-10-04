"""
ComfyUI client for the AutoSplit Studio Krita plugin.

Pure standard-library (urllib) so it runs in Krita's bundled Python with no
extra packages. The standalone comfy_autosplit_test.py drives ComfyUI through
this same module.

  check the server -> upload image -> strip optional Florence-2 -> force alpha
  -> stamp a run id on every AutoSplit node -> queue -> wait -> read the run's
  own output folders (never whatever an older run left on disk)
"""
import json
import os
import time
import uuid
import urllib.request
import urllib.error


def _post_json(url, obj, timeout=120):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError("ComfyUI HTTP %d: %s" % (e.code, body[:1200]))


def _get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ping(comfy_url, timeout=8):
    """Raise if ComfyUI isn't reachable."""
    _get_json(comfy_url + "/system_stats", timeout=timeout)


def check_server(comfy_url, timeout=8):
    """Raise a clear error unless this ComfyUI has the AutoSplit nodes.

    Krita AI Diffusion can start its own managed ComfyUI on the same port;
    that server answers /system_stats but has no SAM 3 or AutoSplit nodes, and
    a split sent there fails with a confusing validation error."""
    ping(comfy_url, timeout)
    try:
        info = _get_json(comfy_url + "/object_info/AutoSplitOrchestratorSAM3", timeout=timeout)
    except Exception:
        info = {}
    if "AutoSplitOrchestratorSAM3" not in info:
        raise RuntimeError(
            "The ComfyUI at %s has no AutoSplit node. Another ComfyUI (for example "
            "Krita AI Diffusion's own server) may be holding the port. Start the "
            "Easy-Install ComfyUI, or point AUTOSPLIT_COMFY_URL at it." % comfy_url)


def new_run_id():
    """Sortable, unique per split: 20261004-153012-a1b2."""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def upload_image(comfy_url, path):
    """Multipart upload into ComfyUI's input folder. Returns the LoadImage ref."""
    boundary = "----autosplit" + uuid.uuid4().hex
    fname = os.path.basename(path)
    with open(path, "rb") as f:
        filedata = f.read()
    parts = []
    parts.append(("--%s\r\n" % boundary).encode())
    parts.append(('Content-Disposition: form-data; name="image"; filename="%s"\r\n'
                  % fname).encode())
    parts.append(b"Content-Type: image/png\r\n\r\n")
    parts.append(filedata)
    parts.append(b"\r\n")
    parts.append(("--%s\r\n" % boundary).encode())
    parts.append(b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n')
    parts.append(("--%s--\r\n" % boundary).encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        comfy_url + "/upload/image", data=body,
        headers={"Content-Type": "multipart/form-data; boundary=%s" % boundary})
    with urllib.request.urlopen(req, timeout=180) as r:
        info = json.loads(r.read().decode("utf-8"))
    name = info.get("name", fname)
    sub = info.get("subfolder", "")
    return (sub + "/" + name) if sub else name


def sanitize_workflow(wf):
    """Drop the optional Florence-2 loader (and dangling links) so the graph is
    SAM3-only and validates even when Florence-2 isn't installed."""
    drop = [k for k, n in wf.items()
            if isinstance(n, dict) and n.get("class_type") == "Florence2ModelLoader"]
    for k in drop:
        wf.pop(k, None)
    remaining = set(wf.keys())
    for n in wf.values():
        if not isinstance(n, dict):
            continue
        inp = n.get("inputs", {})
        for name, val in list(inp.items()):
            if isinstance(val, list) and len(val) == 2 and str(val[0]) not in remaining:
                inp.pop(name, None)
    return wf, len(drop)


def force_alpha_output(wf):
    """Force every AutoSplit orchestrator to save RGBA alpha cutouts (transparent
    background) instead of parts composited on solid white - required for rigging.
    Returns how many orchestrators were changed."""
    n = 0
    for node in wf.values():
        if isinstance(node, dict) and node.get("class_type") == "AutoSplitOrchestratorSAM3":
            node.setdefault("inputs", {})["output_mode"] = "alpha_mask"
            n += 1
    return n


def inject_input_image(wf, image_ref):
    count = 0
    for n in wf.values():
        if isinstance(n, dict) and n.get("class_type") == "LoadImage":
            n.setdefault("inputs", {})["image"] = image_ref
            count += 1
    return count


def stamp_run_id(wf, run_id):
    """Give every orchestrator and head-crop node the run id, so each split
    writes into <output_directory>/<run_id>/. The head crop writes its
    transform next to the facial pass that consumes it."""
    orch = {k: n for k, n in wf.items()
            if isinstance(n, dict) and n.get("class_type") == "AutoSplitOrchestratorSAM3"}
    for n in orch.values():
        n.setdefault("inputs", {})["run_id"] = run_id
    for k, n in wf.items():
        if isinstance(n, dict) and n.get("class_type") == "HeadCropUpscale":
            n.setdefault("inputs", {})["run_id"] = run_id
            for o in orch.values():
                src = o.get("inputs", {}).get("image")
                if isinstance(src, list) and str(src[0]) == str(k):
                    n["inputs"]["output_directory"] = o["inputs"].get("output_directory", "")
    return [n["inputs"].get("output_directory", "") for n in orch.values()]


def run_split(comfy_url, workflow_api_path, image_path, progress=None, timeout_s=1200,
              run_id=None):
    """Upload image, run the SAM3 AutoSplit workflow, wait for completion.
    progress(msg) is an optional callback for status updates.

    Returns {"prompt_id", "run_id", "passes": [{"folder", "metadata", "pass"}]}.
    "folder" is relative to ComfyUI's output directory when the node did not
    report an absolute one."""
    def say(m):
        if progress:
            progress(m)

    wf = json.load(open(workflow_api_path, encoding="utf-8"))
    wf, dropped = sanitize_workflow(wf)
    if dropped:
        say("dropped optional Florence-2 loader -> SAM3-only")
    n_alpha = force_alpha_output(wf)
    if n_alpha:
        say("set %d orchestrator(s) to alpha cutout output" % n_alpha)
    run_id = run_id or new_run_id()
    out_dirs = stamp_run_id(wf, run_id)
    ref = upload_image(comfy_url, image_path)
    say("uploaded %s" % os.path.basename(image_path))
    n = inject_input_image(wf, ref)
    if n == 0:
        raise RuntimeError("No LoadImage node in the workflow - is it API format?")

    client_id = uuid.uuid4().hex
    res = _post_json(comfy_url + "/prompt", {"prompt": wf, "client_id": client_id})
    pid = res["prompt_id"]
    say("queued (%s, run %s); running split..." % (pid[:8], run_id))

    start = time.time()
    last_tick = 0
    while time.time() - start < timeout_s:
        hist = _get_json(comfy_url + "/history/" + pid)
        if pid in hist:
            entry = hist[pid]
            st = entry.get("status", {})
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                detail = msgs[0][1].get("exception_message", "") if msgs else ""
                raise RuntimeError("ComfyUI workflow errored: %s" % (detail or "see ComfyUI console"))
            if st.get("completed") or entry.get("outputs"):
                say("split finished in %.0fs" % (time.time() - start))
                passes = []
                for node_out in entry.get("outputs", {}).values():
                    for p in node_out.get("autosplit", []) or []:
                        passes.append(p)
                if not passes:  # older node: work the folders out ourselves
                    passes = [{"folder": os.path.join(d, run_id), "metadata": None,
                               "pass": "facial" if "facial" in d.lower() else "body"}
                              for d in out_dirs]
                return {"prompt_id": pid, "run_id": run_id, "passes": passes}
        elapsed = time.time() - start
        if elapsed - last_tick >= 3:
            last_tick = elapsed
            say("...running (%.0fs)" % elapsed)
        time.sleep(1.0)
    raise TimeoutError("Split timed out after %ds." % timeout_s)


def _pass_folders(output_dir, subdirs=None, passes=None):
    if passes:
        out = []
        for p in passes:
            folder = p.get("folder") or ""
            if not os.path.isabs(folder):
                folder = os.path.join(output_dir, folder)
            out.append((folder, p.get("pass")))
        return out
    return [(os.path.join(output_dir, sub), None) for sub in (subdirs or [])]


def read_parts(output_dir, subdirs=None, passes=None):
    """Collect part entries from a run. Pass `passes` (from run_split) to read
    exactly that run; `subdirs` alone reads the legacy fixed folders.

    Each entry gets: _path (absolute PNG path), _facial (bool) and _transform
    (the head-crop transform for facial parts, identity otherwise)."""
    entries = []
    for d, pass_name in _pass_folders(output_dir, subdirs, passes):
        meta = os.path.join(d, "parts_metadata.json")
        if not os.path.exists(meta):
            continue
        try:
            data = json.load(open(meta, encoding="utf-8"))
        except Exception:
            continue
        parts = data if isinstance(data, list) else data.get("parts", [])
        facial = (pass_name == "facial" or (isinstance(data, dict) and data.get("pass") == "facial")
                  or "facial" in os.path.basename(os.path.dirname(meta)).lower()
                  or "facial" in d.lower())
        transform = read_head_transform(output_dir, d, data if isinstance(data, dict) else None) \
            if facial else {"x_min": 0, "y_min": 0, "scale": 1.0}
        for p in parts:
            fpath = os.path.join(d, p.get("file", ""))
            if os.path.exists(fpath):
                p = dict(p)
                p["_path"] = fpath
                p["_facial"] = facial
                p["_transform"] = transform
                entries.append(p)
    return entries


def read_head_transform(output_dir, folder=None, meta=None):
    """The facial pass's head-crop transform {x_min, y_min, scale}.
    Looks in the pass metadata, then the pass folder, then the legacy global
    output/head_crop_transform.json. Identity when none is found.
    Facial parts map to the full image as: full = x_min + facial / scale."""
    cands = []
    if meta and meta.get("crop_transform"):
        cands.append(meta["crop_transform"])
    for path in ([os.path.join(folder, "head_crop_transform.json")] if folder else []) + \
            [os.path.join(output_dir, "head_crop_transform.json")]:
        try:
            cands.append(json.load(open(path, encoding="utf-8")))
        except Exception:
            pass
    for t in cands:
        try:
            scale = float(t.get("scale", 1.0)) or 1.0
            return {"x_min": int(t.get("x_min", 0)), "y_min": int(t.get("y_min", 0)),
                    "scale": scale}
        except Exception:
            continue
    return {"x_min": 0, "y_min": 0, "scale": 1.0}


def placement(p):
    """Where a part goes on the full image: (x0, y0, width, height, scale).
    width/height are the target size in full-image pixels; scale != 1 means the
    PNG (cut from the upscaled head crop) must be shrunk to fit."""
    t = p.get("_transform") or {"x_min": 0, "y_min": 0, "scale": 1.0}
    sc = float(t.get("scale", 1.0) or 1.0)
    if p.get("xyxy_full"):
        x0, y0, x1, y1 = p["xyxy_full"]
        return int(x0), int(y0), int(x1 - x0), int(y1 - y0), sc if p.get("_facial") else 1.0
    xyxy = p.get("xyxy") or [0, 0, 0, 0]
    if p.get("_facial"):
        w = int(round((xyxy[2] - xyxy[0] + 1) / sc))
        h = int(round((xyxy[3] - xyxy[1] + 1) / sc))
        return (int(round(t["x_min"] + xyxy[0] / sc)), int(round(t["y_min"] + xyxy[1] / sc)),
                w, h, sc)
    return int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0] + 1), int(xyxy[3] - xyxy[1] + 1), 1.0


def _btf(entries):
    if entries and all(e.get("draw_index") is not None for e in entries):
        return sorted(entries, key=lambda e: e["draw_index"])
    return sorted(entries, key=lambda e: -float(e.get("z_order", e.get("z_rank", 0)) or 0))


def draw_order(entries, anchors=("face", "head")):
    """One back-to-front list for both passes. Facial parts go directly in
    front of the face, not on top of everything, so bangs and hats can still
    cover the eyebrows. (Same rule as autosplit_core.drawing.merge.)"""
    body = _btf([e for e in entries if not e.get("_facial")])
    facial = _btf([e for e in entries if e.get("_facial")])
    for a in anchors:
        for i, e in enumerate(body):
            if str(e.get("tag", "")).strip().lower().replace("_", " ") == a:
                return body[: i + 1] + facial + body[i + 1:]
    return body + facial
