"""
ComfyUI client for the AutoSplit Studio Krita plugin.

Pure standard-library (urllib) so it runs in Krita's bundled Python with no
extra packages. Mirrors the proven comfy_autosplit_test.py logic:
  upload image -> strip optional Florence-2 -> inject image -> queue -> wait.
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


def run_split(comfy_url, workflow_api_path, image_path, progress=None, timeout_s=1200):
    """Upload image, run the SAM3 AutoSplit workflow, wait for completion.
    progress(msg) is an optional callback for status updates."""
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
    ref = upload_image(comfy_url, image_path)
    say("uploaded %s" % os.path.basename(image_path))
    n = inject_input_image(wf, ref)
    if n == 0:
        raise RuntimeError("No LoadImage node in the workflow - is it API format?")

    client_id = uuid.uuid4().hex
    res = _post_json(comfy_url + "/prompt", {"prompt": wf, "client_id": client_id})
    pid = res["prompt_id"]
    say("queued (%s); running split..." % pid[:8])

    start = time.time()
    last_tick = 0
    while time.time() - start < timeout_s:
        hist = _get_json(comfy_url + "/history/" + pid)
        if pid in hist:
            entry = hist[pid]
            st = entry.get("status", {})
            if st.get("completed") or entry.get("outputs"):
                say("split finished in %.0fs" % (time.time() - start))
                return pid
            if st.get("status_str") == "error":
                raise RuntimeError("ComfyUI workflow errored - see ComfyUI console.")
        elapsed = time.time() - start
        if elapsed - last_tick >= 3:
            last_tick = elapsed
            say("...running (%.0fs)" % elapsed)
        time.sleep(1.0)
    raise TimeoutError("Split timed out after %ds." % timeout_s)


def read_parts(output_dir, subdirs):
    """Collect part entries from the orchestrator output folders. Returns a list
    of dicts with keys: tag, file, xyxy, z_rank, _path (absolute PNG path)."""
    entries = []
    for sub in subdirs:
        d = os.path.join(output_dir, sub)
        meta = os.path.join(d, "parts_metadata.json")
        if not os.path.exists(meta):
            continue
        try:
            data = json.load(open(meta, encoding="utf-8"))
        except Exception:
            continue
        parts = data if isinstance(data, list) else data.get("parts", [])
        for p in parts:
            fpath = os.path.join(d, p.get("file", ""))
            if os.path.exists(fpath):
                p = dict(p)
                p["_path"] = fpath
                p["_facial"] = "facial" in sub.lower()
                entries.append(p)
    return entries


def read_head_transform(output_dir):
    """Read the head-crop transform written by HeadCropUpscale (facial pass).
    Returns {x_min, y_min, scale}; identity if not present. Facial parts are
    mapped to full-image space via: full = x_min + facial / scale, and the part
    image is downscaled by 1/scale."""
    path = os.path.join(output_dir, "head_crop_transform.json")
    try:
        t = json.load(open(path, encoding="utf-8"))
        scale = float(t.get("scale", 1.0)) or 1.0
        return {"x_min": int(t.get("x_min", 0)),
                "y_min": int(t.get("y_min", 0)),
                "scale": scale}
    except Exception:
        return {"x_min": 0, "y_min": 0, "scale": 1.0}
