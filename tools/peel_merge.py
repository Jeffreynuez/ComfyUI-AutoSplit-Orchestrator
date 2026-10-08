"""
Pixel ownership + hidden-area fill from "peeled" pictures (prototype).

A peeled picture is the same character with some front layers taken away,
made by an image-edit model in the same pose (the base body without its
clothes, the skirt without its sash, the head without its hair...). Splitting
the peeled picture gives a part's shape and paint where the original picture
hides it. This tool merges that back into a split:

  ownership   every visible pixel goes to the front-most part (draw order),
              so a jacket no longer carries the sleeves drawn in front of it
  fill        for each part named in the plan
                hidden candidates = pixels owned by parts in front of it (plus,
                                    for steps marked claim_unowned, visible
                                    character pixels no part owns)
                added             = peeled mask AND hidden candidates
                part              = owned pixels + peeled pixels in `added`

Original visible pixels are never repainted. Writes a copy of the body pass
(schema 2) for evaluate.py and wiggle.py.

Plan file (JSON list, one entry per peeled picture):
  [{"image": "<peeled picture, same size as the original>",
    "run":   "<split of the peeled picture>",
    "parts": {"torso": ["torso", "sports bra", "briefs"], "left leg": ["left leg"]}}]
A part's peeled shape is the union of the listed tags from that run. An
optional "anchors": [tags] names parts the edit should have left alone; their
pixels fit the colour map that undoes the edit's colour drift (default: the
whole silhouette, keeping the best-agreeing 60%). "colour" forces one of raw,
global, part, hist, or "interior": one flat deep-shadow colour from the visible
pixels of "interior_of" (default the part), for parts behind the head or body.

A step may instead be a rule-based fill with no peeled picture:
  {"hull": {"tag": "hair_back", "group": ["hair_back", "hair_front"],
            "close": 0.08, "colour": "interior" | "shadow" | "median",
            "mode": "close" | "convex", "below": "face"}}
fills the part inside the closed (or convex) outline of the group, behind the parts in
front of it, no higher than the top of `below`, with one flat colour: "interior" is
the group's deep shadow (what the Salena rig paints behind the head), "shadow" the
part's own darkest visible pixels.

Rigging-convention steps (autosplit_core/rigfill.py), applied after the fills:
  {"underlap": {"tag": "left leg", "into": ["torso"], "length": 0.6}}
      the part continues under the parts it joins, toward them, by a share of
      its thickness, coloured from its nearest own pixels;
  {"back_panel": {"of": "jacket", "tag": "jacket back"}}
      a new part behind the body (in front of the back hair): inside the
      garment's convex outline, in the garment's deep interior shadow;
  {"skin_under": {"tag": "face", "features": ["left eye", "nose", ...]}}
      the head under the facial features is plain skin (inpainted from the
      face around them), as the rigger paints the head layer.

    python tools/peel_merge.py --run <body pass> [--run <facial pass>] --source <picture> \
        --plan plan.json --out <new body pass folder> [--no-own] [--raw-colour]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ComfyUI-AutoSplit-Orchestrator"))
from autosplit_core import ownership, rigfill  # noqa: E402
from partsio import back_to_front, load_pass  # noqa: E402

import cv2  # noqa: E402


def rgb_to_lab(rgb):
    return cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def lab_to_rgb(lab):
    lab = np.clip(lab, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).reshape(-1, 3)


def fit_colour_map(peel_rgb, orig_rgb, mask, n=200000, rounds=4, keep=0.6, seed=0):
    """Lab map (4x3, per channel gain + offset) taking the peeled picture's
    colours onto the original's, fitted on pixels the edit left alone. Edits shift colours
    globally (lighter skin, brighter greens), so pixels inside the silhouette
    are sampled and the best-agreeing `keep` share is refitted a few times,
    which drops the pixels the edit really changed. Returns (map, rmse)."""
    idx = np.flatnonzero(mask)
    if len(idx) < 1000:
        return None, None
    rng = np.random.default_rng(seed)
    if len(idx) > n:
        idx = rng.choice(idx, n, replace=False)
    a = rgb_to_lab(peel_rgb.reshape(-1, 3)[idx])
    b = rgb_to_lab(orig_rgb.reshape(-1, 3)[idx])
    # per-channel gain + offset only: a full 3x3 fit lets lightness borrow
    # from hue and wrecks colours the anchors never showed (a green bra fitted
    # on skin turns pink)
    X = np.hstack([a, np.ones((len(a), 1), np.float32)])
    sel = np.ones(len(a), bool)
    M = np.zeros((4, 3), np.float32)
    for _ in range(rounds):
        for c in range(3):
            A_ = np.stack([a[sel, c], np.ones(sel.sum(), np.float32)], axis=1)
            (gain, off), *_ = np.linalg.lstsq(A_, b[sel, c], rcond=None)
            gain = float(np.clip(gain, 0.6, 1.6))
            off = float(np.mean(b[sel, c] - gain * a[sel, c]))
            M[:, c] = 0
            M[c, c] = gain
            M[3, c] = off
        r = np.linalg.norm(X @ M - b, axis=1)
        sel = r <= np.quantile(r, keep)
    rmse = float(np.sqrt(np.mean(np.sum((X[sel] @ M - b[sel]) ** 2, axis=1))))
    return M, rmse


def apply_colour_map(px, M):
    if M is None or len(px) == 0:
        return px
    a = rgb_to_lab(px)
    return lab_to_rgb(np.hstack([a, np.ones((len(a), 1), np.float32)]) @ M)


def lab_err(px, ref):
    if len(px) == 0:
        return None
    return float(np.mean(np.linalg.norm(rgb_to_lab(px) - rgb_to_lab(ref), axis=1)))


def _hist_via(q, gen_part, own_px):
    """Histogram-match q using the mapping that takes the peeled part's
    colours (gen_part) onto the original part's visible colours (own_px)."""
    if len(q) == 0:
        return q
    a, g, r = rgb_to_lab(q), rgb_to_lab(gen_part), rgb_to_lab(own_px)
    out = np.empty_like(a)
    qs = np.linspace(0, 1, 257)
    for c in range(3):
        out[:, c] = np.interp(a[:, c], np.quantile(g[:, c], qs), np.quantile(r[:, c], qs))
    return lab_to_rgb(out)


def keep_islands(add, P, A_area, min_share):
    """Keep added islands that are big or that touch the part's own pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(add.astype(np.uint8), connectivity=8)
    near = cv2.dilate(P.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    touching = set(np.unique(labels[near & add]).tolist())
    thr = min_share * A_area
    ok = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= thr or i in touching]
    return np.isin(labels, ok) & add


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--source", required=True, help="the picture that was split (sets the frame)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-own", action="store_true", help="keep each part's full SAM 3 mask")
    ap.add_argument("--raw-colour", action="store_true",
                    help="keep the peeled picture's colours as they are")
    ap.add_argument("--min-island", type=float, default=0.01)
    ap.add_argument("--max-map-rmse", type=float, default=8.0,
                    help="ignore a fitted colour map whose fit error (Lab) is above this")
    args = ap.parse_args()

    with open(args.plan, encoding="utf-8") as f:
        plan = json.load(f)

    parts = []
    for f in args.run:
        parts += load_pass(f)
    order = back_to_front(parts)
    src_im = Image.open(args.source).convert("RGB")
    W, H = src_im.size
    src_rgb = np.asarray(src_im)
    btf = [(p.tag, p.full_mask(W, H)) for p in order]
    owner, _ = ownership.owner_map(btf)
    vis = {n: owner == i for i, (n, _) in enumerate(btf)}
    silhouette = np.zeros((H, W), bool)
    for _, m in btf:
        silhouette |= m
    # visible character pixels the split gave to no part: a fill may claim them
    # (keeping their own colour) where its peeled shape covers them
    orphan = rigfill.picture_silhouette(src_rgb) & (owner < 0)
    names_idx = {n: i for i, (n, _) in enumerate(btf)}

    body_dir = args.run[0]
    os.makedirs(args.out, exist_ok=True)
    for fn in os.listdir(body_dir):
        if fn.lower().endswith(".png") or fn == "parts_metadata.json":
            shutil.copy2(os.path.join(body_dir, fn), os.path.join(args.out, fn))
    with open(os.path.join(args.out, "parts_metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    entries = {e["tag"]: e for e in meta["parts"]}

    # what each part gets added, from every plan step that names it
    added = {}
    for step in plan:
        if "image" not in step:
            continue
        im = Image.open(step["image"]).convert("RGB")
        if im.size != (W, H):
            im = im.resize((W, H), Image.LANCZOS)
        img = np.asarray(im)
        gmap, grmse = None, None
        if (not args.raw_colour):
            anchor = silhouette
            if step.get("anchors"):
                anchor = np.zeros((H, W), bool)
                for t in step["anchors"]:
                    if t in vis:
                        anchor |= vis[t]
            gmap, grmse = fit_colour_map(img, src_rgb, anchor)
            print("colour map %-28s rmse %.1f" % (os.path.basename(step["image"]), grmse or -1))
        peeled = {}
        for p in load_pass(step["run"]):
            peeled.setdefault(p.tag, np.zeros((H, W), bool))
            peeled[p.tag] |= p.full_mask(W, H)
        for tag, sources in step["parts"].items():
            if tag not in entries:
                continue
            A = np.zeros((H, W), bool)
            for s in sources:
                if s in peeled:
                    A |= peeled[s]
            if not A.any():
                continue
            P = vis[tag] if not args.no_own else next(p for p in order if p.tag == tag).full_mask(W, H)
            box = ownership.expand_box(ownership.bbox(A | P), 0.05, (H, W))
            region, front = ownership.hidden_candidates(btf, tag, box=box, owner=owner)
            bx0, by0, bx1, by1 = box
            if step.get("claim_unowned"):
                claim = np.zeros((H, W), bool)
                claim[by0:by1, bx0:bx1] = orphan[by0:by1, bx0:bx1]
                region = region | claim
            # a limb does not continue into its own hand or foot (the rigger
            # draws the leg over the foot, not the other way round)
            skip = [names_idx[t] for t in step.get("exclude", {}).get(tag, []) if t in names_idx]
            if skip:
                region &= ~np.isin(owner, skip)
            add = A & region & ~P
            if tag in added:
                add &= ~added[tag]["mask"]
            if not add.any():
                continue
            add = keep_islands(add, P, A.sum(), args.min_island)
            rec = added.setdefault(tag, {"mask": np.zeros((H, W), bool),
                                         "rgb": np.zeros((H, W, 3), np.uint8), "behind": set(),
                                         "sources": []})
            px = img[add]
            how = "raw"
            if (not args.raw_colour):
                # candidates: raw, the picture-wide map, a map fitted on this
                # part, and a distribution match to the part's own visible
                # pixels; keep the one that best reproduces the part where both
                # pictures show it
                ref = P & A
                maps = {"raw": lambda q: q}
                if gmap is not None and grmse is not None and (
                        grmse <= args.max_map_rmse or step.get("colour") == "global"):
                    maps["global"] = lambda q, M=gmap: apply_colour_map(q, M)
                own_px = src_rgb[P]
                if len(own_px) >= 300:
                    gen_part = img[A]
                    maps["hist"] = lambda q, g=gen_part, o=own_px: _hist_via(q, g, o)
                if ref.sum() >= 300:
                    Mp, prmse = fit_colour_map(img, src_rgb, ref, rounds=2, keep=0.8)
                    if Mp is not None and prmse <= args.max_map_rmse:
                        maps["part"] = lambda q, M=Mp: apply_colour_map(q, M)
                    scores = {k: lab_err(f(img[ref]), src_rgb[ref]) for k, f in maps.items()}
                    how = min(scores, key=scores.get)
                elif step.get("colour") in maps:
                    how = step["colour"]
                elif "global" in maps:
                    how = "global"
                if step.get("colour") in maps:
                    how = step["colour"]
                px = maps[how](px)
            if step.get("colour") == "interior":
                # a part behind the head / body: one flat deep-shadow colour
                idx = [names_idx[t] for t in step.get("interior_of", [tag]) if t in names_idx]
                px = np.repeat(rigfill.interior_colour(src_rgb[np.isin(owner, idx)])[None], len(px), 0)
                how = "interior"
            rec["mask"] |= add
            rec["rgb"][add] = px
            seen = add & orphan                          # visible: keep the picture's paint
            rec["rgb"][seen] = src_rgb[seen]
            rec["claimed"] = rec.get("claimed", 0) + int(seen.sum())
            rec.setdefault("colour", []).append(how)
            rec["behind"] |= set(front)
            rec["sources"].append(os.path.basename(step["image"]))

    # rule-based fills: a part continues inside the closed outline of a group
    # (back hair behind the head: inside the outline of all the hair), painted
    # with a flat colour taken from the part's own visible pixels
    for step in plan:
        if "hull" not in step:
            continue
        h = step["hull"]
        tag = h["tag"]
        if tag not in entries:
            continue
        grp = np.zeros((H, W), bool)
        for t, m in btf:
            if t in h.get("group", [tag]):
                grp |= m
        if not grp.any():
            continue
        x0, y0, x1, y1 = ownership.bbox(grp)
        if h.get("mode") == "convex":
            pts = cv2.findNonZero(grp.astype(np.uint8))
            closed = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(closed, cv2.convexHull(pts), 1)
            closed = closed.astype(bool)
        else:
            k = max(5, int(h.get("close", 0.08) * max(x1 - x0, y1 - y0)) | 1)
            closed = cv2.morphologyEx(grp.astype(np.uint8), cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            ff = np.pad(closed, 1)
            cv2.floodFill(ff, None, (0, 0), 2)
            closed = (ff[1:-1, 1:-1] != 2)
        P = vis[tag]
        region, front = ownership.hidden_candidates(btf, tag, box=(x0, y0, x1, y1), owner=owner)
        add = closed & region & ~P
        if tag in added:
            add &= ~added[tag]["mask"]
        if not add.any() or P.sum() < 50:
            continue
        if h.get("below") in names_idx:
            # back hair hangs from the scalp: nothing above the top of the head
            head = vis[h["below"]] | (added[h["below"]]["mask"] if h["below"] in added else False)
            if head.any():
                add[:ownership.bbox(head)[1]] = False
        mode = h.get("colour", "shadow")
        if mode == "interior":
            # behind the head and body the rig paints deep shadow
            idx = [names_idx[t] for t in h.get("group", [tag]) if t in names_idx]
            rgb = rigfill.interior_colour(src_rgb[np.isin(owner, idx)])
        else:
            lab = rgb_to_lab(src_rgb[P])
            if mode == "shadow":
                dark = lab[:, 0] <= np.quantile(lab[:, 0], 0.3)
                c = np.median(lab[dark], axis=0)
            else:
                c = np.median(lab, axis=0)
            rgb = lab_to_rgb(c[None, :])[0]
        rec = added.setdefault(tag, {"mask": np.zeros((H, W), bool),
                                     "rgb": np.zeros((H, W, 3), np.uint8), "behind": set(),
                                     "sources": []})
        rec["mask"] |= add
        rec["rgb"][add] = rgb
        rec["behind"] |= set(front)
        rec["sources"].append("hull:" + "+".join(h.get("group", [tag])))
        rec.setdefault("colour", []).append("flat-" + h.get("colour", "shadow"))

    # ---- rigging conventions: underlaps and back panels --------------------
    paint = {}

    def part_paint(tag):
        """(filled mask, colours) of a body part after the fills so far."""
        if tag not in paint:
            p = next(q for q in order if q.tag == tag)
            rgb = np.zeros((H, W, 3), np.uint8)
            pm = np.zeros((H, W), bool)
            p.paint_onto(rgb, pm)
            m = (pm if args.no_own else vis[tag]).copy()
            rec = added.get(tag)
            if rec is not None:
                rgb[rec["mask"]] = rec["rgb"][rec["mask"]]
                m |= rec["mask"]
            paint[tag] = (m, rgb)
        return paint[tag]

    # under the facial features the head is plain skin: the rigger paints the
    # face layer without eyes or mouth, which are layers of their own
    for step in plan:
        if "skin_under" not in step:
            continue
        su = step["skin_under"]
        tag = su["tag"]
        if tag not in entries:
            continue
        under = [names_idx[t] for t in su.get("features", []) if t in names_idx]
        face, rgb = part_paint(tag)
        region = np.isin(owner, under) & cv2.dilate(face.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        if region.sum() < 20:
            continue
        x0, y0, x1, y1 = ownership.expand_box(ownership.bbox(region), 0.1, (H, W))
        known = face[y0:y1, x0:x1] & ~region[y0:y1, x0:x1]
        crop = rgb[y0:y1, x0:x1].copy()
        crop[~known] = 0
        hole = (~known & (region[y0:y1, x0:x1] | ~face[y0:y1, x0:x1])).astype(np.uint8)
        filled = cv2.inpaint(np.ascontiguousarray(crop), hole, 7, cv2.INPAINT_TELEA)
        r = region[y0:y1, x0:x1]
        rec = added.setdefault(tag, {"mask": np.zeros((H, W), bool),
                                     "rgb": np.zeros((H, W, 3), np.uint8), "behind": set(),
                                     "sources": []})
        rec["mask"][y0:y1, x0:x1] |= r
        rec["rgb"][y0:y1, x0:x1][r] = filled[r]
        rec["sources"].append("skin_under:" + "+".join(t for t in su.get("features", []) if t in names_idx))
        rec.setdefault("colour", []).append("inpaint")
        paint.pop(tag, None)

    for step in plan:
        if "underlap" not in step:
            continue
        u = step["underlap"]
        tag = u["tag"]
        into = [t for t in u.get("into", []) if t in entries]
        if tag not in entries or not into:
            continue
        child, rgb = part_paint(tag)
        parent = np.zeros((H, W), bool)
        for t in into:
            parent |= part_paint(t)[0]
        allowed, front = ownership.hidden_candidates(btf, tag, box=(0, 0, W, H), owner=owner)
        stub = rigfill.underlap(child, parent, allowed, float(u.get("length", 0.5)))
        if stub.sum() < 50:
            continue
        cols = rigfill.nearest_colours(rgb, child, stub)
        rec = added.setdefault(tag, {"mask": np.zeros((H, W), bool),
                                     "rgb": np.zeros((H, W, 3), np.uint8), "behind": set(),
                                     "sources": []})
        rec["mask"] |= stub
        rec["rgb"][stub] = cols
        rec["behind"] |= set(into)
        rec["sources"].append("underlap:" + "+".join(into))
        rec.setdefault("colour", []).append("nearest")
        paint.pop(tag, None)

    new_parts = []
    back_hair = [i for i, (t, _) in enumerate(btf) if "hair" in t.lower() and "back" in t.lower()]
    for step in plan:
        if "back_panel" not in step:
            continue
        b = step["back_panel"]
        of = b["of"]
        if of not in entries:
            continue
        g, _ = part_paint(of)
        panel = rigfill.back_panel(g) & (owner >= 0) & ~np.isin(owner, back_hair)
        if panel.sum() < max(200, 0.03 * g.sum()):
            continue
        own_px = src_rgb[vis[of]] if vis[of].any() else src_rgb[g]
        new_parts.append({"tag": b.get("tag", of + " back"), "of": of, "mask": panel,
                          "rgb": rigfill.interior_colour(own_px)})

    report = []
    for p in order:
        if p.facial or p.tag not in entries:
            continue
        tag = p.tag
        full = p.full_mask(W, H)
        own = full if args.no_own else vis[tag]
        rec = added.get(tag)
        if rec is None and own.sum() == full.sum():
            continue  # untouched
        rgb = np.zeros((H, W, 3), np.uint8)
        pm = np.zeros((H, W), bool)
        p.paint_onto(rgb, pm)
        m = own.copy()
        canvas = np.zeros((H, W, 4), np.uint8)
        canvas[own, :3] = rgb[own]
        canvas[own, 3] = 255
        n_add = 0
        if rec is not None:
            add = rec["mask"] & ~own
            canvas[add, :3] = rec["rgb"][add]
            canvas[add, 3] = 255
            m |= add
            n_add = int(add.sum())
        e = entries[tag]
        if not m.any():
            e["flags"] = sorted(set(e.get("flags", [])) | {"fully_covered"})
            report.append((tag, int(full.sum() - own.sum()), 0, []))
            continue
        bx0, by0, bx1, by1 = ownership.bbox(m)
        Image.fromarray(canvas[by0:by1, bx0:bx1]).save(os.path.join(args.out, e["file"]))
        e["xyxy"] = [bx0, by0, bx1, by1]
        e["xyxy_full"] = [bx0, by0, bx1, by1]
        e["area_px"] = int(m.sum())
        flags = set(e.get("flags", []))
        if own.sum() < full.sum():
            flags.add("owned")
        if n_add:
            flags.add("filled")
            e["fill"] = {"added_px": n_add, "behind": sorted(rec["behind"]),
                         "sources": rec["sources"], "colour": rec.get("colour", []),
                         "claimed_unowned_px": rec.get("claimed", 0)}
        e["flags"] = sorted(flags)
        report.append((tag, int(full.sum() - own.sum()), n_add, sorted(rec["behind"]) if rec else []))

    # back panels: new parts directly in front of the back hair
    if new_parts:
        body_entries = meta["parts"]
        hair_idx = [e.get("draw_index", 0) for e in body_entries
                    if "hair" in e["tag"].lower() and "back" in e["tag"].lower()]
        pos = (max(hair_idx) + 1) if hair_idx else 0
        prefix = meta.get("character") or "part"
        for k, npart in enumerate(new_parts):
            m = npart["mask"]
            bx0, by0, bx1, by1 = ownership.bbox(m)
            canvas = np.zeros((by1 - by0, bx1 - bx0, 4), np.uint8)
            sub = m[by0:by1, bx0:bx1]
            canvas[sub, :3] = npart["rgb"]
            canvas[sub, 3] = 255
            slug = "_".join(npart["tag"].lower().split())
            fn = "%s_%s.png" % (prefix, slug)
            Image.fromarray(canvas).save(os.path.join(args.out, fn))
            for e in body_entries:
                if e.get("draw_index") is not None and e["draw_index"] >= pos + k:
                    e["draw_index"] += 1
            body_entries.append({
                "tag": npart["tag"], "source_part": npart["of"], "file": fn, "mask_file": None,
                "mask_type": None, "xyxy": [bx0, by0, bx1, by1], "xyxy_full": [bx0, by0, bx1, by1],
                "draw_index": pos + k, "passthrough": False, "area_px": int(m.sum()),
                "flags": ["back_panel", "filled"], "in_front_of": [],
                "fill": {"added_px": int(m.sum()), "sources": ["back_panel:" + npart["of"]],
                         "colour": ["interior"]}})
            report.append((npart["tag"], 0, int(m.sum()), ["(new part behind the body)"]))
        n = len(body_entries)
        for e in body_entries:
            if e.get("draw_index") is not None:
                e["z_order"] = n - 1 - e["draw_index"]

    meta["fill"] = {"plan": os.path.basename(args.plan), "owned": not args.no_own,
                    "colour_match": not args.raw_colour}
    with open(os.path.join(args.out, "parts_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("%-16s %9s %9s  behind" % ("part", "-owned", "+filled"))
    for tag, lost, n, front in report:
        print("%-16s %9d %9d  %s" % (tag, lost, n, ", ".join(front) or "-"))
    print("filled pass ->", args.out)


if __name__ == "__main__":
    main()
