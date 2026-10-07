"""
Draw order from evidence in the picture.

The June node sorted parts by a fixed name table and used depth only to break
ties, so any part not in the table (a sash, a shin brace) fell to the very back
and a 3/4-view far arm was drawn over the torso. Here the table is only the
last of four cues. For every pair of parts whose outlines overlap we ask:

  colour      the overlap pixels belong to whichever part's own palette
              explains them; that part is in front. (Flat-shaded art makes
              palettes distinct. Adapted from Genie Labs' spine-animation-ai.)
  containment a part sitting inside another's filled outline (an eye inside the
              face, a brace inside the leg's silhouette) is in front of it.
  depth       nearness of each part just around the overlap, with the depth
              model's convention made explicit.
  prior       the rigger name table, as a tie-breaker.

The votes are combined per pair, the most confident constraints are added to a
graph unless they would form a cycle, and a topological sort gives the order.
Pairs that never overlap get no constraint and fall back to the prior.

Everything runs at a reduced analysis size (default longest side 1024), so it
costs well under a second for a 26-part character.
"""
import re

import numpy as np

from . import masks as M

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# --------------------------------------------------------------------------
# prior: rigger conventions, FRONT first. Lower rank = drawn further in front.
# Tokens are matched against the tag's words (left/right removed).
# --------------------------------------------------------------------------
PRIOR_GROUPS = [
    ("iris irides pupil", 0),
    ("eyelash eyelashes eyelid eyelids eyewhite eyeball eye eyes", 1),
    ("eyebrow eyebrows brow brows", 2),
    ("nose mouth lip lips teeth tongue mustache beard", 3),
    ("bangs fringe", 5),
    ("eyewear glasses goggles earring earwear headwear hat cap helmet hood horn horns", 6),
    ("face head ear ears", 8),
    ("sash belt strap straps brace braces bracelet glove gloves cuff cuffs scarf necklace "
     "neckwear tie collar bag pouch holster armband wristband badge weapon sword shield "
     "shoe shoes boot boots sandal sandals footwear sock socks legwarmer legwarmers", 10),
    ("arm arms forearm sleeve sleeves shoulder shoulders", 11),
    ("hand hands fist finger fingers handwear", 11.5),  # wrist tucks under the cuff
    ("jacket coat vest cape cloak topwear outerwear hoodie", 12),
    ("shirt undershirt blouse top tshirt bra dress", 13),
    ("skirt shorts pants trousers jeans bottomwear", 14),
    ("neck torso body chest waist hips belly abdomen", 15),
    ("leg legs thigh thighs shin shins knee calf", 16),
    ("foot feet toe toes", 17),  # bare shin draws over the top of the foot
    ("tail wings wing", 18),
    ("hair ponytail braid", 19),
]
PHRASES = {
    "hair front": 5, "front hair": 5, "hair_front": 5,
    "hair back": 20, "back hair": 20, "hair_back": 20,
    "bottom lips": 3, "upper lips": 3, "shin brace": 10, "leg brace": 10,
    "under shirt": 13, "upper arm": 11, "lower arm": 11,
}
UNKNOWN_RANK = 9  # unknown things are usually props or clothing details on top

_LR = re.compile(r"\b(left|right|l|r)\b")


def _norm(tag):
    t = tag.lower().replace("_", " ").replace("-", " ")
    t = _LR.sub(" ", t)
    return " ".join(t.split())


def prior_rank(tag):
    raw = tag.lower()
    if raw in PHRASES:
        return PHRASES[raw]
    t = _norm(tag)
    if t in PHRASES:
        return PHRASES[t]
    for phrase, rank in PHRASES.items():
        if " " in phrase and phrase in t:
            return rank
    words = t.split()
    for vocab, rank in PRIOR_GROUPS:
        vs = vocab.split()
        if any(w in vs for w in words):
            return rank
    return UNKNOWN_RANK


# --------------------------------------------------------------------------
# colour palettes
# --------------------------------------------------------------------------
_BITS = 4
_LEVELS = 1 << _BITS


def _quantize(rgb_u8):
    q = (rgb_u8 >> (8 - _BITS)).astype(np.int32)
    return (q[..., 0] * _LEVELS + q[..., 1]) * _LEVELS + q[..., 2]


def _palette(qimg, mask):
    n = int(mask.sum())
    if n == 0:
        return None
    h = np.bincount(qimg[mask], minlength=_LEVELS ** 3).astype(np.float32) / n
    cube = h.reshape(_LEVELS, _LEVELS, _LEVELS)
    # neighbour max: tolerate shading and anti-aliasing one level away
    out = cube.copy()
    p = np.pad(cube, 1)
    for dz in (0, 1, 2):
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                np.maximum(out, p[dz:dz + _LEVELS, dy:dy + _LEVELS, dx:dx + _LEVELS], out=out)
    return out.reshape(-1)


def _explained(qimg, region, pal, tau=0.0015):
    if pal is None or not region.any():
        return None
    return float((pal[qimg[region]] >= tau).mean())


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------
def compute_order(rgb, masks, near=None, analysis_max=1024, min_overlap=25,
                  weights=None, prior=None, notch_frac=0.02):
    """
    rgb:    HxWx3 uint8 (or float 0-1) picture the masks were cut from.
    masks:  dict tag -> HxW bool (visible pixels as segmented).
    near:   optional HxW map where higher = closer (see masks.nearness).
    prior:  optional dict tag -> rank override.

    Returns dict:
      back_to_front: [tag, ...]
      z_order: {tag: 0 for the front-most ...}
      edges: [{front, back, confidence, cues}, ...]   (kept constraints)
      dropped: [...]                                  (would have made a cycle)
    """
    # notch is off by default: on the Salena benchmark it cost 3 points of
    # pairwise accuracy (hair framing a face reads as the face cutting into it).
    w = {"colour": 1.0, "containment": 0.6, "notch": 0.0, "depth": 0.4, "prior": 0.25}
    if weights:
        w.update(weights)
    total_w = sum(v for v in w.values() if v > 0) or 1.0
    tags = list(masks.keys())
    if len(tags) < 2:
        return {"back_to_front": tags, "z_order": {t: 0 for t in tags},
                "edges": [], "dropped": []}
    rank = {t: (prior or {}).get(t, prior_rank(t)) for t in tags}

    img = np.asarray(rgb)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = img[..., :3]
    H, W = img.shape[:2]
    s = min(1.0, float(analysis_max) / max(H, W))
    h, wd = max(1, int(round(H * s))), max(1, int(round(W * s)))
    small = cv2.resize(img, (wd, h), interpolation=cv2.INTER_AREA) if s < 1 else img
    q = _quantize(small)

    def down(m):
        m = M.as_bool(m)
        if m.shape != (H, W):
            m = M.resize_to(m.astype(np.float32), (H, W), nearest=True) > 0.5
        if s < 1:
            return cv2.resize(m.astype(np.uint8), (wd, h), interpolation=cv2.INTER_NEAREST) > 0
        return m

    vis = {t: down(masks[t]) for t in tags}
    filled = {t: M.fill_holes(vis[t]) for t in tags}
    claim = np.zeros((h, wd), np.int16)
    for t in tags:
        claim += vis[t]
    own = {t: vis[t] & (claim == 1) for t in tags}
    nm = None
    if near is not None:
        nm = M.depth_2d(near)
        nm = cv2.resize(nm, (wd, h), interpolation=cv2.INTER_AREA)
    pal = {t: _palette(q, own[t] if own[t].sum() >= 30 else vis[t]) for t in tags}
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

    # closing fills the notches an occluder cuts into a part's outline
    r = max(3, int(round(notch_frac * max(h, wd))))
    notch_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    touch_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = {t: cv2.morphologyEx(filled[t].astype(np.uint8), cv2.MORPH_CLOSE, notch_k,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
              for t in tags}
    near_zone = {t: cv2.dilate(filled[t].astype(np.uint8), touch_k) > 0 for t in tags}

    cand = []
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            ov = filled[a] & filled[b]
            n = int(ov.sum())
            touching = int((near_zone[a] & filled[b]).sum())
            if n < min_overlap and touching < min_overlap:
                continue
            cues, score, wsum = {}, 0.0, 0.0
            if n >= min_overlap:
                # colour: who owns the overlap pixels?
                ea, eb = _explained(q, ov, pal[a]), _explained(q, ov, pal[b])
                if ea is not None and eb is not None:
                    v = ea - eb
                    cues["colour"] = round(v, 3)
                    score += w["colour"] * v
                    wsum += w["colour"]
                # containment: a part mostly inside the other's outline is on
                # top - but only when the colours tell the two apart. Two masks
                # claiming the same sleeve (jacket and arm) are a duplicate, not
                # a decal on top of its base.
                ca = n / float(max(1, filled[a].sum()))
                cb = n / float(max(1, filled[b].sum()))
                v = float(np.clip((ca - cb) / 0.5, -1, 1)) if max(ca, cb) > 0.6 else 0.0
                col = cues.get("colour")
                if col is None or abs(col) < 0.1:
                    v = 0.0
                if v:
                    cues["containment"] = round(v, 3)
                    score += w["containment"] * v
                    wsum += w["containment"]
            # notch: does b sit inside a's closed outline (b cuts into a), or
            # the other way round? The cutter is in front.
            b_in_a = int((closed[a] & ~filled[a] & vis[b]).sum())
            a_in_b = int((closed[b] & ~filled[b] & vis[a]).sum())
            tot = b_in_a + a_in_b
            if w["notch"] and tot >= min_overlap:
                v = (a_in_b - b_in_a) / float(tot)
                if abs(v) > 0.2:
                    cues["notch"] = round(v, 3)
                    score += w["notch"] * v
                    wsum += w["notch"]
            # depth: nearness just around where the two meet (the overlap, or
            # the contact strip for parts that only touch)
            if nm is not None:
                zone = ov if n >= min_overlap else (
                    (near_zone[a] & filled[b]) | (near_zone[b] & filled[a]))
                ring = cv2.dilate(zone.astype(np.uint8), ring_k) > 0
                ra, rb = ring & own[a], ring & own[b]
                # a part claimed almost entirely by its neighbour too (a
                # sleeve inside "jacket") has no pixels of its own near the
                # contact: sample its whole mask instead
                if ra.sum() < 10:
                    ra = vis[a]
                if rb.sum() < 10:
                    rb = vis[b]
                if ra.sum() >= 10 and rb.sum() >= 10:
                    d = float(np.median(nm[ra]) - np.median(nm[rb]))
                    span = float(nm.max() - nm.min()) or 1.0
                    v = float(np.clip(d / (0.08 * span), -1, 1))
                    cues["depth"] = round(v, 3)
                    score += w["depth"] * v
                    wsum += w["depth"]
            # prior
            if rank[a] != rank[b]:
                v = 1.0 if rank[a] < rank[b] else -1.0
                cues["prior"] = v
                score += w["prior"] * v
                wsum += w["prior"]
            if wsum == 0 or score == 0:
                continue
            front, back = (a, b) if score > 0 else (b, a)
            # confidence is measured against ALL the evidence a pair could
            # have, so a prior-only pair (no overlap, no depth) stays weak and
            # never outranks a pair the picture actually decides
            cand.append({"front": front, "back": back,
                         "confidence": round(abs(score) / total_w, 3),
                         "overlap_px": int(round(max(n, touching) / (s * s))),
                         "cues": cues})

    # ---- add constraints, most confident first, skipping cycles ---------
    behind = {t: set() for t in tags}  # behind[x] = things that must be drawn before x

    def reaches(src, dst):
        stack, seen = [src], set()
        while stack:
            x = stack.pop()
            if x == dst:
                return True
            if x in seen:
                continue
            seen.add(x)
            stack.extend(behind[x])
        return False

    edges, dropped = [], []
    for c in sorted(cand, key=lambda c: (-c["confidence"], -c["overlap_px"])):
        f, bk = c["front"], c["back"]
        if reaches(bk, f):  # back already must be in front of front: cycle
            dropped.append(c)
            continue
        behind[f].add(bk)
        edges.append(c)

    # ---- topological sort, back-most first ------------------------------
    # Parts with no constraint between them (they never touch) are placed by
    # a blend of the name table and how far away the part is overall, so a
    # 3/4-view far arm still lands behind the torso it never touches.
    max_rank = float(max(rank.values()) or 1)
    farness = {}
    if nm is not None:
        lo, hi = float(nm.min()), float(nm.max())
        for t in tags:
            v = float(np.median(nm[vis[t]])) if vis[t].any() else lo
            farness[t] = 1.0 - ((v - lo) / (hi - lo) if hi > lo else 0.5)
    backness = {t: (0.5 * rank[t] / max_rank + 0.5 * farness[t]) if farness else rank[t] / max_rank
                for t in tags}
    remaining = set(tags)
    order = []
    while remaining:
        ready = [t for t in remaining if not (behind[t] & remaining)]
        if not ready:  # cannot happen (cycles skipped) but stay safe
            ready = list(remaining)
        nxt = max(ready, key=lambda t: (backness[t], -tags.index(t)))
        order.append(nxt)
        remaining.remove(nxt)
    z = {t: len(order) - 1 - i for i, t in enumerate(order)}
    return {"back_to_front": order, "z_order": z, "edges": edges, "dropped": dropped,
            "prior_rank": rank}
