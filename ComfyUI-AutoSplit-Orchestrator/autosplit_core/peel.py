"""
Plan the "peels" that fill hidden areas, from the split alone.

A peel is an image edit of the flat picture that takes some front layers away
in the same pose, so the parts behind them can be cut whole from the edited
picture (see tools/peel_merge.py). This module decides which peels to make
and writes their instructions from the part labels, the draw order and the
pixels, with nothing written for a particular character:

  base body   bare body parts (visible pixels look like skin) covered by
              clothes: one edit removes all those clothes and leaves the body
              in a plain top and briefs (the usual rig convention for a base
              body; set it with base_layer)
  garment     a garment covered by other parts: remove what covers it
  head        a face/head covered by hair: remove the hair
  hair rule   back hair is filled by rule, not by an edit (generated back
              hair came out far too long)

Each part gets a readable name with a colour word from its own pixels
("the brown jacket"), which helps the edit model find it; when a part and what
covers it share a colour, the lighter or darker one says so ("the light green
sash that lies over the green skirt"). Arms whose visible pixels are a sleeve
are left alone: in a rig the sleeve is the arm part.

What the edit models taught us (Salena FA, Oct 2026):
  - "plain underwear" alone gives a topless body, so the base layer names a
    top and briefs;
  - left/right confuses them (anatomical vs viewer), so limbs are always
    removed in pairs ("both of the character's hands");
  - removing arms from in front of a jacket needs the 9B model and the
    "as if the character had no arms ... sleeveless vest" wording;
  - removing a skirt from under a cropped jacket makes the jacket grow, so an
    upper garment is never peeled from under a lower one.

Pure numpy + OpenCV.
"""
import re

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from . import ownership

FACIAL_WORDS = ("eye", "iris", "pupil", "brow", "lash", "nose", "mouth", "lip", "teeth",
                "tooth", "tongue", "eyelid")
HAIR_WORDS = ("hair", "bang", "fringe", "ponytail", "braid", "pigtail")
GARMENT_WORDS = ("brace", "warmer", "sleeve", "glove", "boot", "shoe", "sock", "sash", "belt",
                 "skirt", "shirt", "top", "jacket", "coat", "vest", "dress", "pant", "trouser",
                 "short", "hat", "cap", "hood", "scarf", "cape", "cloak", "armor", "armour",
                 "strap", "bag", "collar", "cuff", "apron", "bra", "underwear", "necklace",
                 "earring", "bracelet", "glasses", "mask", "helmet", "tie", "ribbon", "bow",
                 "pouch", "holster", "sheath", "legging", "stocking", "tight", "robe", "tunic",
                 "sweater", "hoodie", "blouse", "corset", "kilt", "wrap", "band")
BODY_WORDS = ("torso", "chest", "body", "waist", "hip", "neck", "shoulder", "arm", "hand",
              "finger", "leg", "thigh", "shin", "calf", "knee", "foot", "feet", "toe", "face",
              "head", "ear")
HEAD_WORDS = ("face", "head")
LIMB_WORDS = ("arm", "hand", "finger", "leg", "foot", "feet")
BIG_LIMB_WORDS = ("arm", "leg")          # removing these needs the 9B model
LIMB_PLURAL = {"arm": "arms", "hand": "hands", "finger": "hands", "leg": "legs", "foot": "feet",
               "feet": "feet"}
UPPER_WORDS = ("jacket", "coat", "shirt", "top", "vest", "hoodie", "sweater", "blouse", "cardigan",
               "tunic", "robe", "bra", "corset", "cape", "cloak")
LOWER_WORDS = ("skirt", "pant", "trouser", "short", "kilt", "legging")

_ALL_COLOUR_WORDS = {
    "black", "grey", "gray", "white", "brown", "tan", "beige", "cream", "red", "pink", "orange",
    "yellow", "olive", "green", "teal", "cyan", "blue", "navy", "purple", "magenta", "light", "dark",
    "golden", "gold", "silver", "violet", "maroon", "khaki", "ivory", "mint"}
_SHADES = ("light", "dark")


def _words(tag):
    return re.findall(r"[a-z]+", tag.lower().replace("_", " "))


def category(tag):
    """facial | hair | garment | body for a part label."""
    ws = _words(tag)
    text = " ".join(ws)

    def has(words):
        return any(any(w.startswith(k) for w in ws) for k in words)
    if has(FACIAL_WORDS) and not has(("face", "head")):
        return "facial"
    if has(HAIR_WORDS) or "hair" in text:
        return "hair"
    if has(GARMENT_WORDS):
        return "garment"
    if has(BODY_WORDS):
        return "body"
    return "garment"  # an unknown label is most likely clothing or an accessory


def readable(tag):
    """'hair_front' -> 'front hair', 'left shin brace' stays."""
    t = tag.replace("_", " ").strip().lower()
    m = re.match(r"^(.*) (front|back|left|right)$", t)
    if m:
        t = "%s %s" % (m.group(2), m.group(1))
    return t


def _lab(rgb_pixels):
    """CIE Lab (L 0-100, a/b signed) of an (N, 3) uint8 array."""
    lab = cv2.cvtColor(rgb_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab = lab.reshape(-1, 3).astype(float)
    return np.column_stack([lab[:, 0] * 100.0 / 255.0, lab[:, 1] - 128.0, lab[:, 2] - 128.0])


def colour_name(lab):
    """A plain colour word for one Lab colour, from lightness, chroma and hue
    (so a muted dark brown is 'brown', not the nearest grey)."""
    L, a, b = (float(v) for v in lab)
    C = float(np.hypot(a, b))
    h = float(np.degrees(np.arctan2(b, a)) % 360)
    if C < 9:
        return ("black" if L < 18 else "dark grey" if L < 40 else "grey" if L < 65
                else "light grey" if L < 88 else "white")
    if 35 <= h < 105 and C < 38:             # warm and muted: browns and creams
        return ("dark brown" if L < 30 else "brown" if L < 55 else "tan" if L < 68
                else "beige" if L < 82 else "cream")
    if h < 50 or h >= 345:
        return "dark red" if L < 32 else "pink" if L > 70 else "red"
    if h < 80:
        return "brown" if L < 45 else "orange"
    if h < 100:
        return "olive" if L < 60 else "yellow"
    if h < 170:
        base = "green"
    elif h < 225:
        base = "teal" if L < 65 else "cyan"
    elif h < 305:
        if L < 25:
            return "navy"
        base = "blue"
    elif h < 335:
        base = "purple"
    else:
        return "pink" if L > 60 else "magenta"
    return ("dark " + base if L < 35 else "light " + base if L > 75 else base)


def colour_word(rgb_pixels, min_uniform=0.45):
    """The colour word for a part's pixels (their median), or None when the
    part is too many colours for one word to be right."""
    if rgb_pixels is None or len(rgb_pixels) == 0 or cv2 is None:
        return None
    px = rgb_pixels.reshape(-1, 3)
    lab_px = _lab(px)
    lab_med = _lab(np.median(px, axis=0).astype(np.uint8)[None])[0]
    if (np.linalg.norm(lab_px - lab_med, axis=1) < 30).mean() < min_uniform:
        return None
    return colour_name(lab_med)


def lightness(rgb_pixels):
    """Median Lab lightness (0-100) of a part's pixels."""
    if rgb_pixels is None or len(rgb_pixels) == 0 or cv2 is None:
        return None
    return float(np.median(_lab(rgb_pixels.reshape(-1, 3))[:, 0]))


def describe(tag, rgb_pixels=None):
    """'the brown jacket' (colour added unless the label already has one)."""
    name = readable(tag)
    if rgb_pixels is not None and not (set(_words(name)) & _ALL_COLOUR_WORDS):
        c = colour_word(rgb_pixels)
        if c:
            name = "%s %s" % (c, name)
    return "the " + name


def _hue_words(desc):
    return set(_words(desc)) & (_ALL_COLOUR_WORDS - set(_SHADES))


def shade_apart(desc, ref_desc, L, ref_L, min_dl=6.0):
    """When `desc` and `ref_desc` name the same colour, say which is lighter
    or darker: ('the green sash', 'the green skirt', 66, 58) -> 'the light
    green sash'. Only `desc` changes, and only if it has no shade yet."""
    if L is None or ref_L is None or set(_words(desc)) & set(_SHADES):
        return desc
    common = _hue_words(desc) & _hue_words(ref_desc)
    if not common or abs(L - ref_L) < min_dl:
        return desc
    shade = "light" if L > ref_L else "dark"
    word = next(w for w in _words(desc) if w in common)
    return re.sub(r"\b%s\b" % word, "%s %s" % (shade, word), desc, count=1)


def limb_word(tag):
    """'right hand' -> 'hand' (the limb a body label names), else None."""
    for w in _words(tag):
        for k in LIMB_WORDS:
            if w.startswith(k):
                return k
    return None


def limbs_phrase(tags):
    """Limbs are always named in pairs, without left/right (the edit models
    mix up anatomical and viewer sides): "both of the character's hands"."""
    plural = list(dict.fromkeys(LIMB_PLURAL[limb_word(t)] for t in tags if limb_word(t)))
    if not plural:
        return ""
    return "both of the character's " + " and ".join(plural)


def _has(tag, words):
    return any(w.startswith(k) for w in _words(tag) for k in words)


def join_names(names):
    """Merge left/right pairs into a plural and join with commas + 'and'."""
    names = list(dict.fromkeys(names))
    out, used = [], set()
    for n in names:
        if n in used:
            continue
        m = re.match(r"^the (.*?)(left|right) (.*)$", n)
        if m:
            other = "the %s%s %s" % (m.group(1), "right" if m.group(2) == "left" else "left", m.group(3))
            if other in names:
                used |= {n, other}
                out.append("both %s%ss" % (m.group(1), m.group(3).rstrip("s")))
                continue
        used.add(n)
        out.append(n)
    if len(out) <= 1:
        return "".join(out)
    return ", ".join(out[:-1]) + " and " + out[-1]


def is_bare(rgb, mask, skin_lab, max_de=38.0, min_share=0.4):
    """A body part whose visible pixels look like skin (median, or a good
    share of them, near the face's skin colour)."""
    if not mask.any() or skin_lab is None or cv2 is None:
        return False
    px = rgb[mask].reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(px, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
    de = np.linalg.norm(lab - skin_lab, axis=1)
    return bool(np.median(de) < max_de or (de < 25).mean() >= min_share)


def coverers(btf, owner, vis, i, expand=0.25, touch_frac=0.02, min_px=300):
    """Parts drawn in front of part i that cover where it could continue and
    touch what is visible of it."""
    name, m = btf[i]
    region, _ = ownership.hidden_candidates(btf, name, expand=expand, owner=owner)
    own = vis[name]
    if not own.any() or not region.any():
        return {}
    b = ownership.bbox(m)
    r = max(9, int(touch_frac * np.hypot(b[2] - b[0], b[3] - b[1])))
    near = cv2.dilate(own.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
    zone = region & near
    out = {}
    for k in np.unique(owner[zone]):
        if k > i:
            n = int((owner[zone] == k).sum())
            if n >= min_px:
                out[btf[k][0]] = n
    return out


KEEP = (" Keep everything else exactly the same: the same pose, position, size, proportions,"
        " colours, line style and flat cartoon shading, and the same plain background.")
BASE_LAYER = "a simple plain sports top and simple plain briefs"
NO_ARMS = ("Remove both of the character's arms completely, from the shoulders down, together with"
           " the sleeves, cuffs and hands, as if the character had no arms.")


def garment_prompt(target, front, limbs, arms_off):
    """The edit instruction for a garment peel.

    target    'the green skirt'
    front     garments / hair in front of it (already described)
    limbs     body limb tags in front of it
    arms_off  the target is an upper garment behind the arms: use the
              'no arms, sleeveless vest' wording that the 9B model follows"""
    if arms_off:
        head = ("Remove %s from this picture. " % join_names(front)) if front else ""
        return (head + NO_ARMS + " Show %s as a sleeveless vest, so its whole front and both sides"
                " are visible and complete." % target) + KEEP
    parts = []
    if front:
        parts.append("Remove %s that %s over %s" % (join_names(front), "lies" if len(front) == 1 else "lie",
                                                   target))
    if limbs:
        parts.append(("remove " if parts else "Remove ") + limbs_phrase(limbs))
    return (" and ".join(parts) + ", so that %s is fully visible and complete." % target) + KEEP


def plan(btf, rgb, facial=(), base_layer=BASE_LAYER, min_hidden=0.05, garment_peels=True):
    """Peel jobs for a split.

    btf     [(tag, mask)] back to front, full-picture masks
    rgb     the picture (HxWx3 uint8)
    facial  tags of the facial pass (never peel targets)
    Returns a list of jobs: {name, kind, prompt, model, sam_labels, parts,
    anchors?, colour?} plus rule entries ({"hull": ...}) in tools/peel_merge.py
    plan format (paths are added by the caller)."""
    owner, names = ownership.owner_map(btf)
    vis = {n: owner == i for i, n in enumerate(names)}
    cats = {n: ("facial" if n in facial else category(n)) for n in names}
    # colour words only for clothing and hair: "the brown right hand" misleads
    desc = {n: describe(n, rgb[vis[n]] if vis[n].any() and cats[n] in ("garment", "hair") else None)
            for n in names}
    light = {n: lightness(rgb[vis[n]]) if vis[n].any() else None for n in names}

    heads = [n for n in names if cats[n] == "body" and any(w in _words(n) for w in HEAD_WORDS)]
    skin_lab = None
    if heads and vis[heads[0]].any() and cv2 is not None:
        px = rgb[vis[heads[0]]].reshape(-1, 1, 3).astype(np.uint8)
        skin_lab = np.median(cv2.cvtColor(px, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float), axis=0)

    cov, hidden = {}, {}
    for i, n in enumerate(names):
        if cats[n] in ("facial",):
            continue
        cov[n] = coverers(btf, owner, vis, i)
        hidden[n] = sum(cov[n].values()) / max(1, int(btf[i][1].sum()))

    jobs = []

    # 1. base body: bare body parts under clothes
    bare = [n for n in names if cats[n] == "body" and n not in heads
            and is_bare(rgb, vis[n], skin_lab) and cov.get(n)]
    clothes = []
    for n in bare:
        clothes += [c for c in cov[n] if cats[c] == "garment"]
    clothes = list(dict.fromkeys(clothes))
    bare = [n for n in bare if any(cats[c] == "garment" for c in cov[n])]
    if bare and clothes:
        parts = {n: [n] for n in bare}
        sam = list(bare)
        torso = [n for n in bare if any(w in _words(n) for w in ("torso", "chest", "body"))]
        if torso:
            extra = ["underwear", "sports top", "bra", "briefs", "bikini bottom", "neck"]
            parts[torso[0]] = [torso[0]] + extra
            sam += extra
        jobs.append({
            "name": "base_body", "kind": "base body",
            "prompt": ("Edit this 2D character illustration. Remove %s, so that the character's"
                       " plain base body is visible, wearing only %s. Keep exactly the same pose,"
                       " position, size, proportions, face, hair, skin tone, line style and flat"
                       " cartoon shading, and the same plain background."
                       % (join_names([desc[c] for c in clothes]), base_layer)),
            "model": "4b", "sam_labels": sam, "parts": parts,
            "anchors": [n for n in names if cats[n] == "body" and (n in heads or is_bare(rgb, vis[n], skin_lab))],
            "colour": "global",
            "targets": bare, "removes": clothes,
        })

    # 2. garments covered by other parts
    if garment_peels:
        for n in names:
            if cats[n] != "garment" or hidden.get(n, 0) < min_hidden:
                continue
            upper = _has(n, UPPER_WORDS)
            front = [c for c in cov[n] if cats[c] != "facial"
                     # a cropped top tucked behind a skirt: removing the skirt makes it grow
                     and not (upper and cats[c] == "garment" and _has(c, LOWER_WORDS))]
            if not front:
                continue
            limbs = [c for c in front if cats[c] == "body" and limb_word(c)]
            cloth = [shade_apart(desc[c], desc[n], light[c], light[n])
                     for c in front if c not in limbs and cats[c] in ("garment", "hair")]
            arms_off = upper and any(limb_word(c) == "arm" for c in limbs)
            big = any(limb_word(c) in BIG_LIMB_WORDS for c in limbs)
            word = re.sub(r"\W+", "_", n.strip().lower())
            labels = list(dict.fromkeys([n, readable(n), desc[n][4:]] + (["vest"] if arms_off else [])))
            jobs.append({
                "name": "garment_" + word, "kind": "garment",
                "prompt": garment_prompt(desc[n], cloth, limbs, arms_off),
                "model": "9b" if big else "4b",
                "sam_labels": labels, "parts": {n: labels},
                "targets": [n], "removes": front,
            })

    # 3. head under hair
    for h in heads:
        hair_front = [c for c in cov.get(h, {}) if cats[c] == "hair"]
        if hair_front and hidden.get(h, 0) >= min_hidden:
            jobs.append({
                "name": "head", "kind": "head",
                "prompt": ("Remove all of the character's hair, so that the whole head is visible:"
                           " the face, the forehead up to the top of the skull, both ears and the"
                           " neck.") + KEEP,
                "model": "4b", "sam_labels": ["head", "face", "left ear", "right ear"],
                "parts": {h: ["head", "face", "left ear", "right ear"]},
                "targets": [h], "removes": hair_front,
            })

    # 4. back hair by rule
    hair = [n for n in names if cats[n] == "hair"]
    for n in hair:
        if "back" in _words(n) or n.lower().startswith("back"):
            jobs.append({"hull": {"tag": n, "group": hair, "mode": "convex", "colour": "shadow"}})
    return jobs
