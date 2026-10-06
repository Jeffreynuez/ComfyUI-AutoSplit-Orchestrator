"""
Pixel ownership and where a part may continue behind others.

Every visible pixel of the picture belongs to exactly one part: the front-most
part (in draw order) whose mask covers it. A part's *hidden candidate* region
is where it could legitimately continue out of sight: pixels owned by parts in
front of it, near the part. Hidden-area fill paints only there, because
anything outside that region would have been visible in the picture.

Pure numpy; masks are HxW bool arrays in one shared frame.
"""
import numpy as np


def nested_pairs(masks_btf, inside=0.6, bigger=1.3):
    """(outer, inner) index pairs where the smaller mask lies mostly inside
    the bigger one: |A & B| >= inside * |B| and |A| >= bigger * |B|.

    SAM 3 only marks pixels it can see, so one visible mask sitting inside
    another is not an occlusion: it is a sub-part the bigger label swallowed
    (the sleeves inside "jacket", the sash inside "skirt", the eyes inside
    "face"). The sub-part owns those pixels whatever the draw order says."""
    areas = [int(m.sum()) for _, m in masks_btf]
    pairs = []
    for a, (_, ma) in enumerate(masks_btf):
        for b, (_, mb) in enumerate(masks_btf):
            if a == b or areas[b] == 0 or areas[a] < bigger * areas[b]:
                continue
            if (ma & mb).sum() >= inside * areas[b]:
                pairs.append((a, b))
    return pairs


def owner_map(masks_btf, nested=True):
    """masks_btf: list of (name, mask) back to front.
    Returns (owner, names): owner[y, x] = index into names of the part that
    owns the pixel, -1 where no part does. The front-most part wins, except
    that a nested sub-part (see nested_pairs) keeps its pixels from the part
    that contains it."""
    names = [n for n, _ in masks_btf]
    if not masks_btf:
        return None, names
    owner = np.full(masks_btf[0][1].shape, -1, np.int32)
    for i, (_, m) in enumerate(masks_btf):
        owner[m] = i
    if nested:
        for a, b in nested_pairs(masks_btf):
            mb = masks_btf[b][1]
            owner[mb & (owner == a)] = b
    return owner, names


def visible(masks_btf, nested=True):
    """Per part, the pixels it owns. Returns {name: mask}."""
    owner, names = owner_map(masks_btf, nested)
    return {n: owner == i for i, n in enumerate(names)}


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_box(box, frac, shape, min_px=16):
    x0, y0, x1, y1 = box
    dx = max(min_px, int(round((x1 - x0) * frac)))
    dy = max(min_px, int(round((y1 - y0) * frac)))
    H, W = shape
    return max(0, x0 - dx), max(0, y0 - dy), min(W, x1 + dx), min(H, y1 + dy)


def hidden_candidates(masks_btf, name, expand=0.35, box=None):
    """Where `name` may continue behind the parts in front of it.

    The region is the pixels owned by parts drawn in front of `name`, limited
    to the part's bounding box grown by `expand` of its size (or to `box`,
    x0, y0, x1, y1, when the caller knows better, e.g. a body template).
    Returns (region, front_names): front_names are the parts that own pixels
    inside the region, i.e. the ones actually covering it."""
    names = [n for n, _ in masks_btf]
    i = names.index(name)
    owner, _ = owner_map(masks_btf)
    own = owner == i
    if box is None:
        b = bbox(masks_btf[i][1])
        if b is None:
            return np.zeros_like(own), []
        box = expand_box(b, expand, own.shape)
    x0, y0, x1, y1 = box
    lim = np.zeros_like(own)
    lim[y0:y1, x0:x1] = True
    # only parts drawn in FRONT: paint added behind a sub-part that is drawn
    # behind this part would cover that sub-part in the rest pose
    region = lim & (owner > i)
    front = sorted({names[k] for k in np.unique(owner[region]) if k > i})
    return region, front


def hidden_share(masks_btf, name):
    """Fraction of a part's own mask that parts in front of it cover."""
    names = [n for n, _ in masks_btf]
    i = names.index(name)
    m = masks_btf[i][1]
    if not m.any():
        return 0.0
    owner, _ = owner_map(masks_btf)
    return float((m & (owner > i)).sum() / m.sum())
