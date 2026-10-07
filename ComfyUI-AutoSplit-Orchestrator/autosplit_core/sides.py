"""
Left/right for bilateral parts.

SAM 3 is not reliable about the word "left": a prompt like "left hand" can
come back with either hand, and both prompts can return the SAME hand. So we
never take the side from the prompt. We collect every instance SAM 3 found
for the pair, keep two distinct ones, and name them from their position in the
picture under an explicit convention:

    anatomical (default, rigger convention): the character's own left. For a
        front-facing character that is the part on the VIEWER'S RIGHT.
    viewer: the part on the viewer's left is "left".

`facing="back"` flips the anatomical mapping for back views (turnarounds).
"""
from . import masks as M


def distinct_instances(instances, iou_dup=0.5):
    """instances: list of (mask, score). Drop near-duplicates (IoU > iou_dup),
    keeping the higher score. Returns the survivors, best score first."""
    keep = []
    for m, s in sorted(instances, key=lambda t: -t[1]):
        if all(M.iou(m, k[0]) <= iou_dup for k in keep):
            keep.append((m, s))
    return keep


def pick_pair(instances, min_area_ratio=0.2, iou_dup=0.5):
    """Choose two instances that look like a left/right pair: distinct, and not
    wildly different in size. Returns [(mask, score), (mask, score)] or the
    single best one when no pair exists."""
    inst = distinct_instances(instances, iou_dup)
    if len(inst) < 2:
        return inst
    best = inst[0]
    a0 = max(1, int(best[0].sum()))
    for other in inst[1:]:
        a1 = int(other[0].sum())
        ratio = min(a0, a1) / float(max(a0, a1))
        if ratio >= min_area_ratio:
            return [best, other]
    return [best]


def assign(mask_a, mask_b, convention="anatomical", facing="front"):
    """Name two masks. Returns (left, right, info)."""
    ca, cb = M.centroid(mask_a), M.centroid(mask_b)
    viewer_left, viewer_right = (mask_a, mask_b) if ca[0] <= cb[0] else (mask_b, mask_a)
    mirrored = convention == "anatomical" and facing != "back"
    if mirrored:
        left, right = viewer_right, viewer_left
    else:
        left, right = viewer_left, viewer_right
    return left, right, {"lr_source": "geometry", "convention": convention,
                         "facing": facing, "mirrored": mirrored}


def split_union(mask, convention="anatomical", facing="front", min_area=50):
    """Fallback when SAM 3 returned one mask covering both sides: split it into
    its two largest connected components and name them."""
    comps = M.components(mask, min_area=min_area)
    if len(comps) < 2:
        return None
    left, right, info = assign(comps[0], comps[1], convention, facing)
    info["lr_source"] = "components"
    return left, right, info
