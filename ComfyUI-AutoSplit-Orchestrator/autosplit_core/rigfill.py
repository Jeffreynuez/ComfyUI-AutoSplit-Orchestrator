"""
Rule-based hidden-area fills that follow rigging conventions, not the
picture: what a rigger paints behind other parts so the puppet holds up when
it moves.

  underlap     a limb continues under the part it joins, toward the joint:
               the thigh up under the briefs to the hip, the upper arm in
               under the torso / jacket at the shoulder, the neck up behind
               the head. The child's edge where it meets the parent is swept
               toward the parent by a share of the child's thickness.
  back panel   an open or wrap-around garment gets a separate back layer
               behind the body (jacket back, skirt back), inside its own
               outline, painted in the garment's deep interior shadow.

Lengths and colours were read off Jeffrey's hand-cut Salena FA rig (Oct 2026):
thighs reach ~0.6 of the leg's thickness above the briefs line, the far
sleeve ~0.35 of the arm's thickness behind the jacket, the neck ~0.5 of the
torso's thickness up behind the head, and garment backs are near black
(the garment's colour at a quarter of its lightness, little chroma).

Pure numpy + OpenCV.
"""
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def thickness(mask):
    """Twice the largest inscribed radius: how wide a limb is."""
    if not mask.any():
        return 0.0
    d = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)
    return 2.0 * float(d.max())


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else None


def _crop_box(masks, pad, shape):
    H, W = shape
    m = np.zeros(shape, bool)
    for x in masks:
        m |= x
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    return (max(0, xs.min() - pad), max(0, ys.min() - pad),
            min(W, xs.max() + 1 + pad), min(H, ys.max() + 1 + pad))


def underlap(child, parent, allowed, length_frac, touch_px=4, step=3):
    """Pixels where `child` continues under `parent`.

    child, parent  full-canvas bool masks (after any fill)
    allowed        where the child may go (hidden behind parts in front of it)
    length_frac    sweep length as a share of the child's thickness

    The child's pixels touching the parent are swept along the direction from
    the child's centroid to the parent's, up to the length, and kept inside
    the parent and `allowed`."""
    if cv2 is None or not child.any() or not parent.any():
        return np.zeros_like(child)
    L = length_frac * thickness(child)
    if L < 2:
        return np.zeros_like(child)
    box = _crop_box([child, parent], int(L) + 8, child.shape)
    x0, y0, x1, y1 = box
    c, p, a = child[y0:y1, x0:x1], parent[y0:y1, x0:x1], allowed[y0:y1, x0:x1]
    k = 2 * touch_px + 1
    contact = c & cv2.dilate(p.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    if not contact.any():
        return np.zeros_like(child)
    v = centroid(p) - centroid(c)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.zeros_like(child)
    v /= n
    h, w = c.shape
    stub = np.zeros_like(c)
    ys, xs = np.nonzero(contact)
    for t in np.arange(step, L + step, step):
        yy = np.round(ys + v[1] * t).astype(int)
        xx = np.round(xs + v[0] * t).astype(int)
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        stub[yy[ok], xx[ok]] = True
    # close the gaps a rounded sweep leaves, then keep it inside the parent
    stub = cv2.morphologyEx(stub.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    stub &= p & a & ~c
    out = np.zeros_like(child)
    out[y0:y1, x0:x1] = stub
    return out


def nearest_colours(rgb, src, target, inset=6):
    """Colours for `target` pixels copied from the nearest `src` pixel, taken a
    few pixels inside the source so edge lines and cast shadows are not
    smeared across the fill. Returns an (N, 3) uint8 array in target order."""
    if not target.any():
        return np.zeros((0, 3), np.uint8)
    box = _crop_box([src, target], 2, src.shape)
    x0, y0, x1, y1 = box
    s = src[y0:y1, x0:x1]
    if inset:
        k = 2 * inset + 1
        inner = cv2.erode(s.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        if inner.sum() >= 0.2 * s.sum():
            s = inner
    if not s.any():
        return np.zeros((int(target.sum()), 3), np.uint8)
    _, lab = cv2.distanceTransformWithLabels((~s).astype(np.uint8), cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    sy, sx = np.nonzero(s)
    # label ids of the source pixels, in the same raster order distanceTransform uses
    lut = np.zeros(lab.max() + 1, np.int64)
    lut[lab[sy, sx]] = np.arange(len(sy))
    ty, tx = np.nonzero(target[y0:y1, x0:x1])
    k = lut[lab[ty, tx]]
    return rgb[y0:y1, x0:x1][sy[k], sx[k]]


def main_islands(mask, min_share=0.1):
    """The mask without islands smaller than `min_share` of its largest one
    (stray cuffs or specks left by ownership)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask.copy()
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= min_share * areas.max()]
    return np.isin(labels, keep)


def back_panel(garment, min_share=0.25):
    """Inside the garment's convex outline but not the garment itself: where
    its back panel shows between and around the front panels. Islands under a
    quarter of the largest (a cuff the split gave the jacket) are left out of
    the outline."""
    g = main_islands(garment, min_share)
    pts = cv2.findNonZero(g.astype(np.uint8))
    if pts is None:
        return np.zeros_like(garment)
    hull = np.zeros(garment.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    return hull.astype(bool) & ~garment


def interior_colour(rgb_pixels, lightness=0.25, chroma=0.3):
    """The garment's deep interior shadow: its median colour at a share of the
    lightness and chroma (Salena's jacket and skirt backs are near black)."""
    lab = cv2.cvtColor(rgb_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab = np.median(lab.reshape(-1, 3).astype(float), axis=0)
    lab[0] *= lightness
    lab[1:] = 128 + (lab[1:] - 128) * chroma
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2RGB)
    return out.reshape(3)


def picture_silhouette(rgb, tol=8.0, border=8):
    """The character's pixels in a picture on a plain background: everything
    more than `tol` (Lab) from the background colour, read off the picture's
    border, with enclosed holes filled. Lets a fill claim visible pixels the
    split left to no part (a neck between face and torso)."""
    H, W = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    edge = np.concatenate([lab[:border].reshape(-1, 3), lab[-border:].reshape(-1, 3),
                           lab[:, :border].reshape(-1, 3), lab[:, -border:].reshape(-1, 3)])
    bg = np.median(edge, axis=0)
    fg = (np.linalg.norm(lab - bg, axis=2) > tol).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ff = np.pad(fg, 1)
    cv2.floodFill(ff, None, (0, 0), 2)
    return ff[1:-1, 1:-1] != 2
