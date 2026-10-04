"""Mask helpers (numpy + OpenCV). Masks are 2-D bool or {0,1} arrays."""
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - ComfyUI always ships OpenCV
    cv2 = None


def as_bool(m):
    m = np.asarray(m)
    while m.ndim > 2:
        m = m[0]
    return m > 0.5 if m.dtype != bool else m


def fill_holes(mask):
    """Fill fully enclosed holes. A part with a strap or brace across it comes
    back from SAM 3 with a hole where the strap is; filling it gives the part's
    outline, which is what draw-order evidence needs."""
    m = as_bool(mask)
    if cv2 is None or not m.any():
        return m.copy()
    padded = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
    padded[1:-1, 1:-1] = m
    ff = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, ff, (0, 0), 2)          # 2 = reachable from outside
    return padded[1:-1, 1:-1] != 2


def clean(mask, close_ksize=5, island_min_frac=0.02, fill=True):
    """Morphological close, optional hole fill, drop islands smaller than
    island_min_frac of the largest component. Same recipe the June node used."""
    m = as_bool(mask)
    if cv2 is None or not m.any():
        return m.copy()
    u8 = m.astype(np.uint8) * 255
    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    m = u8 > 0
    if fill:
        m = fill_holes(m)
    return drop_islands(m, island_min_frac)


def drop_islands(mask, min_frac=0.02):
    m = as_bool(mask)
    if cv2 is None or not m.any():
        return m.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m.copy()
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_ids = 1 + np.nonzero(areas >= min_frac * areas.max())[0]
    return np.isin(labels, keep_ids)


def components(mask, min_area=50):
    """Connected components, largest first, as bool masks."""
    m = as_bool(mask)
    if cv2 is None or not m.any():
        return []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    ids.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    return [labels == i for i in ids]


def iou(a, b):
    a, b = as_bool(a), as_bool(b)
    u = np.logical_or(a, b).sum()
    return 0.0 if u == 0 else float(np.logical_and(a, b).sum()) / float(u)


def centroid(mask):
    ys, xs = np.nonzero(as_bool(mask))
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def resize_to(arr, shape, nearest=False):
    """Resize a 2-D array to (H, W)."""
    if arr.shape[:2] == tuple(shape):
        return arr
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    out = cv2.resize(arr.astype(np.float32), (shape[1], shape[0]), interpolation=interp)
    return out


def depth_2d(depth):
    """Any IMAGE-like depth (tensor or array, [B,H,W,C]/[H,W,C]/[H,W]) to a 2-D
    float32 array."""
    d = depth
    if hasattr(d, "detach"):
        d = d.detach().cpu().numpy()
    d = np.asarray(d, dtype=np.float32)
    while d.ndim > 3:
        d = d[0]
    if d.ndim == 3:
        d = d[..., 0]
    return d


def nearness(depth, near="bright"):
    """Return a 2-D map where HIGHER means CLOSER to the viewer, whatever the
    depth model's convention. Marigold writes near = dark; Depth Anything
    writes near = bright (it predicts disparity)."""
    d = depth_2d(depth)
    return d if near == "bright" else (d.max() - d)


def split_by_depth(mask, near_map, min_share=0.15, min_gap=0.04):
    """Split one mask into (front, back) by Otsu on its nearness values.
    Used for hair, where bangs and back hair are usually one connected blob.
    Returns None when the two groups are too small or too close in depth."""
    m = as_bool(mask)
    if cv2 is None or m.sum() < 200:
        return None
    nm = near_map if near_map.shape == m.shape else resize_to(near_map, m.shape)
    vals = nm[m]
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-6:
        return None
    scaled = ((vals - lo) / (hi - lo) * 255).astype(np.uint8)
    t, _ = cv2.threshold(scaled.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = lo + (t / 255.0) * (hi - lo)
    front = m & (nm > thr)
    back = m & ~front
    share = front.sum() / float(m.sum())
    if share < min_share or share > 1 - min_share:
        return None
    if float(nm[front].mean() - nm[back].mean()) < min_gap * (hi - lo):
        return None
    front, back = drop_islands(front, 0.05), drop_islands(back, 0.05)
    return front, back
