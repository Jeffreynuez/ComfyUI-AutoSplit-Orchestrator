"""
Mask refinement helpers for the SAM3 AutoSplit orchestrator.

Kept in a separate module so the logic can be compiled/checked independently
of the large nodes_sam3.py. Imported lazily (inside a try/except) by the
orchestrator, so any failure degrades gracefully to the un-refined mask.

Both functions take and return torch tensors and never raise: on any error
they return the input mask unchanged.
"""


def color_grow(mask_2d, image_tensor, tol=0.10, dilate_px=10):
    """
    Grow a part mask to include pixels whose colour matches the part's median
    colour AND are connected to the existing mask, within a slightly dilated
    bounding box. Recovers thin skin slivers that peek between occluding straps.

    Returns a torch float tensor [H,W] on the input's device/dtype, or the
    input mask unchanged on any error.
    """
    import torch
    try:
        import numpy as np
        import cv2
        device = mask_2d.device
        dtype = mask_2d.dtype
        m = (mask_2d.detach().cpu().numpy() > 0.5)
        if not m.any():
            return mask_2d
        H, W = m.shape

        img = image_tensor
        while hasattr(img, "dim") and img.dim() > 3:
            img = img[0]
        img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[2]:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[2] > 3:
            img = img[:, :, :3]
        if img.shape[0] != H or img.shape[1] != W:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)

        ys, xs = np.where(m)
        y0 = max(0, int(ys.min()) - dilate_px); y1 = min(H - 1, int(ys.max()) + dilate_px)
        x0 = max(0, int(xs.min()) - dilate_px); x1 = min(W - 1, int(xs.max()) + dilate_px)
        bbox = np.zeros_like(m); bbox[y0:y1 + 1, x0:x1 + 1] = True

        med = np.median(img[m], axis=0)
        dist = np.sqrt(((img - med[None, None, :]) ** 2).sum(axis=2))
        similar = (dist < tol) & bbox

        seed = (m | similar).astype(np.uint8)
        num, labels = cv2.connectedComponents(seed, connectivity=8)
        keep_labels = set(np.unique(labels[m]).tolist())
        keep_labels.discard(0)
        grown = np.zeros_like(m)
        for lbl in keep_labels:
            grown |= (labels == lbl)
        grown = m | (grown & similar)

        return torch.from_numpy(grown.astype(np.float32)).to(device=device, dtype=dtype)
    except Exception as e:
        print("[AutoSplit-SAM3] color_grow skipped (%s)" % e)
        return mask_2d


def complete_occluded(mask_2d, other_masks, current_idx, min_island_frac=0.02, bbox_pad=4):
    """
    If a part is fragmented by an occluder sitting on top of it (e.g. a shin
    brace across a leg), bridge the fragments into one solid silhouette by
    adding the occluder pixels that fall within the part's bounding box.

    Only acts when the part has 2+ significant connected components (i.e. it is
    actually fragmented), so clean single-piece parts are left untouched.

    Returns (completed_mask_tensor, added_region_tensor_or_None); on any error
    returns (mask_2d, None).
    """
    import torch
    try:
        import numpy as np
        import cv2
        device = mask_2d.device
        dtype = mask_2d.dtype
        m = (mask_2d.detach().cpu().numpy() > 0.5).astype(np.uint8)
        if m.sum() == 0:
            return mask_2d, None
        H, W = m.shape

        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num <= 1:
            return mask_2d, None
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            return mask_2d, None
        max_area = float(areas.max())
        sig = int((areas >= min_island_frac * max_area).sum())
        if sig <= 1:
            return mask_2d, None

        occ = np.zeros((H, W), np.uint8)
        for j, om in enumerate(other_masks):
            if j == current_idx or om is None:
                continue
            o = om
            while hasattr(o, "dim") and o.dim() > 2:
                o = o.squeeze(0)
            o = (o.detach().cpu().numpy() > 0.5).astype(np.uint8)
            if o.shape != (H, W):
                o = cv2.resize(o, (W, H), interpolation=cv2.INTER_NEAREST)
            occ |= o
        if occ.sum() == 0:
            return mask_2d, None

        ys, xs = np.where(m > 0)
        y0 = max(0, int(ys.min()) - bbox_pad); y1 = min(H - 1, int(ys.max()) + bbox_pad)
        x0 = max(0, int(xs.min()) - bbox_pad); x1 = min(W - 1, int(xs.max()) + bbox_pad)
        bbox = np.zeros((H, W), np.uint8); bbox[y0:y1 + 1, x0:x1 + 1] = 1
        bridge = ((occ > 0) & (bbox > 0) & (m == 0)).astype(np.uint8)
        if bridge.sum() == 0:
            return mask_2d, None

        completed = ((m > 0) | (bridge > 0)).astype(np.float32)
        comp_t = torch.from_numpy(completed).to(device=device, dtype=dtype)
        added_t = torch.from_numpy(bridge.astype(np.float32)).to(device=device, dtype=dtype)
        return comp_t, added_t
    except Exception as e:
        print("[AutoSplit-SAM3] complete_occluded skipped (%s)" % e)
        return mask_2d, None
