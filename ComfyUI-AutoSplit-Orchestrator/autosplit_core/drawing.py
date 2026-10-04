"""
One draw order across the body pass and the facial pass (plain Python).

The facial pass runs on an upscaled head crop, so it orders its parts among
themselves only. The June exporters then drew every facial part above every
body part, which puts eyebrows over bangs. The right place for facial features
is just in front of the face they sit on: in front of "face"/"head", behind
anything the body pass put in front of the face (bangs, a hat, a raised hand).
"""

ANCHORS = ("face", "head")


def _tag(e):
    return str(e.get("tag", "")).strip().lower().replace("_", " ")


def merge(body_btf, facial_btf, anchors=ANCHORS):
    """body_btf, facial_btf: lists of part entries, back-most first.
    Returns one list, back-most first, with the facial parts inserted
    directly in front of the first anchor found in the body list (or on top
    when there is no anchor)."""
    body = list(body_btf)
    for a in anchors:
        for i, e in enumerate(body):
            if _tag(e) == a:
                return body[: i + 1] + list(facial_btf) + body[i + 1:]
    return body + list(facial_btf)


def back_to_front(entries):
    """Sort one pass's entries back-most first. Prefers draw_index (schema 2,
    0 = back), then z_order (0 = front)."""
    if entries and all(e.get("draw_index") is not None for e in entries):
        return sorted(entries, key=lambda e: e["draw_index"])
    return sorted(entries, key=lambda e: -float(e.get("z_order", e.get("z_rank", 0)) or 0))
