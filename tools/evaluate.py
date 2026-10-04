"""
Score an AutoSplit run against a hand-cut ground truth (see gt_from_psd.py).

What it measures, per run:
  * per part: IoU, precision, recall and boundary F-score (3 px tolerance)
    against the part's VISIBLE region in the hand-cut rig;
  * left/right: whether a part labelled left actually overlaps the rig's left
    part more than its right one (anatomical convention, as the rigger named it);
  * draw order: for every pair of predicted parts whose whole (amodal) shapes
    overlap in the rig, does the run put the right one in front;
  * coverage: how much of the character the parts cover, and how much is
    claimed by more than one part;
  * rest reconstruction: composite the parts back-to-front and compare with
    the source picture (visible-only cuts should rebuild it almost exactly).

Usage:
    python tools/evaluate.py --gt <gt folder> --map tools/gt_maps/salena_fa.json \
        --run <body pass folder> [--run <facial pass folder>] [--json report.json]

Needs numpy, Pillow and OpenCV.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from partsio import GroundTruth, back_to_front, composite, load_pass  # noqa: E402

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

LEFT_RE = re.compile(r"^(left[ _])|([ _]left)$")
RIGHT_RE = re.compile(r"^(right[ _])|([ _]right)$")


def norm_tag(tag, mapping, aliases):
    t = tag.strip().lower()
    for cand in (t, aliases.get(t), t.replace("_", " "), aliases.get(t.replace("_", " "))):
        if cand and cand in mapping:
            return cand
    return None


def side(tag):
    t = tag.lower()
    if LEFT_RE.search(t):
        return "left"
    if RIGHT_RE.search(t):
        return "right"
    return None


def mirror(tag):
    t = tag.lower()
    if "left" in t:
        return t.replace("left", "right")
    if "right" in t:
        return t.replace("right", "left")
    return t


def boundary(mask):
    k = np.ones((3, 3), np.uint8)
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, k)).astype(bool)


def boundary_f(pred, gt, tol=3):
    if cv2 is None or not pred.any() or not gt.any():
        return 0.0
    bp, bg = boundary(pred), boundary(gt)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    bg_d = cv2.dilate(bg.astype(np.uint8), k).astype(bool)
    bp_d = cv2.dilate(bp.astype(np.uint8), k).astype(bool)
    p = (bp & bg_d).sum() / max(1, bp.sum())
    r = (bg & bp_d).sum() / max(1, bg.sum())
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def iou(a, b):
    u = (a | b).sum()
    return 0.0 if u == 0 else (a & b).sum() / u


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--run", action="append", required=True,
                    help="a pass folder holding parts_metadata.json (repeat for the facial pass)")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--min-overlap", type=int, default=150,
                    help="pixels of whole-shape overlap before a pair's order counts")
    args = ap.parse_args()

    gt = GroundTruth(args.gt)
    W, H = gt.W, gt.H
    with open(args.map, encoding="utf-8") as f:
        mp = json.load(f)
    mapping = {k.lower(): v for k, v in mp["parts"].items()}
    aliases = {k.lower(): v.lower() for k, v in mp.get("aliases", {}).items()}

    # ---- ground-truth groups -------------------------------------------
    solid = [l for l in gt.layers if not l.get("is_overlay")]
    groups = {}
    for tag, pats in mapping.items():
        rx = [re.compile(p, re.I) for p in pats]
        groups[tag] = [l for l in solid if any(r.fullmatch(l["name"].strip()) for r in rx)]
    top = gt.top_index_map()
    sil = top >= 0

    def gt_visible(tag):
        idx = [l["index"] for l in groups[tag]]
        return np.isin(top, idx) if idx else np.zeros_like(sil)

    def gt_group_top(tag):
        m = np.full((H, W), -1, np.int32)
        for l in groups[tag]:
            m[gt.layer_mask(l)] = l["index"]
        return m

    # ---- predicted parts -----------------------------------------------
    parts = []
    for folder in args.run:
        parts += load_pass(folder)
    order = back_to_front(parts)
    pos = {id(p): i for i, p in enumerate(order)}

    report = {"parts": [], "unmapped": [], "missing": [], "lr": [], "order": {}}
    pred = {}
    for p in parts:
        key = norm_tag(p.tag, mapping, aliases)
        if key is None:
            report["unmapped"].append(p.tag)
            continue
        pred.setdefault(key, []).append(p)

    for tag in mapping:
        if tag in ("hair_front", "hair_back", "mouth") and tag not in pred:
            continue  # optional splits
        if tag == "hair" and ("hair_front" in pred or "hair_back" in pred) and tag not in pred:
            continue
        if tag not in pred and groups[tag]:
            report["missing"].append(tag)

    masks = {}
    for tag, plist in pred.items():
        m = np.zeros((H, W), bool)
        for p in plist:
            m |= p.full_mask(W, H)
        masks[tag] = m
        g = gt_visible(tag)
        inter = (m & g).sum()
        row = {
            "tag": tag,
            "iou": round(float(iou(m, g)), 4),
            "precision": round(float(inter / max(1, m.sum())), 4),
            "recall": round(float(inter / max(1, g.sum())), 4),
            "boundary_f": round(float(boundary_f(m, g)), 4),
            "pred_px": int(m.sum()), "gt_px": int(g.sum()),
        }
        report["parts"].append(row)
        s = side(tag)
        if s and mirror(tag) in mapping:
            same, other = row["iou"], float(iou(m, gt_visible(mirror(tag))))
            report["lr"].append({"tag": tag, "iou_same_side": same,
                                 "iou_other_side": round(other, 4),
                                 "correct": same >= other})

    # ---- draw order ----------------------------------------------------
    tops = {t: gt_group_top(t) for t in pred}
    pairs, agree, weight_ok, weight_all, ambiguous = [], 0, 0, 0, 0
    tags = sorted(pred)
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            la = {l["index"] for l in groups[a]}
            lb = {l["index"] for l in groups[b]}
            if not la or not lb or la & lb:
                continue
            ta, tb = tops[a], tops[b]
            ov = (ta >= 0) & (tb >= 0)
            n = int(ov.sum())
            if n < args.min_overlap:
                continue
            frac_a = float((ta[ov] > tb[ov]).mean())
            if 0.2 < frac_a < 0.8:
                ambiguous += 1
                continue
            gt_a_front = frac_a >= 0.5
            pa = max(pos[id(p)] for p in pred[a])
            pb = max(pos[id(p)] for p in pred[b])
            pred_a_front = pa > pb
            ok = gt_a_front == pred_a_front
            agree += ok
            weight_all += n
            weight_ok += n if ok else 0
            pairs.append({"a": a, "b": b, "overlap_px": n,
                          "gt_front": a if gt_a_front else b,
                          "pred_front": a if pred_a_front else b, "correct": ok})
    report["order"] = {
        "pairs_scored": len(pairs), "ambiguous_pairs_skipped": ambiguous,
        "pairwise_accuracy": round(agree / len(pairs), 4) if pairs else None,
        "overlap_weighted_accuracy": round(weight_ok / weight_all, 4) if weight_all else None,
        "wrong": [p for p in pairs if not p["correct"]],
    }

    # ---- coverage + rest reconstruction --------------------------------
    claim = np.zeros((H, W), np.int16)
    for p in parts:
        claim += p.full_mask(W, H)
    union = claim > 0
    rest = np.asarray(composite(order, W, H)).astype(np.int16)
    src = gt.composite().astype(np.int16)
    diff = np.abs(rest[..., :3] - src[..., :3]).max(axis=2)
    report["coverage"] = {
        "silhouette_recall": round(float((union & sil).sum() / max(1, sil.sum())), 4),
        "outside_silhouette_px": int((union & ~sil).sum()),
        "multi_claimed_share": round(float((claim > 1).sum() / max(1, union.sum())), 4),
    }
    report["rest_reconstruction"] = {
        "mean_abs_error": round(float(diff[sil].mean()), 2),
        "pixels_off_by_30_plus": round(float((diff[sil] > 30).mean()), 4),
    }

    # ---- summary -------------------------------------------------------
    ious = [r["iou"] for r in report["parts"]]
    bfs = [r["boundary_f"] for r in report["parts"]]
    lr_ok = [r["correct"] for r in report["lr"]]
    report["summary"] = {
        "parts_predicted": len(parts), "tags_mapped": len(pred),
        "missing": len(report["missing"]),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "mean_boundary_f": round(float(np.mean(bfs)), 4) if bfs else None,
        "lr_accuracy": round(sum(lr_ok) / len(lr_ok), 4) if lr_ok else None,
        "order_pairwise_accuracy": report["order"]["pairwise_accuracy"],
        "silhouette_recall": report["coverage"]["silhouette_recall"],
        "multi_claimed_share": report["coverage"]["multi_claimed_share"],
        "rest_mean_abs_error": report["rest_reconstruction"]["mean_abs_error"],
    }

    print("== summary ==")
    for k, v in report["summary"].items():
        print("  %-26s %s" % (k, v))
    print("\n%-18s %6s %6s %6s %6s" % ("part", "IoU", "prec", "recall", "bndF"))
    for r in sorted(report["parts"], key=lambda r: r["iou"]):
        print("%-18s %6.3f %6.3f %6.3f %6.3f" % (r["tag"], r["iou"], r["precision"],
                                                  r["recall"], r["boundary_f"]))
    if report["missing"]:
        print("\nmissing:", ", ".join(report["missing"]))
    if report["unmapped"]:
        print("unmapped tags:", ", ".join(report["unmapped"]))
    bad_lr = [r["tag"] for r in report["lr"] if not r["correct"]]
    if bad_lr:
        print("left/right swapped:", ", ".join(bad_lr))
    if report["order"]["wrong"]:
        print("\nwrong draw order (gt front -> predicted front):")
        for w in report["order"]["wrong"]:
            print("  %s over %s  (run puts %s in front, %d px)" % (
                w["gt_front"], w["b"] if w["gt_front"] == w["a"] else w["a"],
                w["pred_front"], w["overlap_px"]))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("\nreport ->", args.json)


if __name__ == "__main__":
    main()
