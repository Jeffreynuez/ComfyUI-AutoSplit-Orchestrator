"""
Pixel ownership + hidden-area fill from "peeled" pictures (command line).

The merge itself lives in autosplit_core/fillmerge.py, which the fill node
uses too; this script is its command line.

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
      garment's convex outline (from its peeled shape when it has one), in
      the garment's deep interior shadow. With "of": [hair parts],
      "hair": true and "below": "face" it builds the back hair at the very
      back when the split found none;
  {"skin_under": {"tag": "face", "features": ["left eye", "nose", ...]}}
      the head under the facial features is plain skin (inpainted from the
      face around them), as the rigger paints the head layer.

    python tools/peel_merge.py --run <body pass> [--run <facial pass>] --source <picture> \
        --plan plan.json --out <new body pass folder> [--no-own] [--raw-colour]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ComfyUI-AutoSplit-Orchestrator"))
from autosplit_core.fillmerge import (_hist_via, apply_colour_map, fit_colour_map,  # noqa: E402,F401
                                      keep_islands, lab_err, lab_to_rgb, merge, rgb_to_lab)


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
    ap.add_argument("--min-island", type=float, default=0.3,
                    help="keep an added island that does not touch the part when it is at"
                         " least this share of the part's peeled shape")
    ap.add_argument("--max-map-rmse", type=float, default=8.0,
                    help="ignore a fitted colour map whose fit error (Lab) is above this")
    args = ap.parse_args()
    merge(args.run, args.source, args.plan, args.out, no_own=args.no_own, raw_colour=args.raw_colour,
          min_island=args.min_island, max_map_rmse=args.max_map_rmse)


if __name__ == "__main__":
    main()
