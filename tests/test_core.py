"""
Unit tests for autosplit_core (numpy + OpenCV only, no GPU, no ComfyUI).

    python -m pytest tests          # or
    python tests/test_core.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "ComfyUI-AutoSplit-Orchestrator"))
from autosplit_core import drawing, masks, ordering, ownership, peel, sides  # noqa: E402


def rect(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


# ---------------------------------------------------------------- sides --
def test_anatomical_front_view_mirrors():
    viewer_left = rect(100, 200, 40, 60, 10, 40)
    viewer_right = rect(100, 200, 40, 60, 160, 190)
    left, right, info = sides.assign(viewer_left, viewer_right, "anatomical", "front")
    assert left is viewer_right and right is viewer_left and info["mirrored"]


def test_viewer_convention_and_back_view_do_not_mirror():
    a = rect(100, 200, 40, 60, 10, 40)
    b = rect(100, 200, 40, 60, 160, 190)
    left, _, _ = sides.assign(b, a, "viewer", "front")
    assert left is a
    left, _, info = sides.assign(a, b, "anatomical", "back")
    assert left is a and not info["mirrored"]


def test_pick_pair_drops_duplicates_and_tiny_partners():
    a = rect(100, 200, 40, 60, 10, 40)
    a_dup = rect(100, 200, 41, 60, 10, 40)
    b = rect(100, 200, 40, 60, 160, 190)
    speck = rect(100, 200, 0, 2, 0, 2)
    pair = sides.pick_pair([(a, 0.9), (a_dup, 0.8), (speck, 0.7), (b, 0.6)])
    assert len(pair) == 2 and pair[1][0] is b


def test_split_union_names_components():
    both = rect(100, 200, 40, 60, 10, 40) | rect(100, 200, 40, 60, 160, 190)
    left, right, info = sides.split_union(both)
    assert masks.centroid(left)[0] > masks.centroid(right)[0]  # anatomical: left on viewer right
    assert info["lr_source"] == "components"


# ---------------------------------------------------------------- masks --
def test_fill_holes_and_clean_keep_size_at_border():
    m = rect(20, 20, 2, 18, 2, 18)
    m[8:12, 8:12] = False
    assert masks.fill_holes(m)[10, 10]
    assert masks.clean(m).sum() == 256


def test_split_by_depth():
    m = rect(100, 100, 10, 90, 10, 90)
    near = np.zeros((100, 100), np.float32)
    near[10:40, 10:90] = 1.0          # top band is closer
    front, back = masks.split_by_depth(m, near)
    assert front[20, 50] and back[70, 50]


def test_nearness_flips_dark_convention():
    d = np.array([[0.1, 0.9]], np.float32)
    assert masks.nearness(d, "dark")[0, 0] > masks.nearness(d, "dark")[0, 1]
    assert masks.nearness(d, "bright")[0, 1] > masks.nearness(d, "bright")[0, 0]


# ------------------------------------------------------------- ordering --
def _scene():
    """Skin-coloured torso with a green strap across it, an eye on a face."""
    h, w = 240, 240
    img = np.full((h, w, 3), 80, np.uint8)
    torso = rect(h, w, 100, 220, 60, 180)
    img[torso] = (210, 150, 120)
    strap = rect(h, w, 150, 170, 60, 180)
    img[strap] = (40, 160, 60)
    face = rect(h, w, 10, 90, 80, 160)
    img[face] = (220, 170, 140)
    eye = rect(h, w, 40, 52, 95, 115)
    img[eye] = (250, 250, 250)
    return img, {
        # SAM-style masks: the torso and face come back with the strap / eye
        # inside them (holes filled), the strap and eye on their own
        "torso": torso, "strap": strap, "face": face, "left eye": eye,
    }


def test_strap_and_eye_drawn_in_front():
    img, ms = _scene()
    res = ordering.compute_order(img, ms)
    z = res["z_order"]
    assert z["strap"] < z["torso"], res["edges"]
    assert z["left eye"] < z["face"], res["edges"]


def test_colour_beats_a_wrong_prior():
    img, ms = _scene()
    # pretend the name table thinks torso goes in front of the strap
    res = ordering.compute_order(img, ms, prior={"strap": 15, "torso": 10})
    assert res["z_order"]["strap"] < res["z_order"]["torso"]


def test_prior_rank_handles_suffixes_and_unknowns():
    r = ordering.prior_rank
    assert r("iris_left") == r("left iris") < r("left eye") < r("face")
    assert r("left shin brace") < r("left leg")
    assert r("green sash") < r("skirt")
    assert r("hair_back") > r("torso") > r("hair_front")


# -------------------------------------------------------------- drawing --
def test_merge_puts_facial_parts_in_front_of_face_only():
    body = [{"tag": "hair_back"}, {"tag": "torso"}, {"tag": "face"}, {"tag": "hair_front"}]
    face = [{"tag": "left eye"}, {"tag": "left iris"}]
    merged = [e["tag"] for e in drawing.merge(body, face)]
    assert merged == ["hair_back", "torso", "face", "left eye", "left iris", "hair_front"]


def test_merge_without_anchor_goes_on_top():
    merged = [e["tag"] for e in drawing.merge([{"tag": "torso"}], [{"tag": "nose"}])]
    assert merged == ["torso", "nose"]


# ------------------------------------------------------------ ownership --
def test_owner_map_gives_each_pixel_to_the_front_most_part():
    jacket = rect(100, 100, 20, 80, 10, 90)        # SAM 3 jacket includes the sleeve
    sleeve = rect(100, 100, 20, 80, 70, 90)        # drawn in front of the jacket
    vis = ownership.visible([("jacket", jacket), ("sleeve", sleeve)])
    assert not (vis["jacket"] & vis["sleeve"]).any()
    assert vis["sleeve"].sum() == sleeve.sum()
    assert vis["jacket"].sum() == jacket.sum() - sleeve.sum()


def test_nested_sub_part_keeps_its_pixels_whatever_the_order():
    jacket = rect(100, 100, 20, 80, 10, 90)        # swallowed the far sleeve
    far_sleeve = rect(100, 100, 30, 80, 75, 88)    # drawn BEHIND the jacket
    vis = ownership.visible([("far sleeve", far_sleeve), ("jacket", jacket)])
    assert vis["far sleeve"].sum() == far_sleeve.sum()
    assert not (vis["jacket"] & far_sleeve).any()
    # the jacket is drawn in front, so it must not be filled over the sleeve
    region, front = ownership.hidden_candidates([("far sleeve", far_sleeve), ("jacket", jacket)],
                                                "jacket", expand=0.2)
    assert front == [] and not region.any()


def test_hidden_candidates_are_front_parts_near_the_part():
    torso = rect(200, 200, 40, 70, 80, 120)        # only the neck shows
    shirt = rect(200, 200, 70, 150, 60, 140)       # in front, right below it
    far = rect(200, 200, 180, 199, 0, 20)          # in front, but far away
    btf = [("torso", torso), ("shirt", shirt), ("far", far)]
    region, front = ownership.hidden_candidates(btf, "torso", expand=1.0)
    assert front == ["shirt"]
    assert region[90, 100] and not region[190, 10] and not region[50, 100]
    assert abs(ownership.hidden_share(btf, "torso")) < 1e-9


# ----------------------------------------------------------------- peel --
def test_peel_categories_and_names():
    assert peel.category("left shin brace") == "garment"
    assert peel.category("under shirt") == "garment"
    assert peel.category("hair_back") == "hair"
    assert peel.category("left leg") == "body" and peel.category("face") == "body"
    assert peel.category("bottom lips") == "facial" and peel.category("eyebrow_left") == "facial"
    assert peel.category("glowing amulet") == "garment"          # unknown: an accessory
    assert peel.readable("hair_front") == "front hair"
    assert peel.join_names(["the brown left shin brace", "the brown right shin brace", "the sash"]) \
        == "both brown shin braces and the sash"
    assert peel.describe("green sash", np.full((10, 3), 120, np.uint8)) == "the green sash"


def test_peel_colour_words_and_shades():
    def name(rgb):
        return peel.colour_word(np.array([rgb] * 4, np.uint8))
    assert name((99, 71, 59)) == "brown"            # muted dark brown, not "dark grey"
    assert name((127, 103, 72)) == "brown" and name((197, 185, 160)) == "beige"
    assert name((102, 151, 83)) == "green" and name((37, 83, 65)) == "dark green"
    assert name((80, 80, 80)) == "dark grey" and name((40, 80, 200)) == "blue"
    assert name((220, 30, 30)) == "red" and name((120, 50, 160)) == "purple"
    assert peel.shade_apart("the green sash", "the green skirt", 66, 58) == "the light green sash"
    assert peel.shade_apart("the green sash", "the green skirt", 60, 58) == "the green sash"
    assert peel.shade_apart("the brown jacket", "the green skirt", 30, 58) == "the brown jacket"
    assert peel.limbs_phrase(["right hand"]) == "both of the character's hands"
    assert peel.limbs_phrase(["left arm", "right hand"]) == "both of the character's arms and hands"


def test_peel_garment_prompts():
    p = peel.garment_prompt("the green skirt", ["the light green sash"], ["right hand"], False)
    assert p.startswith("Remove the light green sash that lies over the green skirt and remove both"
                        " of the character's hands, so that the green skirt is fully visible")
    p = peel.garment_prompt("the brown jacket", [], ["right arm"], True)
    assert "as if the character had no arms" in p and "the brown jacket as a sleeveless vest" in p
    assert "right" not in p.split("Keep")[0]


def test_peel_plan_from_a_split():
    H, W = 240, 200
    rgb = np.full((H, W, 3), 80, np.uint8)
    skin, shirt, cloth = (225, 160, 135), (235, 225, 195), (75, 160, 75)
    face = rect(H, W, 10, 50, 70, 130)
    hair_back = rect(H, W, 5, 70, 55, 145)
    hair_front = rect(H, W, 5, 25, 60, 140)
    torso = rect(H, W, 50, 150, 60, 140)        # mostly under the shirt
    top = rect(H, W, 60, 120, 55, 145)
    leg = rect(H, W, 140, 235, 75, 125)
    skirt = rect(H, W, 140, 190, 55, 145)       # covers the top of the leg
    sash = rect(H, W, 145, 185, 100, 140)
    for m, c in ((hair_back, (30, 120, 90)), (face, skin), (torso, skin), (leg, skin),
                 (top, shirt), (skirt, cloth), (sash, (165, 215, 140)), (hair_front, (40, 150, 110))):
        rgb[m] = c
    btf = [("hair_back", hair_back), ("torso", torso), ("leg", leg), ("face", face),
           ("under shirt", top), ("skirt", skirt), ("green sash", sash), ("hair_front", hair_front)]
    jobs = peel.plan(btf, rgb)
    kinds = {j.get("kind", "rule") for j in jobs}
    assert {"base body", "garment", "head", "rule"} <= kinds, kinds
    base = next(j for j in jobs if j.get("kind") == "base body")
    assert set(base["targets"]) == {"torso", "leg"}
    assert "skirt" in base["removes"] and "under shirt" in base["removes"]
    assert "underwear" in base["parts"]["torso"]
    assert "sports top" in base["prompt"]                      # never a topless base body
    skirt_job = next(j for j in jobs if j.get("name") == "garment_skirt")
    assert "the light green sash that lies over the green skirt" in skirt_job["prompt"]
    assert skirt_job["model"] == "4b"
    assert any("hull" in j and j["hull"]["tag"] == "hair_back" for j in jobs)


def test_peel_jacket_behind_arm_and_skirt():
    H, W = 240, 200
    rgb = np.full((H, W, 3), 80, np.uint8)
    jacket = rect(H, W, 40, 150, 50, 150)
    skirt = rect(H, W, 130, 200, 50, 150)       # the jacket hem tucks behind the skirt
    arm = rect(H, W, 40, 160, 40, 75)           # a sleeve over the jacket's side
    for m, c in ((jacket, (127, 103, 72)), (skirt, (102, 151, 83)), (arm, (120, 95, 65))):
        rgb[m] = c
    jobs = peel.plan([("jacket", jacket), ("skirt", skirt), ("right arm", arm)], rgb)
    job = next(j for j in jobs if j.get("name") == "garment_jacket")
    assert job["removes"] == ["right arm"] and job["model"] == "9b"
    assert "sleeveless vest" in job["prompt"] and "skirt" not in job["prompt"]
    assert "vest" in job["sam_labels"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("ok   ", fn.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL ", fn.__name__, e)
    print("%d/%d passed" % (len(fns) - failed, len(fns)))
    sys.exit(1 if failed else 0)
