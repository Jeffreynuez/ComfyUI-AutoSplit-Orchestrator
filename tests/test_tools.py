"""
Tests for the scoring and fill helpers in tools/ (numpy + OpenCV + Pillow).

    python tests/test_tools.py
"""
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


evaluate = _load("evaluate")
peel_merge = _load("peel_merge")
peel_plan = _load("peel_plan")
add_fill_group = _load("add_fill_group")


def test_colour_sim_same_paint_scores_one_and_other_paint_zero():
    skin = np.array([[224, 160, 140]] * 3, np.uint8)
    near = np.array([[226, 161, 141]] * 3, np.uint8)
    green = np.array([[90, 170, 90]] * 3, np.uint8)
    assert np.allclose(evaluate.colour_sim(skin, near), 1.0)
    assert np.allclose(evaluate.colour_sim(skin, green), 0.0)


def test_lab_of_white_and_black():
    lab = evaluate.srgb_to_lab(np.array([[255, 255, 255], [0, 0, 0]], np.uint8))
    assert abs(lab[0, 0] - 100) < 0.5 and abs(lab[1, 0]) < 0.5


def test_colour_map_undoes_a_global_shift_and_ignores_changed_pixels():
    rng = np.random.default_rng(1)
    orig = rng.integers(40, 200, (200, 200, 3)).astype(np.uint8)
    lab = peel_merge.rgb_to_lab(orig.reshape(-1, 3))
    peel = peel_merge.lab_to_rgb(lab * np.array([0.9, 1.05, 1.05]) + np.array([20, 3, -2]))
    peel = peel.reshape(200, 200, 3)
    peel[:60] = rng.integers(0, 255, (60, 200, 3))      # the part the edit changed
    M, rmse = peel_merge.fit_colour_map(peel, orig, np.ones((200, 200), bool))
    back = peel_merge.apply_colour_map(peel[100:].reshape(-1, 3), M)
    assert peel_merge.lab_err(back, orig[100:].reshape(-1, 3)) < 3.0
    assert M[0, 1] == 0 and M[1, 0] == 0                 # no cross-channel terms


def test_peel_workflow_wires_edit_into_a_full_size_split():
    job = {"name": "garment_skirt", "prompt": "Remove the green sash ...", "model": "9b",
           "sam_labels": ["skirt", "green skirt"]}
    tmpl = peel_plan._template_nodes(peel_plan.DEFAULT_TEMPLATE)
    wf = peel_plan.peel_workflow(job, "picture.png", "autosplit_fill/t", tmpl)
    assert wf["1"]["inputs"]["image"] == "picture.png"
    assert "9b" in wf["3"]["inputs"]["unet_name"]
    assert wf["41"]["inputs"]["image"] == ["14", 0]                      # edit, scaled back
    assert wf["41"]["inputs"]["width"] == ["40", 0] and wf["40"]["inputs"]["image"] == ["1", 0]
    o = wf["45"]["inputs"]
    assert o["image"] == ["41", 0] and o["depth_map"] == ["43", 0] and o["sam3_model"] == ["44", 0]
    assert o["part_labels"] == "skirt\ngreen skirt" and o["run_id"] == "garment_skirt"
    assert o["output_directory"] == "autosplit_fill/t" and o["auto_lr_split_parts"] == ""
    assert wf["42"]["inputs"]["filename_prefix"] == "autosplit_fill/t/garment_skirt_up"


def test_add_fill_group_wires_the_fill_to_the_split():
    def node(i, typ, inputs=(), outputs=()):
        return {"id": i, "type": typ, "pos": [0, 100 * i], "size": [100, 100], "mode": 0,
                "inputs": [{"name": n, "type": t, "link": l} for n, t, l in inputs],
                "outputs": [{"name": n, "type": t, "links": ls} for n, t, ls in outputs]}
    wf = {"last_node_id": 26, "last_link_id": 6, "groups": [{"id": 1, "bounding": [0, -500, 10, 10]}],
          "nodes": [node(1, "LoadImage", outputs=[("IMAGE", "IMAGE", [1, 4])]),
                    node(23, "DepthAnythingV2Preprocessor", outputs=[("IMAGE", "IMAGE", [2])]),
                    node(10, "easy sam3ModelLoader", outputs=[("sam3_model", "EASY_SAM3_MODEL", [3])]),
                    node(15, "AutoSplitOrchestratorSAM3",
                         inputs=[("image", "IMAGE", 1), ("depth_map", "IMAGE", 2),
                                 ("sam3_model", "EASY_SAM3_MODEL", 3)],
                         outputs=[("report", "STRING", None), ("preview_grid", "IMAGE", None)]),
                    node(24, "HeadCropUpscale", inputs=[("image", "IMAGE", 4)],
                         outputs=[("cropped_image", "IMAGE", [5])]),
                    node(26, "AutoSplitOrchestratorSAM3", inputs=[("image", "IMAGE", 5)],
                         outputs=[("report", "STRING", None), ("preview_grid", "IMAGE", None)])],
          "links": [[1, 1, 0, 15, 0, "IMAGE"], [2, 23, 0, 15, 1, "IMAGE"], [3, 10, 0, 15, 2, "EASY_SAM3_MODEL"],
                    [4, 1, 0, 24, 0, "IMAGE"], [5, 24, 0, 26, 0, "IMAGE"]]}
    fid = add_fill_group.add_fill_group(wf)
    fill = next(n for n in wf["nodes"] if n["id"] == fid)
    by_id = {l[0]: l for l in wf["links"]}
    src = [(by_id[i["link"]][1], by_id[i["link"]][2]) for i in fill["inputs"]]
    assert src == [(1, 0), (23, 0), (10, 0), (15, 0), (26, 0)]
    assert wf["last_link_id"] == max(by_id) and wf["last_node_id"] >= fid + 1
    assert by_id[fill["inputs"][3]["link"]][0] in next(n for n in wf["nodes"] if n["id"] == 15)["outputs"][0]["links"]
    g = wf["groups"][-1]
    assert g["bounding"][1] + g["bounding"][3] <= -500 + 1         # above the old groups


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
