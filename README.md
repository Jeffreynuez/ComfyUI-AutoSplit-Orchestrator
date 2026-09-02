# 2D AutoSplit Studio

Turn a flat 2D character illustration into named, z-ordered, anatomically labeled part PNGs that assemble and rig in Spine. **26 parts in about 33 seconds**, verified in Spine Pro 4.3.22.

Before a rigger can start rigging, someone has to cut the character up: separate the arms from the torso, the hair from the face, the irises from the eyes, name every piece, and stack them in the right draw order. On a detailed character that is roughly **two hours of manual work in Photoshop**. This does it in one click from inside Krita.

---

## What it does

Point it at a single flat character image. It runs a two-pass segmentation in ComfyUI, then hands you the result three ways: as named layers back inside Krita, as part PNGs on disk, and as a Spine project you can import and rig immediately.

**The pipeline**

1. **Body pass.** SAM 3 text-grounded segmentation with multiple prompt variations per body part, so a part that misses on one phrasing still gets caught on another.
2. **Head crop and upscale.** The face is a small fraction of a full-body illustration, so it gets cropped and upscaled 2x before the facial pass. That is the difference between finding an iris and not.
3. **Facial pass.** A second segmentation run over the upscaled crop, picking up eyes, irises, eyebrows, nose and lips separately.
4. **Depth for draw order.** Marigold depth estimation decides what sits in front of what, so the parts come back z-ordered rather than in an arbitrary pile.
5. **Mask cleanup.** An OpenCV routine closes morphology, flood-fills enclosed holes, and drops islands smaller than 2% of the largest component, which is what stops a jacket mask from arriving with pinholes in it.
6. **Left/right resolution.** Bilateral parts are split and labeled anatomically, so you get `left_arm` and `right_arm` rather than two files called `arm`.

**The output**

A worked example is committed in [`spine_export/`](spine_export/): 26 parts of a single character, 17 body and 9 facial, plus the Spine skeleton JSON. That is what a real run produces.

---

## Repo contents

| Path | What it is |
|---|---|
| `ComfyUI-AutoSplit-Orchestrator/` | The custom ComfyUI nodes written for this project: `AutoSplitOrchestratorSAM3` (the main orchestrator), `HeadCropUpscale`, `AsymmetricMaskExpand`, `SplitLeftRight`, plus a SAM 2 fallback path and a refine pass. Install into `ComfyUI/custom_nodes/`. |
| `krita_plugin/autosplit_studio/` | The Krita plugin. A docker with **Split Character**, **Export to Spine**, and **Generate Skin**. Pure standard library `urllib`, so it runs in Krita's bundled Python with nothing to install. |
| `krita_plugin/autosplit_studio/comfy_client.py` | Drives ComfyUI over its HTTP API: upload the image, strip the optional Florence-2 loader, inject the image, queue the prompt, poll for completion. |
| `spine_export.py` | Reads the split output plus the head-crop transform and writes a Spine 4.3.22 project: root bone, a slot and region attachment per part, default skin, correct draw order. |
| `comfy_autosplit_test.py` | Standalone API driver. Same logic as the plugin, runnable from a terminal, useful for debugging without Krita in the loop. |
| `restyle_test.py` | Inpaint-based restyle pass, driven the same way. |
| `2D_Character_AutoSplit_SAM3.json` | The ComfyUI workflow in API format. |
| `Flux_Klein_Krita_Inpaint.json`, `NovaAnimeXL_Krita_Inpaint.json` | Inpainting workflows used by the restyle path. |
| `spine_export/` | Sample output from a real run: part PNGs and the Spine skeleton JSON. |

---

## Requirements

- **ComfyUI**, running locally on port `8188`
- **Krita 5.x** for the plugin (optional; the standalone driver works without it)
- **Spine 4.3+** to import the exported skeleton (tested against 4.3.22)
- ComfyUI extensions: `comfyui-easy-use` (SAM 3 loader), Marigold depth, and optionally Florence-2
- Python: `requests` for the standalone driver only. The Krita plugin has no dependencies.

## Setup

1. Copy `ComfyUI-AutoSplit-Orchestrator/` into your `ComfyUI/custom_nodes/` folder and restart ComfyUI.
2. In ComfyUI, open **Settings** and enable **Dev mode Options**.
3. Load `2D_Character_AutoSplit_SAM3.json`. If you change the graph, re-export it with **Workflow → Export (API)** back into this folder.
4. Copy `krita_plugin/autosplit_studio/` and `autosplit_studio.desktop` into your Krita `pykrita` folder, then enable **AutoSplit Studio** in Krita's Python Plugin Manager.
5. Point it at your machine with two environment variables. No source edits required.

### Configuration

Every path is read from the environment with a sensible fallback, so nothing in this repo is machine-specific.

| Variable | What it sets | Default |
|---|---|---|
| `AUTOSPLIT_PROJECT` | This repo's folder | inferred from the script location |
| `AUTOSPLIT_COMFY_ROOT` | Your ComfyUI install root | `../ComfyUI-Easy-Install/ComfyUI` |
| `AUTOSPLIT_COMFY_URL` | ComfyUI server address | `http://127.0.0.1:8188` |
| `AUTOSPLIT_COMFY_OUTPUT` | ComfyUI output dir | `<COMFY_ROOT>/output` |
| `AUTOSPLIT_INPUT` | Character image to split | `<project>/input/character.png` |
| `AUTOSPLIT_OUT` | Where Spine output is written | `<project>/spine_export` |
| `AUTOSPLIT_CHARACTER` | Name used for the skeleton JSON | `character` |

Setting `AUTOSPLIT_PROJECT` matters for the Krita plugin specifically, since the plugin gets copied into Krita's `pykrita` folder and can no longer infer where the repo lives.

## Use

Open a flat character image in Krita and click **Split Character**. The plugin exports the active document, runs the workflow on a background thread so Krita stays responsive, and imports each part back as a named layer positioned at its bounding box in correct z-order, grouped into a folder. Then click **Export to Spine**, import the JSON into Spine via **Import Data**, and point the images path at `spine_export/images/`. The character arrives assembled and ready to rig.

---

## Notes and limitations

- Coordinate handling is the fiddly part. Image space is y-down with a top-left origin, Spine is y-up. The exporter puts the origin at the character's bounding-box bottom-centre so the feet land near y=0, and facial parts, which come from the 2x upscaled crop, are scaled by `1/scale` and offset by the crop origin to land correctly on the full image.
- Quality tracks the source art. Flat, clearly separated character art segments well. Heavy overlap, painterly rendering, and busy backgrounds do not.
- A first version of this was built in 2024 and **shelved**, because the segmentation models of that era could not cut the art cleanly enough to trust in production. Shipping something unreliable into an artist's daily workflow costs more trust than it saves time. It was picked back up in 2026 once the models had caught up, and this is that version.

## License

No license yet. All rights reserved for now.
