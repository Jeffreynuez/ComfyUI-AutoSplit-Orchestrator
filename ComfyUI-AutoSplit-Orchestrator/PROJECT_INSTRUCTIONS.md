# 2D AutoSplit Project — Instructions & Reference

**Owner:** Jeffrey De La Nuez — Character Rigger / Animator / AI Technical Artist (10 yrs, 100+ characters, thousands of parts)
**Last updated:** 2026-06-26

---

## 1. What this project is

A multi-stage **ComfyUI pipeline** that automates Jeffrey's 2D character rigging workflow — taking a flat, never-rigged character illustration and turning it into clean, named, riggable body parts, then painting the hidden/occluded regions so each part is whole, with a style LoRA to keep everything in his flat-shaded 2D art style.

The end-target for the split parts is **Jeffrey's own rigging tool/workflow** — *not* Spine. The deliverable is **clean PNG parts + rich metadata** (bounding box, depth/z-order, anatomical L/R). No Spine JSON or atlas export is required.

### The stages

| Stage | Name | Status | What it does |
|-------|------|--------|--------------|
| 1 | **AutoSplit Orchestrator (SAM3)** | Working, actively refined | Auto-segments a full-body character into named, z-ordered, anatomically-labeled body-part PNGs + metadata sidecar |
| 2 | **Part Painter** | Wired, results poor | Flux Fill + Redux inpaints occluded regions so each part is complete behind overlaps |
| 3 | **Style LoRA** | Planned | Train a LoRA to teach Flux the flat-shaded 2D character style (fixes Stage 2's core problem) |
| 4 | **Skin Variants** | Future | Once a character is rigged once, reuse its part alpha masks to generate restyled skin variants cheaply (see §6) |

---

## 2. Stage 1 — AutoSplit Orchestrator (SAM3)

**Primary node:** `AutoSplitOrchestratorSAM3` — display name *"Auto-Split Orchestrator (SAM3)"* — in `nodes_sam3.py`.
SAM3 (text-grounded segmentation) is the active path and produces **significantly better results** than the older SAM2 version (`nodes.py`, `AutoSplitOrchestrator`, which used Florence-2 detect → SAM2). SAM3 tries multiple prompt variations per part and only falls back to Florence-2 if it fails.

**Pipeline:** Phase 1 segments every part via SAM3 (per-part prompt variations, accept/reject by confidence + area). Phase 2 runs occlusion analysis against the depth map, optional inpaint, crop + save, and writes the metadata sidecar. Returns `(report, preview_grid)`.

### Key parameters (SAM3 node)

- `image`, `depth_map`, `sam3_model` (EASY_SAM3_MODEL) — required inputs
- `part_labels` (multiline) — one body part per line, each sent as a SAM3 text prompt
- `character_name`, `output_directory` (default `split_parts_sam3`)
- `padding` (default 20), `confidence_threshold` (default 0.30), `occlusion_threshold` (default 0.15)
- `mask_expand_pixels` (default 15), `mask_blur` (0–64, default 0 — soft feathered edges)
- `enable_inpainting` (default True)
- `output_mode` — `alpha_mask` | `solid_background` (default `solid_background`)
- `background_color` (COLORCODE hex picker, default `#FFFFFF`)
- `mask_output_mode` — `blank` (all-black PNG for manual painting, default) | `part_silhouette` (white-where-part/black-elsewhere) | `none` (no mask file)
- `auto_lr_split_parts` (default `hands / feet / ears`)
- `lr_convention` — **`anatomical`** (default) | `viewer`
- `cluster_hair_front_back` (default True — splits hair into hair_front / hair_back by depth)
- `passthrough_parts` (default `nose / mouth` — pixels kept verbatim, never inpainted/composited)
- `write_metadata_sidecar` (default True — writes `parts_metadata.json` with bbox, depth_median, z-order)
- `florence2_model` (FL2MODEL) — optional fallback

Area filter constants: `MAX_PART_AREA_PCT = 25.0`, `MIN_PART_AREA_PCT = 0.01`.

### Two-pass facial detail (confirmed working)

Facial features are tiny on a full-body sheet (a mouth can be ~40×40 px). The `HeadCropUpscale` node (`nodes_head_crop.py`, *"Head Crop + Upscale (Facial Detail)"*) auto-detects the head via SAM3, crops with `padding_pct` 0.25 (captures hair/ears), upscales `scale_factor` 2.0, and crops/scales the depth map to match (avoids a second Marigold pass). It feeds a **second orchestrator instance** configured only for facial parts (eyes, iris, mouth, nose, eyebrow) at confidence 0.25, saving to `split_parts_sam3_facial`. It lives in a **bypassable purple ComfyUI group** above the body pipeline.

### Supporting nodes

- `AsymmetricMaskExpand` (`nodes_mask_expand.py`) — *"Asymmetric Mask Expand (AutoSplit)"*
- `SplitLeftRight` (`nodes_split_lr.py`) — *"Split Left/Right (AutoSplit)"*

---

## 3. Stage 2 — Part Painter

**Workflow:** `user/default/workflows/2D_Character_Part_Painter.json`.

Loads an isolated part, the user draws/uses a mask over areas to fill, and Flux Fill inpaints the missing content with Redux style conditioning from the original character. Wiring uses standard nodes: `InpaintCropImproved` → `INPAINT_ExpandMask` / `INPAINT_MaskedFill` → `InpaintModelConditioning` → `KSampler` (35 steps, euler/karras, denoise 0.9) → `VAEDecode` → `InpaintStitchImproved`.

**Models:** UNet `flux1-fill-dev-Q8_0.gguf` (via `UnetLoaderGGUF`); VAE `vae/flux/ae.safetensors`; CLIP `clip_l` + `t5xxl_fp8_e4m3fn`; style `flux1-redux-dev.safetensors` + `sigclip_vision_patch14_384.safetensors`; LoRA `loras/Flux/2d-auto-splitter.safetensors` at strength 1.

**Known problem (the reason Stage 3 exists):** Flux Fill produces **photorealistic output instead of flat-shaded 2D art** — it doesn't understand the style, so inpaint results are poor. A VAE incompatibility also drove the switch away from the ComfyUI-Flux-Inpainting plugin to the standard nodes above. Compositing surrounding parts for context (the Redux step) is the right idea and worth leaning into.

---

## 4. Stage 3 — Style LoRA (planned)

Train a LoRA on Jeffrey's large dataset of character parts to teach Flux the flat-shaded 2D style, then load it (LoraLoader between `UnetLoaderGGUF` and `KSampler`) so Stage 2 produces in-style fills. Hardware (§7) means quantization (FP8), gradient checkpointing, 8-bit optimizer, batch size 1–2 — full fine-tuning is not feasible. A `2d-auto-splitter.safetensors` LoRA already exists and is used as a style aid in Stage 2.

---

## 5. Conventions (always follow)

- **Anatomical L/R** is the default everywhere parts are labeled — the *character's own* left/right, not viewer-relative. The character's left hand exits on the viewer's right but is still "left." Keep `anatomical` the default; offer `viewer` as a toggle.
- **passthrough_parts** (nose, mouth, and any tiny facial feature) keep source pixels **verbatim** — the inpainter destroys small features.
- **New/optional features go in bypassable ComfyUI groups positioned *above* the existing setup**, so the base pipeline stays intact and features can be toggled.
- **Deliverable = clean PNG parts + rich metadata.** No Spine export.
- **Working style:** when asked to work on something, ask clarifying questions, raise concerns, and offer suggestions/recommendations before/while executing — don't just execute.

---

## 6. Stage 4 — Skin Variants (future, inspired by "Spine Skin Studio")

Leonardo Cunha's *Spine Skin Studio* automates making **new skins for already-rigged characters**: AI-generate a restyled character, then "auto-split" it by **reusing the original texture's alpha channels** (not segmentation) and inject as a new skin. His tool sidesteps segmentation because the rig already exists — it's **complementary** to this project's SAM3 splitter, which solves the harder upstream problem of rigging a flat illustration from scratch.

**The plan:** once a character has been split/rigged once via Stage 1, its part PNGs' alpha channels *are* reusable masks. Adopt Leonardo's trick to generate restyled **skin variants** cheaply — AI-restyle the full character, then re-cut it using the existing part masks instead of re-segmenting. Works fine targeting Jeffrey's own rig tool (no Spine needed).

His backend is also instructive: ComfyUI workflows hardcoded in an in-app "AI terminal," downscaling 4K for Comfy (especially when compositing parts for context), managing upscale via Comfy with mesh placement, upscale deferrable to project-finish. (An app wrapper is a *much* later concern — keep ComfyUI as the engine.)

---

## 7. Environment & file map

**Hardware:** RTX 4080 Super (16 GB VRAM), 64 GB RAM, Windows, Python 3.12.10, PyTorch 2.9.1+cu130, cudaMallocAsync. 16 GB VRAM drives GGUF Q8_0 Flux Fill + fp8 T5.

**ComfyUI root:** set `AUTOSPLIT_COMFY_ROOT`, or use the default ComfyUI-Easy-Install layout beside this project.

**Custom node dir:** `custom_nodes\ComfyUI-AutoSplit-Orchestrator\`
- `nodes_sam3.py` — active SAM3 orchestrator
- `nodes.py` — SAM2 orchestrator (legacy)
- `nodes_head_crop.py`, `nodes_mask_expand.py`, `nodes_split_lr.py`, `__init__.py`
- (no README; no `workflows/` folder inside the node dir)

**Workflows:** `user\default\workflows\`
- `2D_Character_AutoSplit_SAM3.json` — main pipeline (SAM3 + facial pass + new widgets)
- `2D_Character_AutoSplit_Pipeline.json` — SAM2 pipeline
- `2D_Character_Part_Painter.json` — Stage 2

**Outputs:** `output\split_parts_sam3\` (body), `output\split_parts_sam3_facial\` (face), `output\split_parts\` (SAM2) — each with `parts_metadata.json`.

**Models:** `models\loras\Flux\2d-auto-splitter.safetensors`, `models\vae\flux\ae.safetensors`, `flux1-fill-dev-Q8_0.gguf`.

**Dependencies:** `comfyui-easy-sam3` (SAM3, `EASY_SAM3_MODEL`), `comfyui-florence2` (fallback), `ComfyUI-segment-anything-2` (SAM2 path). SAM3/HeadCrop node loading is wrapped in try/except so the package degrades gracefully if a dependency is missing.

---

## 8. Memory (Pinecone)

This project's persistent memory lives in the **`comfyui`** namespace of the `claude-memory` Pinecone index. At the start of a session, search it (and `shared`) for relevant prior context; after substantive work (a decision, fix, blocker, or new convention), write a flat record back. See the `pinecone-memory` skill for record shape and routing.

---

## 9. Where we left off / likely next steps

Last session ended right after adding `mask_output_mode` (blank/part_silhouette/none) plus the COLORCODE background picker and `mask_blur`, and patching both orchestrators in the workflow JSON — *"restart ComfyUI and reload to see the new dropdown."* The two-pass facial detail pass was confirmed working.

Natural next frontiers: (a) verify the new Stage 1 widgets end-to-end after restart; (b) tackle **Stage 2** — the Flux Fill style problem, which really points at **Stage 3 (the style LoRA)** as the unlock; (c) Stage 4 skin variants once a character is fully rigged once.
