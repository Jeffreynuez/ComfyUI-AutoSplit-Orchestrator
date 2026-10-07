"""
autosplit_core: the parts of AutoSplit that don't need ComfyUI, torch or Krita.

Pure numpy + OpenCV (drawing.py is plain Python), so the same logic runs inside
the ComfyUI node, in the command-line tools and in unit tests.

    masks     hole filling, cleanup, connected components, depth splits
    sides     picking a left/right pair and naming it by a stated convention
    ordering  draw order from evidence in the picture (colour, containment,
              depth) with the name table only as a tie-breaker
    drawing   merging the body and facial passes into one draw order
    metadata  the parts_metadata.json contract (schema 2)
"""
