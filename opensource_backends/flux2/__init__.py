"""FLUX.2-dev storyboard backend.

Selected with `run_all.py --storyboard-backend flux2`. Generates each storyboard
panel separately (N variants per panel, conditioned on the source manga panel
plus the per-character reference crops), has a Gemini reviewer pick the best
variant per panel, then tiles the winners into the same 3x2 sheet the gemini
backend emits as a single image.

Module map:
    bridge.py                     prompt/job-spec building; launches worker.py in
                                  the `sb_diffusers` conda env (cross-env subprocess)
    worker.py                     the ONLY file importing torch/diffusers; runs in
                                  that separate env and must never import src.* or
                                  any sibling module here
    evaluate_panel_candidates.py  Gemini best-of-N per-panel reviewer
    grid_compose.py               final storyboard sheet composition
"""
