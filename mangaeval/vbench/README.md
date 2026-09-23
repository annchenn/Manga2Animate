# MangaEval: video metrics (VBench++)

The video-level metrics of MangaEval. Setup, input format and the full metric
list are in [`../README.md`](../README.md) (section "VBench++ metrics").

| file | metrics |
|---|---|
| `auto_clip.py` | splits each video into its 6 panel clips (`need_clip/<video>/panel_01.mp4` ... `panel_06.mp4`) |
| `evaluate_i2v.py` | I2VB, SC, BC, MS (VBench++ I2V, unchanged) |
| `tools/evaluate_i2v_subject_dinov3_official.py` | I2VS with a DINOv3 ViT-B/16 backbone (VBench++ scoring formula unchanged) |
| `tools/modified_dynamic_degree.py` | DD as a continuous score: mean RAFT flow magnitude, normalised by 256/min(H,W) (`normalized_modified_dd_mean` column) |

`vbench/` and `vbench2_beta_i2v/` are the upstream VBench and VBench++ I2V code
(https://github.com/Vchitect/VBench, commit
d49e4f04744da375cb7e8f548c8c3d41d5fdd015), kept unchanged under its
[LICENSE](LICENSE).

The DINOv3 evaluator expects the DINOv3 source and the
`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` checkpoint under
`third_party/dinov3_official/`; its extra dependencies are in
`requirements-dinov3.txt`.
