# Manga2Animate VBench

This directory contains the code used to evaluate Manga2Animate with
[VBench](https://github.com/Vchitect/VBench) and VBench++ I2V. It is a
code-only snapshot: videos, reference images, model checkpoints, caches,
workspaces, logs, and generated reports are deliberately excluded.

The retained scope matches the completed Manga2Animate evaluation. VBench-2.0
is not included because it was not part of that run.

## Layout

- evaluate_i2v.py: VBench++ I2V entry point
- vbench/: upstream VBench implementation
- vbench2_beta_i2v/: upstream VBench++ I2V implementation
- tools/: Manga2Animate preparation, evaluation, and reporting tools

See [the tools guide](tools/README.md) for the custom workflow groups.

## Setup

From this directory, create the same environment used by VBench and install
the local package with pip install -r requirements.txt followed by
pip install -e .

The supplemental DINOv3 timm evaluator requires timm 1.0.20 and downloads its
model through the normal Hugging Face cache. Because upstream VBench pins an
older timm release, use requirements-dinov3.txt in a separate environment. The optional Meta-official
DINOv3 evaluator expects the dinov3 source and
dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth checkpoint under
third_party/dinov3_official/. That directory is ignored so third-party
metadata and the large checkpoint cannot be committed accidentally.

## Runtime data

The preparation scripts preserve the original experiment layout and look for
manga2animate_video/, manga2animate_character_ref/, manga2animate_scene_ref/,
and manga2animate_prompt/ at this repository root.

Copy or symlink those inputs locally when running an evaluation. All four
paths, plus evaluation_runs/ and .cache/, are ignored by Git. Do not add media,
weights, evaluation output, or machine-specific manifests here.

## Upstream

The VBench code is derived from commit d49e4f04744da375cb7e8f548c8c3d41d5fdd015.
Upstream licensing is retained in [LICENSE](LICENSE).
