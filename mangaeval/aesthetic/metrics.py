"""No-reference aesthetic metrics for anime/manga-style images.

Both wrappers follow the reference inference code of each model exactly, so
that scores are comparable with the numbers reported in the paper. Do not
change the inference logic without re-validating the scores.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class AestheticMetric(ABC):
    """Common interface: load a model onto a device, then score one PIL image."""

    name: str
    higher_is_better: bool = True
    native_range: tuple[float, float]

    @abstractmethod
    def load(self, device: str) -> None: ...

    @abstractmethod
    def score(self, image: Image.Image) -> float: ...


class AestheticShadowV2(AestheticMetric):
    """1.1B-param ViT anime aesthetic classifier, binary {hq, lq}.

    Weights: NeoChen1024/aesthetic-shadow-v2-backup, a community mirror of the
    now-deleted shadowlilac/aesthetic-shadow-v2 (identical safetensors/config).
    License: CC-BY-NC-4.0 (non-commercial) -- see README.md before any
    commercial use of scores produced by this metric.
    """

    name = "aesthetic_shadow_v2"
    native_range = (0.0, 1.0)
    hf_id = "NeoChen1024/aesthetic-shadow-v2-backup"

    def load(self, device: str) -> None:
        from transformers import pipeline

        dev = 0 if "cuda" in device else -1
        self._pipe = pipeline("image-classification", model=self.hf_id, device=dev)

    def score(self, image: Image.Image) -> float:
        out = self._pipe(image.convert("RGB"), top_k=None)
        for d in out:
            if d["label"].lower() == "hq":
                return float(d["score"])
        for d in out:
            if d["label"].lower() == "lq":
                return float(1.0 - d["score"])
        return float(out[0]["score"])


class WaifuScorerV3(AestheticMetric):
    """CLIP ViT-L/14 features (L2-normalized) + a small trained MLP head.

    CLIP backbone: OpenAI's official `clip` package (not open_clip) -- the
    weights and preprocessing must match exactly for the MLP head below to be
    valid, since it was trained against that exact feature space.
    MLP head weights: Eugeoter/waifu-scorer-v3 (model.safetensors).
    """

    name = "waifu_scorer_v3"
    native_range = (0.0, 10.0)
    hf_id = "Eugeoter/waifu-scorer-v3"

    def load(self, device: str) -> None:
        import clip
        import torch
        import torch.nn as nn
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        class _MLP(nn.Module):
            def __init__(self, input_size: int = 768):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 2048), nn.ReLU(), nn.BatchNorm1d(2048), nn.Dropout(0.3),
                    nn.Linear(2048, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
                    nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.2),
                    nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
                    nn.Linear(128, 32), nn.ReLU(),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                return self.layers(x)

        self._torch = torch
        self._dev = torch.device(device)
        self._clip_model, self._clip_preprocess = clip.load("ViT-L/14", device=self._dev)

        weight_path = hf_hub_download(self.hf_id, "model.safetensors")
        mlp = _MLP(input_size=768)
        mlp.load_state_dict(load_file(weight_path))
        mlp.to(self._dev).eval()
        self._mlp = mlp

    def score(self, image: Image.Image) -> float:
        torch = self._torch
        with torch.no_grad():
            img = image.convert("RGB")
            # Official implementation duplicates the single image into a batch of
            # 2 to sidestep a BatchNorm1d batch=1 edge case; kept for parity.
            px = torch.stack([self._clip_preprocess(img), self._clip_preprocess(img)]).to(self._dev)
            feat = self._clip_model.encode_image(px)
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            feat = feat.to(dtype=torch.float32, device=self._dev)
            pred = self._mlp(feat)
            return float(pred.clamp(0, 10)[0].item())


METRICS: dict[str, type[AestheticMetric]] = {
    AestheticShadowV2.name: AestheticShadowV2,
    WaifuScorerV3.name: WaifuScorerV3,
}
