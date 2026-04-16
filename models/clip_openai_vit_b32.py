from __future__ import annotations

from typing import Any

from transformers import AutoModel, AutoProcessor
import torch

from .clip_model import ClipModel, EncodeDenseOutput

class OpenAIViTB32Clip(ClipModel):
    def __init__(self, device: str | None = None, load_on_init: bool = True) -> None:
        super().__init__(
            model_id="openai/clip-vit-base-patch32",
            device=device,
            load_on_init=load_on_init,
        )
        self.cls_token = True
