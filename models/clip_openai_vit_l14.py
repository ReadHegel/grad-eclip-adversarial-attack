from __future__ import annotations

from typing import Any

from transformers import AutoModel, AutoProcessor

from .clip_model import ClipModel


class OpenAIViTL14Clip(ClipModel):
    def __init__(self, device: str | None = None, load_on_init: bool = True) -> None:
        super().__init__(
            model_id="openai/clip-vit-large-patch14",
            device=device,
            load_on_init=load_on_init,
        )

