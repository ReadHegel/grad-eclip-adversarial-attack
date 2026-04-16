from __future__ import annotations

from typing import Any

from transformers import AutoModel, AutoProcessor

from .clip_model import ClipModel


class GoogleSiglip2B32_256Clip(ClipModel):
    def __init__(self, device: str | None = None, load_on_init: bool = True) -> None:
        super().__init__(
            model_id="google/siglip2-base-patch32-256",
            device=device,
            load_on_init=load_on_init,
        )
        self.cls_token = False

