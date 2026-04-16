from __future__ import annotations

from typing import Any


from .clip_model import ClipModel


class FacebookMetaClip2B16Clip(ClipModel):
    def __init__(self, device: str | None = None, load_on_init: bool = True) -> None:
        super().__init__(
            model_id="facebook/metaclip-2-worldwide-b16",
            device=device,
            load_on_init=load_on_init,
        )

