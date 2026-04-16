from .clip_model import ClipModel
from .clip_openai_vit_b32 import OpenAIViTB32Clip
from .clip_openai_vit_b16 import OpenAIViTB16Clip
from .clip_openai_vit_l14 import OpenAIViTL14Clip
from .clip_google_siglip2_b32_256 import GoogleSiglip2B32_256Clip
from .clip_facebook_metaclip2_b16 import FacebookMetaClip2B16Clip


CLIP_MODEL_REGISTRY = {
    "openai-vit-b32": OpenAIViTB32Clip,
    "openai-vit-b16": OpenAIViTB16Clip,
    "openai-vit-l14": OpenAIViTL14Clip,
    "google-siglip2-b32-256": GoogleSiglip2B32_256Clip,
    "facebook-metaclip2-b16": FacebookMetaClip2B16Clip,
}


def build_clip_model(model_key: str, **kwargs):
    """Simple factory to build one of registered CLIP model wrappers."""
    if model_key not in CLIP_MODEL_REGISTRY:
        available = ", ".join(sorted(CLIP_MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model_key={model_key}. Available: {available}")
    return CLIP_MODEL_REGISTRY[model_key](**kwargs)


__all__ = [
    "ClipModel",
    "OpenAIViTB32Clip",
    "OpenAIViTB16Clip",
    "OpenAIViTL14Clip",
    "GoogleSiglip2B32_256Clip",
    "FacebookMetaClip2B16Clip",
    "build_clip_model",
    "CLIP_MODEL_REGISTRY",
]
