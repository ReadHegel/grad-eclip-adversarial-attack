"""
Image preprocessing aligned with Grad-ECLIP attack generation (models/clip_model.py).

Attack path (ruin):
  1. _proccess_keepsize — bicubic resize so H,W are divisible by ViT patch size (no square crop)
  2. ToTensor — values in [0, 1]
  3. Normalize(CLIP mean/std) — only inside the attack loop for CLIP forward; saved PNGs are
     clamped RGB in [0, 1] before ToPILImage.

Training uses the same geometry and (for CLIP) the same normalization as at attack time.
No RandomFlip, no ImageNet resize to 224, no extra augmentations.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch16"


@lru_cache(maxsize=4)
def get_clip_preprocess_config(model_id: str) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    """CLIP image_mean, image_std, and vision patch size (matches attack model)."""
    from transformers import CLIPConfig, CLIPImageProcessor

    processor = CLIPImageProcessor.from_pretrained(model_id)
    config = CLIPConfig.from_pretrained(model_id)
    mean = tuple(float(x) for x in processor.image_mean)
    std = tuple(float(x) for x in processor.image_std)
    patch_size = int(config.vision_config.patch_size)
    return mean, std, patch_size


def align_image_to_patch_grid(image: Image.Image, patch_size: int) -> Image.Image:
    """
    Same geometry as ClipModel._proccess_keepsize: round spatial dims to patch multiples (bicubic).
    """
    image = image.convert("RGB")
    width, height = image.size
    new_width = int(width / patch_size + 0.5) * patch_size
    new_height = int(height / patch_size + 0.5) * patch_size
    if (new_width, new_height) == (width, height):
        return image
    resize = transforms.Resize((new_height, new_width), interpolation=InterpolationMode.BICUBIC)
    return resize(image)


def build_preprocess_transform(
    model_name: str,
    *,
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID,
) -> Callable[[Image.Image], "torch.Tensor"]:
    """
    Single transform for both train and eval (no augmentation).

    - cnn: patch-grid align + ToTensor ([0, 1], preserves subtle L_inf perturbations)
    - clip: patch-grid align + ToTensor + CLIP Normalize (identical to attack forward input)
    """
    import torch

    if model_name == "clip":
        mean, std, patch_size = get_clip_preprocess_config(clip_model_id)

        def transform(image: Image.Image) -> torch.Tensor:
            image = align_image_to_patch_grid(image, patch_size)
            tensor = transforms.ToTensor()(image)
            return transforms.Normalize(mean=mean, std=std)(tensor)

        return transform

    if model_name == "cnn":
        _, _, patch_size = get_clip_preprocess_config(clip_model_id)

        def transform(image: Image.Image) -> torch.Tensor:
            image = align_image_to_patch_grid(image, patch_size)
            return transforms.ToTensor()(image)

        return transform

    raise ValueError(f"Unknown model_name: {model_name!r}. Use 'cnn' or 'clip'.")
