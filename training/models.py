from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformers import CLIPModel, CLIPProcessor

ModelName = Literal["cnn", "clip"]


class SimpleCNN(nn.Module):
    """Lightweight CNN baseline for binary attack detection."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ClipClassifier(nn.Module):
    """CLIP vision encoder + linear head (fine-tune full model or freeze backbone)."""

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch16",
        num_classes: int = 2,
        freeze_vision: bool = False,
    ) -> None:
        super().__init__()
        from transformers import CLIPModel

        self.clip = CLIPModel.from_pretrained(model_id)
        embed_dim = self.clip.config.projection_dim
        self.head = nn.Linear(embed_dim, num_classes)

        if freeze_vision:
            for param in self.clip.vision_model.parameters():
                param.requires_grad = False
            for param in self.clip.visual_projection.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Same variable-resolution path as Grad-ECLIP attack (interpolate_pos_encoding).
        vision_outputs = self.clip.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=True,
        )
        image_features = self.clip.visual_projection(vision_outputs.pooler_output)
        return self.head(image_features)


def build_model(
    name: ModelName,
    *,
    num_classes: int = 2,
    clip_model_id: str = "openai/clip-vit-base-patch16",
    freeze_clip_vision: bool = False,
) -> nn.Module:
    if name == "cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "clip":
        return ClipClassifier(
            model_id=clip_model_id,
            num_classes=num_classes,
            freeze_vision=freeze_clip_vision,
        )
    raise ValueError(f"Unknown model name: {name}. Use 'cnn' or 'clip'.")


def build_clip_processor(model_id: str = "openai/clip-vit-base-patch16") -> "CLIPProcessor":
    from transformers import CLIPProcessor

    return CLIPProcessor.from_pretrained(model_id)
