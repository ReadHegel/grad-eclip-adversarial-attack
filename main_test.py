from __future__ import annotations

from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt

from models import build_clip_model


import os
from dotenv import load_dotenv
import inspect
import numpy as np
import cv2

MODEL_KEYS = [
    # "openai-vit-b32",
    "openai-vit-b16",
    # "openai-vit-l14",
    # "google-siglip2-b32-256",
    # "facebook-metaclip2-b16",
]


def visualize(hmap, image, out_path):
    print(f"Original image shape: {image.size}")
    print(f"Heatmap shape: {hmap.shape}")
    resize = torchvision.transforms.Resize(
        image.size[::-1], interpolation=torchvision.transforms.InterpolationMode.NEAREST
    )

    image = np.asarray(image.copy())
    hmap = resize(hmap.unsqueeze(0))[0].detach().cpu().numpy()

    color = cv2.applyColorMap((hmap * 255).astype(np.uint8), cv2.COLORMAP_JET)  # cv2 to plt
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    c_ret = np.clip(image * (1 - 0.5) + color * 0.5, 0, 255).astype(np.uint8)

    cv2.imwrite(out_path, cv2.cvtColor(c_ret, cv2.COLOR_RGB2BGR))


def load_env():
    load_dotenv()
    if "HF_TOKEN" not in os.environ:
        print("Ostrzeżenie: HF_TOKEN nie został znaleziony w zmiennych środowiskowych.")


def tensor_stats(t, name="Tensor"):
    print(f"--- {name} ---")
    print(f"Shape:  {list(t.shape)}")
    print(f"Dtype:  {t.dtype}")
    print(f"Device: {t.device}")
    print(f"Min:    {t.min().item():.4f}")
    print(f"Max:    {t.max().item():.4f}")
    print(f"Mean:   {t.mean().item():.4f}")
    print(f"Std:    {t.std().item():.4f}")
    print(f"NaNs:   {torch.isnan(t).sum().item()}")


def run_forward_smoke_test() -> None:
    image = Image.open("DatasetUtils/data/generated_datasets/openai-vit-b16-adv-recognition-ds/images/test/test_000002.jpg").convert("RGB")
    text = "A group of guys standing behind tables on a stage before a presentation."

    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    print(device)

    for model_key in MODEL_KEYS:
        print(f"\n=== Testing {model_key} ===")
        model = build_clip_model(model_key, device=device, load_on_init=False)

        model.load_model()
        # model.print_model_info()

        # for model_key_prim in MODEL_KEYS:
        #     ruined_img = Image.open(f"DatasetUtils/data/generated_datasets/openai-vit-b16-adv-recognition-ds/images/test/test_000001.jpg").convert("RGB")

        emap, _ = model.explain(image=image, text=text)
        visualize(emap, image, f"tmp.png")

        # target_img = Image.open("Images/targets/smiley.gif").convert("RGBA")
        # target_tensor = 1 - torchvision.transforms.ToTensor()(target_img)[0, :, :]

        # ruined_img, _, _, _ = model.ruin(image, text, target_tensor)
        # ruined_img.save(f"Images/Output/ruined_{model_key}.png")

        # emap_ruined, _ = model.explain(image=ruined_img, text=text)
        # visualize(emap_ruined, ruined_img, f"Images/Output/emap_ruined_{model_key}.png")

        # finally:
        # Free VRAM after each model before loading the next one.
        model.offload_from_gpu()
        model.unload_model()

    print(f"\nSmoke test finished on device: {device}")


if __name__ == "__main__":
    load_env()
    run_forward_smoke_test()
