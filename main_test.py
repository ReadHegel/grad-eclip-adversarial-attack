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
    "openai-vit-b32",
    "openai-vit-b16",
    "openai-vit-l14",
    "google-siglip2-b32-256",
    "facebook-metaclip2-b16",
]

def visualize(hmap, image, out_path):
    print(f"Original image shape: {image.size}")
    print(f"Heatmap shape: {hmap.shape}")
    resize = torchvision.transforms.Resize(image.size[::-1], interpolation=torchvision.transforms.InterpolationMode.NEAREST)

    image = np.asarray(image.copy())
    hmap = resize(hmap.unsqueeze(0))[0].detach().cpu().numpy()

    color = cv2.applyColorMap((hmap*255).astype(np.uint8), cv2.COLORMAP_JET) # cv2 to plt
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
    image = Image.open("Images/SampleImages/dog_and_car.png").convert("RGB")
    text = "a dog in a car waiting for traffic lights"

    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    print(device)

    for model_key in MODEL_KEYS:
        print(f"\n=== Testing {model_key} ===")
        model = build_clip_model(model_key, device="cpu", load_on_init=False)

        #try:
        model.load_model()

        model.print_model_info()

        #print(inspect.getsource(model.model.vision_model.head.forward))


        emap, _ = model.explain(image=image, text=text)
        visualize(emap, image, f"Images/Output/output_{model_key}.png")

    


        if use_cuda:
            model.move_to_gpu(0)

        outputs = model.forward(image=image, text=text)
        output_keys = list(outputs.keys()) if hasattr(outputs, "keys") else []
        print(f"Forward OK. Output keys: {output_keys}")

        # except Exception as exc:
        #     print(f"Forward FAILED for {model_key}: {exc}")

        #finally:
        # Free VRAM after each model before loading the next one.
        model.offload_from_gpu()
        model.unload_model()

    print(f"\nSmoke test finished on device: {device}")


if __name__ == "__main__":
    load_env()
    run_forward_smoke_test()
