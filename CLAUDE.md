# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements **Grad-ECLIP adversarial attacks** on CLIP (Contrastive Language-Image Pre-training) models. It provides:
- **Grad-ECLIP explainability**: Generates visual heatmaps showing which image regions drive image-text similarity
- **Adversarial attacks**: Gradient-based optimization that manipulates CLIP explanations while preserving embeddings

## Setup

```bash
uv sync
cp .env.template .env
# Fill in HF_TOKEN (HuggingFace API token) and HF_HOME (cache directory)
export $(cat .env | xargs)
```

## Running

**Smoke test** (loads all models, generates heatmaps to `Images/Output/`):
```bash
uv run python main_test.py
```

**Full attack experiments**:
```bash
uv run jupyter notebook broken_grad_eclip_HF_v4.ipynb
```

PyTorch is installed from the `pytorch-cu124` index (configured in `pyproject.toml` via `[tool.uv.sources]`). The `requirements.txt` is kept as a historical reference but `uv sync` is the canonical way to install dependencies.

## Architecture

### Model System

All models share a common abstract base class in `models/clip_model.py`:

- `encode_dense(image, text)` — Extracts Q, K, V tensors and attention outputs from the **final transformer layer** via forward hooks; returns intermediate representations needed for gradient computation
- `explain(image, text)` — Computes Grad-ECLIP heatmap: gradients of image-text cosine similarity w.r.t. attention outputs, combined with V values and Q-K cosine similarity, ReLU'd and reshaped to a spatial grid
- `forward(image, text)` — Standard inference returning cosine similarity
- `get_text_features(text)` / `get_image_features(image)` — Standalone embedding extraction
- GPU management: `move_to_gpu(gpu_index)`, `offload_from_gpu()`, `unload_model()`

### Model Registry

`models/__init__.py` exports `build_clip_model(model_key)` as the factory entry point. Registered keys:

| Key | Model |
|-----|-------|
| `openai_vit_b32` | openai/clip-vit-base-patch32 |
| `openai_vit_b16` | openai/clip-vit-base-patch16 |
| `openai_vit_l14` | openai/clip-vit-large-patch14 |
| `google_siglip2_b32_256` | google/siglip2-b32-256 |
| `facebook_metaclip2_b16` | facebook/metaclip-2-worldwide-b16 |

Adding a new model requires only a minimal subclass that sets `model_id` and `cls_token`.

### Data Paths

- Input images: `Images/SampleImages/`
- Output heatmaps: `Images/Output/`

### Adversarial Attack Pattern (from notebook)

1. Load image + text, generate baseline explanation heatmap
2. Initialize perturbation delta (bounded by `DELTA`, applied via sigmoid)
3. Optimization loop with Adam: minimize `explanation_MSE + embedding_preservation_loss`
4. Visualize embedding trajectories in 3D

### Key Implementation Notes

- `encode_dense()` uses PyTorch forward hooks to capture intermediate tensors — hooks must be registered before the forward pass and removed after
- `cls_token` flag on each model controls whether the CLS token position is included or excluded when reshaping attention outputs to spatial grids
- Images are resized to multiples of the model's patch size; heatmaps are upsampled back to original resolution
- Only OpenAI CLIP variants are fully tested; Google SigLIP-2 and Facebook MetaCLIP-2 wrappers exist but may need additional work
