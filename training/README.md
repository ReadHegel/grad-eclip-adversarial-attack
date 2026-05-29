# Training — adversarial attack recognition

Binary classification on [ReadHegel/openai-vit-b16-adv-recognition](https://huggingface.co/datasets/ReadHegel/openai-vit-b16-adv-recognition): predict `is_attacked` (0 = clean COCO image, 1 = Grad-ECLIP adversarial image). See `DatasetUtils/prepare_dataset.py` for how the dataset was built.

| Split | Samples |
|-------|---------|
| train | 4000    |
| test  | 1000    |

All downloads and caches use **`/tmp`** (HF datasets, transformers, checkpoints).

**Preprocessing** matches Grad-ECLIP attack generation (`training/preprocess.py`): patch-grid alignment (e.g. 640×478 → 640×480), no resize to 224, no augmentations. CLIP uses the same `ToTensor` + CLIP `Normalize` as in `models/clip_model.py`; CNN uses `ToTensor` only in `[0, 1]`.

Use a **smaller batch size** than for 224×224 (e.g. 8–16) — full-resolution tensors are large.

## Quick start

```bash
# Fast smoke check (dataset + one forward pass; add --clip to test CLIP weights)
uv run python -m training.smoke

# CNN baseline (config at top of train.py)
uv run python -m training.train

# CLIP fine-tuning
uv run python -m training.train --model clip

# Train both and compare test metrics
uv run python -m training.train --all

# Override hyperparameters
uv run python -m training.train --model cnn --epochs 20 --batch-size 64 --lr 3e-4

# Log ROC-AUC each epoch and save ROC plots to the run directory
uv run python -m training.train --model cnn --epochs 10 --plot-roc
```

Edit defaults in the **configuration block** at the top of `training/train.py` (`MODEL`, `NUM_EPOCHS`, `LEARNING_RATE`, `FREEZE_CLIP_VISION`, etc.).

## Models

| Key   | Class            | Description                          |
|-------|------------------|--------------------------------------|
| `cnn` | `SimpleCNN`      | Small convolutional baseline         |
| `clip`| `ClipClassifier` | `openai/clip-vit-base-patch16` + linear head |

Checkpoints are saved under `/tmp/adv_recognition_checkpoints/<model>_<timestamp>/`.

## Adding a new model

1. Implement `nn.Module` in `training/models.py`.
2. Register it in `build_model()`.
3. Add a choice to `--model` in `train.py` and optionally to `--all`.
