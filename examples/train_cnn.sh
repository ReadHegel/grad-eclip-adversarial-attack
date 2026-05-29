uv run python -m training.train \
  --model cnn \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-4 \
  --device cuda \
  --num-workers 2 \
  --plot-roc