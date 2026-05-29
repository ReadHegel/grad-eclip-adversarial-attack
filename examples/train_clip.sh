uv run python -m training.train \
  --model clip \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-5 \
  --device cuda \
  --num-workers 2 \
  --plot-roc