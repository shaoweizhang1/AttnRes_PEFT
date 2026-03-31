python ./scripts/evaluate.py \
  --backend transformers \
  --method base \
  --model_dir ./model \
  --data_path ./data/rte/validation.json \
  --save_dir ./results/main/base_rte \
  --max_length 256 \
  --max_new_tokens 8 \
  --batch_size 8
