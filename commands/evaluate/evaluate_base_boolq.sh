python ./scripts/evaluate.py \
  --backend transformers \
  --method base \
  --model_dir ./model \
  --data_path ./data/boolq/validation.json \
  --save_dir ./results/main/base_boolq \
  --max_length 512 \
  --max_new_tokens 8 \
  --batch_size 8
