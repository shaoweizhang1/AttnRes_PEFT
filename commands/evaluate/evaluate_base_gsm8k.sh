python ./scripts/evaluate.py \
  --backend transformers \
  --method base \
  --model_dir ./model \
  --data_path ./data/gsm8k/test.json \
  --save_dir ./results/main/base_gsm8k \
  --max_length 1024 \
  --max_new_tokens 64 \
  --batch_size 8
