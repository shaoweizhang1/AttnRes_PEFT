python ./scripts/evaluate.py \
  --backend transformers \
  --method lora \
  --base_model_dir ./model \
  --adapter_dir ./checkpoints/lora_gsm8k \
  --data_path ./data/gsm8k/test.json \
  --save_dir ./results/main/lora_gsm8k \
  --max_length 1024 \
  --max_new_tokens 64 \
  --batch_size 8
