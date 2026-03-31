python ./scripts/evaluate.py \
  --backend transformers \
  --method lora \
  --base_model_dir ./model \
  --adapter_dir ./checkpoints/lora_boolq \
  --data_path ./data/boolq/validation.json \
  --save_dir ./results/main/lora_boolq \
  --max_length 512 \
  --max_new_tokens 8 \
  --batch_size 8
