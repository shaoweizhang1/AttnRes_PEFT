python ./scripts/evaluate.py \
  --backend transformers \
  --method lora \
  --base_model_dir ./model \
  --adapter_dir ./checkpoints/lora_rte \
  --data_path ./data/rte/validation.json \
  --save_dir ./results/main/lora_rte \
  --max_length 256 \
  --max_new_tokens 8 \
  --batch_size 8
