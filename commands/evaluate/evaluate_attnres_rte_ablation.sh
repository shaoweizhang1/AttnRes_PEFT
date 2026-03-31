for lb in 1 2 4 8 16 20 28; do
  python ./scripts/evaluate.py \
    --backend transformers \
    --method attnres \
    --base_model_dir ./model \
    --adapter_dir ./checkpoints/ablation/attnres_rte_lookback_${lb} \
    --data_path ./data/rte/validation.json \
    --save_dir ./results/ablation/attnres_rte_lookback_${lb} \
    --max_length 256 \
    --max_new_tokens 8 \
    --batch_size 8 \
    --attnres_lookback ${lb} \
    --attnres_gate_init 0.0
 done
