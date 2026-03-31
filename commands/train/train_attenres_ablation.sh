for lb in 16 8 4 2 1; do
  python ./scripts/train.py \
    --method attnres \
    --model_dir ./model \
    --train_path ./data/rte/train.json \
    --save_dir ./checkpoints/ablation/attnres_rte_lookback_${lb} \
    --max_length 256 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --logging_steps 10 \
    --save_strategy epoch \
    --attnres_lookback ${lb} \
    --attnres_gate_init 0.0 \
    --use_wandb \
    --wandb_project  \
    --wandb_run_name  \
    --wandb_entity 
 done
