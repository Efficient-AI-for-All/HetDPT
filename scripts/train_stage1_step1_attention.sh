#!/usr/bin/env bash
set -euo pipefail

python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port=14019 \
  --use_env train_stage1_step1.py \
  --data-set IMNET \
  --data-path /path/to/imagenet \
  --model deit_base_patch16_224_copy_lambda_shrink \
  --epochs 30 \
  --finetune /path/to/deit_base_patch16_224.pth \
  --prune_layer_attn '[0]' \
  --prune_layer_act '[]' \
  --batch-size 256 \
  --output_dir ./outputs/stage1_step1_attention \
  --dist-eval
