#!/usr/bin/env bash
set -euo pipefail

python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port=14020 \
  --use_env train_stage1_step1.py \
  --data-set IMNET \
  --data-path /path/to/imagenet \
  --model deit_base_patch16_224_copy_lambda_shrink \
  --epochs 5 \
  --finetune /path/to/deit_base_patch16_224.pth \
  --prune_layer_attn '[]' \
  --prune_layer_act '[7]' \
  --batch-size 256 \
  --output_dir ./outputs/stage1_step1_activation \
  --dist-eval
