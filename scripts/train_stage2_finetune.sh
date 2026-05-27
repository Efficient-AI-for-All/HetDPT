#!/usr/bin/env bash
set -euo pipefail

python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --master_port=21912 \
  --use_env train_stage2.py \
  --data-set IMNET \
  --data-path /path/to/imagenet \
  --model deit_base_distilled_patch16_224_lambda_shrink \
  --epochs 400 \
  --teacher-model deit_base_patch16_224 \
  --teacher-path /path/to/deit_base_patch16_224.pth \
  --finetune /path/to/deit_base_distilled_patch16_224.pth \
  --prune_layer_attn '[0,3,7,8,11]' \
  --prune_layer_act '[2,7,8,10,11]' \
  --batch-size 512 \
  --output_dir ./outputs/stage2_finetune \
  --dist-eval
