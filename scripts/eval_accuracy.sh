#!/usr/bin/env bash
set -euo pipefail

python evaluate.py \
  --data-set IMNET \
  --data-path /path/to/imagenet \
  --model deit_base_distilled_patch16_224_lambda_shrink \
  --finetune /path/to/hetdpt_base.pth \
  --prune_layer_attn '[0,3,7,8,11]' \
  --prune_layer_act '[2,7,8,10,11]' \
  --batch-size 256 \
  --output_dir ./outputs/eval_base \
  --dist-eval \
  --eval
