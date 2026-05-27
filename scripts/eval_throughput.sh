#!/usr/bin/env bash
set -euo pipefail

python measure_throughput.py \
  --data-set IMNET \
  --model deit_base_distilled_patch16_224_lambda_shrink \
  --finetune /path/to/hetdpt_base.pth \
  --prune_layer_attn '[0,3,7,8,11]' \
  --prune_layer_act '[2,7,8,10,11]' \
  --batch-size 256
