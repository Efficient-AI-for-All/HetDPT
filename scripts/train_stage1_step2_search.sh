#!/usr/bin/env bash
set -euo pipefail

python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --master_port=21911 \
  --use_env train_stage1_step2.py \
  --data-set IMNET \
  --data-path /path/to/imagenet \
  --model deit_base_patch16_224 \
  --finetune /path/to/deit_base_patch16_224.pth \
  --target_act_n 5 \
  --target_attn_n 5 \
  --candidate_act '[9,10,11,8,7,4,6,1]' \
  --candidate_attn '[1,11,9,10,8,7,6,5]' \
  --output_dir ./outputs/stage1_step2_search
