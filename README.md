# HetDPT

Official public code cleanup for **HetDPT: Rethinking Depth Pruning for Vision Transformers: A Heterogeneity-Aware Perspective**.

This repository is organized for GitHub readers: the original supplementary code has been flattened into clear entry points plus a small `hetdpt/` helper package. The original codebase called the project BoundaryDPT in a few places; the public-facing name is unified here as **HetDPT** to match the accepted paper.

## What is included

- Stage 1 Step 1 candidate fine-tuning: `train_stage1_step1.py`
- Stage 1 Step 2 heterogeneous attention/activation search: `train_stage1_step2.py`
- Stage 2 HetDPT fine-tuning: `train_stage2.py`
- Accuracy evaluation: `evaluate.py`
- Throughput measurement: `measure_throughput.py`
- Shared model, dataset, engine, loss, sampler, and pruning utilities in `hetdpt/`

The first public release only reorganizes the provided code. It does not add unprepared HetDPT+ code and does not attempt to retrain full ImageNet as part of repository validation.

## Installation

```bash
conda create -n hetdpt python=3.9
conda activate hetdpt
pip install -r requirements.txt
```

The original experiments use ImageNet-style datasets and DeiT checkpoints. Download datasets/checkpoints separately and pass their paths through the script arguments below.

## Project layout

```text
README.md                 # public usage guide
requirements.txt          # lightweight dependencies from the provided code
train_stage1_step1.py     # candidate pruning fine-tuning
train_stage1_step2.py     # mixed HetDPT search
train_stage2.py           # final fine-tuning
evaluate.py               # top-1 accuracy evaluation
measure_throughput.py     # throughput measurement
hetdpt/                   # shared implementation modules
scripts/                  # readable command templates
```

## Quick evaluation example

```bash
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
```

## Throughput example

```bash
python measure_throughput.py \
  --data-set IMNET \
  --model deit_base_distilled_patch16_224_lambda_shrink \
  --finetune /path/to/hetdpt_base.pth \
  --prune_layer_attn '[0,3,7,8,11]' \
  --prune_layer_act '[2,7,8,10,11]' \
  --batch-size 256
```

## Training pipeline

HetDPT follows the two-stage depth-pruning pipeline described in the paper. Use the scripts in `scripts/` as editable templates rather than one-size-fits-all reproduction commands.

1. `scripts/train_stage1_step1_attention.sh` or `scripts/train_stage1_step1_activation.sh`: estimate candidate behavior for a single layer type.
2. `scripts/train_stage1_step2_search.sh`: run mixed attention/activation search.
3. `scripts/train_stage2_finetune.sh`: fine-tune the selected pruned model.

Important parameters:

- `--data-path`: ImageNet root directory.
- `--finetune`: checkpoint loaded before evaluation or fine-tuning.
- `--teacher-model` and `--teacher-path`: optional distillation teacher settings.
- `--prune_layer_attn`: attention-layer indices removed by HetDPT.
- `--prune_layer_act`: activation-layer indices removed by HetDPT.
- `--output_dir`: output directory for logs and checkpoints.

