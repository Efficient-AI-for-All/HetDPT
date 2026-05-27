"""stage 1 step 2 mixed attention/activation search.

This entry point is adapted from the original ICML supplementary code and
keeps the HetDPT training/evaluation behavior while exposing a cleaner public
file name for GitHub users.
"""

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

import torch.nn as nn
from timm.models.vision_transformer import Attention, Block
from hetdpt.search_layers import Learnable_Attention, Learnable_Block, Learnable_Gelu_Hard

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from timm.scheduler import create_scheduler
from hetdpt.lr_scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from hetdpt.datasets import build_dataset
from hetdpt.engine import train_one_epoch, evaluate
from hetdpt.losses import DistillationLoss
from hetdpt.samplers import RASampler
from hetdpt.augment import new_data_aug_generator

from hetdpt import search_layers as models

from hetdpt import utils

import bisect

class Prune_Num(object):
    def __init__(self, search_epochs, target_act_n, target_attn_n, multi_stage) -> None:
        '''
        `multi_stage` must be consistent with `target_act_n + target_attn_n` and `search_epochs`.
        example:
        multi_stage=[8,10,12,14,16,18,20,22], search_epochs = 120, target_act_n = 4, target_attn_n = 4
        For example, 8 + 10 + 12 + 14 + 16 + 18 + 20 + 22 equals 120.
        The list has 8 entries, matching `len(multi_stage) == target_act_n + target_attn_n`.
        This means epochs 0-7 are warmup; epochs 8-17 are stage 1, where the first pruning action happens at the end of epoch 17; later ranges are 18-29, 30-43, 44-59, 60-77, 78-97, 98-119, and 120-299.
        The learning-rate schedule should align with this, with peaks at epochs 8, 18, 30, 44, 60, 78, 98, and 120.
        '''
        self.search_epochs = search_epochs
        self.target_act_n = target_act_n
        self.target_attn_n = target_attn_n
        self.prune_order = self.get_prune_order(target_act_n,target_attn_n)
        self.interval = search_epochs // (target_act_n + target_attn_n)
        self.multi_stage = multi_stage
        self.get_cumulative()
        assert len(multi_stage) == target_act_n + target_attn_n
        assert sum(multi_stage) == search_epochs
        
    def get_prune_order(self, target_act_n, target_attn_n):
        if target_act_n >= target_attn_n:
            major_target = 'act'
            secondary_target = 'attn'
            repeat_time = target_act_n
            secondary_time = target_attn_n
        else:
            major_target = 'attn'
            secondary_target = 'act'
            repeat_time = target_attn_n
            secondary_time = target_act_n
        prune_order = []
        for i in range(repeat_time):
            prune_order.append(major_target)
            if secondary_time > 0 :
                prune_order.append(secondary_target)
                secondary_time -= 1
        return prune_order
    def get_cumulative(self):
        # Build the cumulative-sum list.
        self.cumulative = []
        total = 0
        for d in self.multi_stage:
            total += d
            self.cumulative.append(total)
    def get_prune_num(self,epoch):
        if epoch > self.search_epochs:
            return self.target_act_n, self.target_attn_n
        else:
            # cur_step = (epoch - self.interval) // self.interval
            cur_step = bisect.bisect_right(self.cumulative, epoch) - 1
            print(cur_step)
            if cur_step < 0:
                return 0,0
            else:
                exe_prune = self.prune_order[:cur_step+1]
                act_prune_num = exe_prune.count('act')
                attn_prune_num = exe_prune.count('attn')
                return act_prune_num, attn_prune_num
            
            
# This version updates pruning only when a new pruning target appears, and updates only the new target.
class Flag_Get(object):
    def __init__(self,candidate_act, candidate_attn) -> None:
        self.candidate_act = candidate_act
        self.candidate_attn = candidate_attn
        self.prune_act = 0
        self.prune_attn = 0
        self.flag_list = [1]*24
        self.pruned_act = []
        self.pruned_attn = []
    
    # slop param
    def get_flag_list(self, slope_param, prune_act, prune_attn):
        if self.prune_act == prune_act and self.prune_attn == prune_attn:
            return self.flag_list
        else:
            if self.prune_act == prune_act: # Need to prune an attention layer.
                attn_slope = slope_param[::2]
                candidate = {}
                for i, slope in enumerate(attn_slope):
                    if i not in self.pruned_attn and i in self.candidate_attn:
                        candidate[i] = slope
                min_attn = min(candidate, key=lambda k:candidate[k])
                self.flag_list[min_attn*2] = 0
                self.pruned_attn.append(min_attn)
                self.prune_attn += 1
                
                print('slope_param:',slope_param)
                print('attn_slope:', attn_slope)
                print('min_attn:', min_attn)
                print('pruned_attn', self.pruned_attn)
            else: # Need to prune an activation layer.
                act_slope = slope_param[1::2]
                candidate = {}
                for i, slope in enumerate(act_slope):
                    if i not in self.pruned_act and i in self.candidate_act:
                        candidate[i] = slope
                min_act = min(candidate, key=lambda k:candidate[k])
                self.flag_list[min_act*2 + 1] = 0
                self.pruned_act.append(min_act)
                self.prune_act += 1
                
                print('slope_param:',slope_param)
                print('act_slope:', act_slope)
                print('min_act:', min_act)
                print('pruned_act', self.pruned_act)
            return self.flag_list

def swap_activation(model, current="", device='cpu'):
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        fqn = f"{current}.{name}" if current else name
        if isinstance(child, torch.nn.GELU):
            print(f'swapping act at {fqn}')
            swap_act = Learnable_Gelu_Hard(device = device)
            setattr(model, name, swap_act)
            del child
        else:
            swap_activation(child, current=fqn, device=device)

def swap_attn4model(model, current="", device='cpu'):
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        fqn = f"{current}.{name}" if current else name
        if isinstance(child, Attention):
            print(f'swapping attn at {fqn}')
            swap_attn = Learnable_Attention(attention=child, device = device)
            setattr(model, name, swap_attn)
            # del child
        else:
            swap_attn4model(child, current=fqn, device=device)
            
def swap_block4model(model, current="", device='cpu'):
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        fqn = f"{current}.{name}" if current else name
        if isinstance(child, Block):
            print(f'swapping block at {fqn}')
            swap_attn = Learnable_Block(block=child, device = device)
            setattr(model, name, swap_attn)
            # del child
        else:
            swap_block4model(child, current=fqn, device=device)               
        
        

def get_args_parser():
    parser = argparse.ArgumentParser('HetDPT stage 1 step 2 mixed attention/activation search', add_help=False)
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cyclecosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=15, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=False)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='deit_small', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=True, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # candidate and search
    parser.add_argument('--search_epochs', default=80, type=int, help='Number of epochs used for mixed attention/activation search.')
    parser.add_argument('--target_act_n', default=4, type=int, help='Target number of activation layers to prune during search.')
    parser.add_argument('--target_attn_n', default=4, type=int, help='Target number of attention layers to prune during search.')
    parser.add_argument('--candidate_act', default=[9,10,11,8,7,4,6,1], type=list, help='Candidate activation-layer indices considered by HetDPT search.')
    parser.add_argument('--candidate_attn', default=[1,11,9,10,8,7,6,5], type=list, help='Candidate attention-layer indices considered by HetDPT search.')
    parser.add_argument('--multi_stage', default=[3,5,7,9,11,13,15,17], type=list, help='Epoch milestones for progressively increasing pruning pressure.')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)
    prune_num = Prune_Num(search_epochs=args.search_epochs, target_act_n=args.target_act_n, target_attn_n=args.target_attn_n)
    flag_get = Flag_Get(candidate_act=args.candidate_act, candidate_attn=args.candidate_attn)
    # if args.distillation_type != 'none' and args.finetune and not args.eval:
    #     raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )

                    
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
    swap_activation(model,device=device)
    swap_attn4model(model,device=device)
    swap_block4model(model,device=device)
    output_dir = Path(args.output_dir)
    slope_param = []
    for name, param in model.named_parameters():
        print(name)
        if 'learn_mask' in name:
            slope_param.append(param.data.item())
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(str(slope_param) + '\n')
            
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
            
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    # output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args = args,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
             

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
            
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        
        
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        prune_act, prune_attn = prune_num.get_prune_num(epoch)
        slope_param = []
        for name, param in model.named_parameters():
            if 'learn_mask' in name:
                slope_param.append(param.data.item())
        if args.output_dir and utils.is_main_process():
            with (output_dir/"log.txt").open("a") as f:
                f.write(str(slope_param)+'\n')
        
        flag_list = flag_get.get_flag_list(slope_param=slope_param, prune_act=prune_act, prune_attn=prune_attn)
        
        i = 0
        for module in model.modules():
            if isinstance(module, Learnable_Attention) or isinstance(module, Learnable_Gelu_Hard):
                module.set_flag(flag_list[i])
                i += 1
        if args.output_dir and utils.is_main_process():
            with (output_dir/"log.txt").open("a") as f:
                f.write(str(i)+'\n')
                f.write(str(flag_list)+'\n')
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HetDPT stage 1 step 2 mixed attention/activation search', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
