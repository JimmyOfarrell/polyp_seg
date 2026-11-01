import torch
import torch.optim as optim
import math
import wandb
import argparse
import os
import sys
import shlex
from segment_anything import sam_model_registry
from utils.utils import disable_batchnorm_running_stats, str2bool, set_seed
from utils.model_utils import get_model, get_standard_model
from utils.dataset_utils import get_train_val_dataloaders
from utils.scheduler import WarmupCosineSchedule, WarmupLinearSchedule
from train import train_single_epoch
from utils.saver import Saver

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def main(args=None, sam_args=None, saver=None):
    requested_device = args['device']
    requested_cuda_amp = args['use_cuda_amp']
    cuda_available = torch.cuda.is_available()
    use_sam = args['use_sam']

    print("[MAIN] Device configuration:")
    if str(requested_device).lower() == 'cuda' and cuda_available:
        num_gpus = torch.cuda.device_count()
        device_id = int(args['device_id'])
        if device_id >= num_gpus or device_id < 0:
            print(f"[WARNING] device_id={device_id} is out of range (only {num_gpus} GPU(s) available). Using device 0.")
            device_id = args['device_id'] = 0
        args['device'] = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)
        gpu_name = torch.cuda.get_device_name(device_id)
        gpu_props = torch.cuda.get_device_properties(device_id)
        total_vram_gb = gpu_props.total_memory / (1024**3)
        cc = (gpu_props.major, gpu_props.minor)
        print(f"[MAIN] Device            : {args['device']}")
        print(f"[MAIN] GPU name          : {gpu_name}")
        print(f"[MAIN] Total VRAM        : {total_vram_gb:.2f} GB")
        print(f"[MAIN] CUDA version      : {torch.version.cuda}")
        print(f"[MAIN] PyTorch version   : {torch.__version__}")
        print(f"[MAIN] Compute capability: {cc[0]}.{cc[1]}")

        if cc < (6, 0) and requested_cuda_amp: print(f"[WARNING] GPU compute capability {cc[0]}.{cc[1]} is below 6.0. Mixed precision (AMP) may not work properly.")
        if cc < (7, 0) and requested_cuda_amp: print(f"[WARNING] GPU compute capability {cc[0]}.{cc[1]} is below 7.0. Tensor Cores not available; AMP will be limited.")

        try:
            free_vram, total_vram = torch.cuda.mem_get_info(device_id)
            free_vram_gb = free_vram / (1024**3)
            print(f"[MAIN] Free VRAM     : {free_vram_gb:.2f} GB")

            if use_sam and free_vram_gb < 8:
                print(f"[WARNING] Free VRAM ({free_vram_gb:.2f} GB) may be insufficient for Phase 1 with SAM (ViT-B). Consider:")
                print(f"         - Reduce batch size if applicable")
                print(f"         - Using --use_cuda_amp True")
                print(f"         - Switching to smaller SAM model")
        except Exception as e: print(f"[WARNING] Could not query VRAM: {type(e).__name__}: {e}")
    else:
        if str(requested_device).lower() == 'cuda' and not cuda_available: print(f"[WARNING] CUDA requested but not available. Falling back to CPU.")
        args['device'] = torch.device("cpu")
        print(f"[MAIN] Device         : CPU")
        print(f"[MAIN] PyTorch version: {torch.__version__}")
        if use_sam: print(f"[WARNING] Running Phase 1 with SAM on CPU will be extremely slow! Consider using a GPU for training.")
        if requested_cuda_amp:
            print(f"[WARNING] AMP is enabled but device is CPU. Disabling AMP.")
            args['use_cuda_amp'] = False
    print()

    device = args['device']
    use_cuda_amp = args['use_cuda_amp']
    disable_bn = args['disable_batchnorm_running_stats']

    if use_sam:
        config = {'trainable': 'all', 'sam_loaded': True, 'loss': 'gt'}
        stage = None
        pl_status = 'trainable'
        distill_alpha = None
        scenario_label = 'Phase 1 (SAM-guided)'
        loss_desc = 'l(p, gt)'
    elif args['phase_2_mode'] == 'single':
        pl_frozen = not args['single_pl_trainable']
        loss_type = args['single_loss_type']
        config = { 'trainable': 'aggregator' if pl_frozen else 'all', 'sam_loaded': (loss_type == 'mixed'), 'loss': loss_type}
        stage = None
        pl_status = 'frozen' if pl_frozen else 'trainable'
        distill_alpha = args['single_distill_weight'] if loss_type == 'mixed' else None
        scenario_label = f"Single | PL={pl_status} | loss={loss_type}"
        loss_desc = {'gt':    'l(p, gt)', 'mixed': f'l(p, sam+gt) [alpha={args["single_distill_weight"]}]'}[loss_type]
    else:
        stage = args['stage']
        if stage == 1:
            loss_type = args['stage1_loss_type']
            config = {'trainable': 'aggregator', 'sam_loaded': True, 'loss': loss_type}
            pl_status = 'frozen'
            distill_alpha = args['stage1_distill_weight'] if loss_type == 'mixed' else None
            loss_desc = {'sam':   'l(p, sam)', 'mixed': f'l(p, sam+gt) [alpha={args["stage1_distill_weight"]}]'}[loss_type]
        else:
            config = {'trainable': 'all', 'sam_loaded': False, 'loss': 'gt'}
            pl_status = 'trainable'
            distill_alpha = None
            loss_desc = 'l(p, gt)'
        scenario_label = f"Two-stage | Stage {stage} | loss={config['loss']}" 
    sam_needed = args['sam_needed']  
    
    if sam_needed:
        print("[MAIN] SAM model Configuration:")
        try: sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
        except Exception as e: raise RuntimeError(f"Failed to load SAM '{sam_args['model_type']}' from '{sam_args['sam_checkpoint']}': {e}")
        sam.to(device=device)
        sam_params = sum(p.numel() for p in sam.parameters())
        sam_image_encoder_params = sum(p.numel() for p in sam.image_encoder.parameters())
        sam_prompt_encoder_params = sum(p.numel() for p in sam.prompt_encoder.parameters())
        mask_decoder_params = sum(p.numel() for p in sam.mask_decoder.parameters())
        sam_memory_mb = sum(p.numel() * p.element_size() for p in sam.parameters()) / (1024**2)
        img_dim = sam.image_encoder.img_size
        if use_sam: sam_role = 'forward pass (produces mask)'
        elif config['loss'] == 'sam': sam_role = 'forward pass (distillation target only)'
        else: sam_role = 'forward pass (distillation target + loss)'  
        print(f"[MAIN] {'SAM role':<32}: {sam_role}")
        print(f"[MAIN] {'SAM input resolution':<32}: {img_dim}x{img_dim}")
        print(f"[MAIN] {'SAM Parameters':<32}: {sam_params:,} ({sam_params/1e6:.2f}M)")
        print(f"[MAIN] {'SAM Memory':<32}: {sam_memory_mb:.2f} MB")
        print(f"[MAIN] {'SAM Device':<32}: {args['device']}")
        print(f"[MAIN] {'SAM Mode':<32}: Frozen (eval)")
        print(f"[MAIN] {'Image Encoder Parameters':<32}: {sam_image_encoder_params:,} ({sam_image_encoder_params/1e6:.2f}M)")
        print(f"[MAIN] {'Prompt Encoder Parameters':<32}: {sam_prompt_encoder_params:,} ({sam_prompt_encoder_params/1e6:.2f}M)")
        print(f"[MAIN] {'Mask Decoder Parameters':<32}: {mask_decoder_params:,} ({mask_decoder_params/1e6:.2f}M)")
        if disable_bn: disable_batchnorm_running_stats(sam)
        print()    
    else:
        sam = None
        img_dim = int(args['input_size'])
        print(f"[MAIN] {'SAM status':<32}: Disabled (not needed for this scenario)")
        print(f"[MAIN] {'Input resolution':<32}: {img_dim}x{img_dim}\n")

    if not args['use_aggregator']:
        print('*'*80)
        print(f"[MAIN] Building prompt learner (backbone + decoder) [ModelEmb]:")
        model = get_model(args)
        print('*'*80)
    else:
        print('*'*80)
        print(f"[MAIN] Building prompt learner (backbone + decoder) + aggregator [HardNetSegmentor]:")
        model = get_standard_model(args)
        print('*'*80)
    model = model.to(device)
    model_params = sum(p.numel() for p in model.parameters())
    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    print(f"\n[MAIN] Model Parameters      : {model_params:,}")
    print(f"[MAIN] Trainable Parameters  : {model_trainable_params:,}")
    print(f"[MAIN] Model Memory          : {model_memory_mb:.2f} MB")
    print(f"[MAIN] Model on device       : {device}\n")
    if disable_bn: disable_batchnorm_running_stats(model)

    if config['trainable'] == 'aggregator':
        for p in model.parameters(): p.requires_grad = False
        for p in model.segmentor.parameters(): p.requires_grad = True
        params_to_optimize = list(model.segmentor.parameters())
    else:
        for p in model.parameters(): p.requires_grad = True
        params_to_optimize = list(model.parameters())    

    args['sam_loaded']     = sam_needed
    args['loss_type']      = config['loss']
    args['distill_alpha']  = distill_alpha
    args['trainable_scope'] = config['trainable']
    args['current_stage']  = stage
    args['phase_2_label']  = scenario_label
    args['phase_2_mode']   = args['phase_2_mode']

    
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen    = n_total - n_trainable
    train_ratio = 100.0 * n_trainable / max(n_total, 1)
    bytes_per_param = next(model.parameters()).element_size()
    param_mem_mb = n_total * bytes_per_param / (1024**2)
    grad_mem_mb  = n_trainable * bytes_per_param / (1024**2)
    n_bb_tr  = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad) if hasattr(model, 'backbone') else 0
    n_bb     = sum(p.numel() for p in model.backbone.parameters()) if hasattr(model, 'backbone') else 0
    n_dec_tr = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad) if hasattr(model, 'decoder') else 0
    n_dec    = sum(p.numel() for p in model.decoder.parameters()) if hasattr(model, 'decoder') else 0
    n_seg_tr = sum(p.numel() for p in model.segmentor.parameters() if p.requires_grad) if getattr(model, 'segmentor', None) is not None else 0
    n_seg    = sum(p.numel() for p in model.segmentor.parameters()) if getattr(model, 'segmentor', None) is not None else 0
    has_aggregator = getattr(model, 'segmentor', None) is not None
    aggregator_status = 'N/A (no aggregator)' if not has_aggregator else 'trainable'
    print(f"[MAIN] {scenario_label}")
    print(f"[MAIN] ─────────────────────────────────────────────────────")
    print(f"[MAIN] Prompt Learner   : {pl_status}")
    print(f"[MAIN] Aggregator       : {aggregator_status}")
    print(f"[MAIN] SAM loaded       : {'Yes' if sam_needed else 'No'}")
    print(f"[MAIN] Loss             : {loss_desc}")
    print(f"[MAIN] Input resolution : {img_dim}x{img_dim}")
    if distill_alpha is not None: print(f"[MAIN] distill alpha    : {distill_alpha}")
    print(f"[MAIN] Trainable params : {n_trainable:,} / {n_total:,} ({train_ratio:.2f}%)")
    print()

    print("[MAIN] Parameter Optimization Summary:")
    print("-" * 58)
    print(f"{'Property':<26} | {'Value'}")
    print("-" * 58)
    print(f"{'Phase':<26} | {scenario_label}")
    print(f"{'Model type':<26} | {type(model).__name__}")
    print(f"{'Input resolution':<26} | {img_dim}x{img_dim}")
    print(f"{'Total params':<26} | {n_total:,} ({n_total/1e6:.2f}M)")
    print(f"{'Trainable params':<26} | {n_trainable:,} ({n_trainable/1e6:.2f}M)")
    print(f"{'Frozen params':<26} | {n_frozen:,} ({n_frozen/1e6:.2f}M)")
    print(f"{'Trainable ratio':<26} | {train_ratio:.2f}%")
    print("-" * 58)
    if n_bb:  print(f"{'Backbone (HarDNet)':<26} | {n_bb_tr:,} / {n_bb:,} ({100.0*n_bb_tr/max(n_bb,1):.1f}%)")
    if n_dec: print(f"{'Decoder (SmallDecoder)':<26} | {n_dec_tr:,} / {n_dec:,} ({100.0*n_dec_tr/max(n_dec,1):.1f}%)")
    if n_seg: print(f"{'Aggregator':<26} | {n_seg_tr:,} / {n_seg:,} ({100.0*n_seg_tr/max(n_seg,1):.1f}%)")
    print("-" * 58)
    print(f"{'Param memory (fp32)':<26} | {param_mem_mb:.2f} MB")
    print(f"{'Grad memory (fp32)':<26} | {grad_mem_mb:.2f} MB")
    print(f"{'Total (param+grad)':<26} | {param_mem_mb + grad_mem_mb:.2f} MB")
    if sam is not None:
        print("-" * 58)
        print(f"{'SAM (frozen)':<26} | {sam_params:,} ({sam_params/1e6:.2f}M) — {sam_role}")
    print("-" * 58)
    print()

    optimizer_type = args['optim']
    lr, wd = float(args['learning_rate']), float(args['WD'])
    momentum = float(args['momentum'])
    if optimizer_type =='Adam': optimizer = optim.Adam(params_to_optimize, lr=lr, weight_decay=wd)
    elif optimizer_type == 'SGD': optimizer = optim.SGD(params_to_optimize, lr=lr, weight_decay=wd, momentum=momentum)
    elif optimizer_type == 'AdamW': optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=wd)
    else: raise ValueError(f"Unsupported optimizer '{optimizer_type}'. Choose from: Adam, SGD, AdamW.")

    n_opt = sum(p.numel() for p in params_to_optimize)
    if n_opt != n_trainable: print(f"[WARNING] Optimizer has {n_opt:,} params but model has {n_trainable:,} trainable. Freeze logic may be inconsistent!")
    n_groups = len(optimizer.param_groups)

    print("[MAIN] Optimizer Summary:")
    print("-" * 58)
    print(f"{'Property':<22} | {'Value'}")
    print("-" * 58)
    print(f"{'Optimizer':<22} | {type(optimizer).__name__}")
    print(f"{'Learning rate':<22} | {lr}")
    print(f"{'Weight decay':<22} | {wd}")
    if optimizer_type == 'SGD':
        print(f"{'Momentum':<22} | {momentum}")
        if momentum > 0: print(f"{'Nesterov':<22} | {optimizer.param_groups[0].get('nesterov', False)}")
    elif optimizer_type in ('Adam', 'AdamW'):
        betas = optimizer.param_groups[0]['betas']
        print(f"{'Betas':<22} | {betas}")
        print(f"{'Eps':<22} | {optimizer.param_groups[0]['eps']}")    
    print(f"{'Param groups':<22} | {n_groups}")
    print(f"{'Total opt. params':<22} | {n_opt:,} ({n_opt/1e6:.2f}M)\n") 

    train_dataloader, val_dataloader = get_train_val_dataloaders(args, img_dim)

    num_epochs = args['epochs']
    if args['use_scheduler']:
        steps_per_epoch = math.ceil(len(train_dataloader) / args['acc_steps'])
        scheduler_type = args['scheduler']
        gamma = float(args['scheduler_gamma'])
        total_steps = steps_per_epoch * num_epochs
        print(f"[SCHEDULER] steps_per_epoch = {steps_per_epoch}")
        if scheduler_type == 'StepLR':
            step_size = steps_per_epoch * args['scheduler_step']
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            print(f"[SCHEDULER] StepLR | step_size={step_size} (={args['scheduler_step']} epochs) | gamma={gamma}")
        elif scheduler_type == 'MultiStepLR':
            milestones = [int(total_steps * 0.3), int(total_steps * 0.6), int(total_steps * 0.9)]
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
            print(f"[SCHEDULER] MultiStepLR | milestones={milestones} (steps) ≈ epochs {[round(m/steps_per_epoch,1) for m in milestones]}")
        elif scheduler_type == 'ExponentialLR':
            print(f"[WARNING] ExponentialLR decay per batch (gamma={gamma}) — likely too aggressive")
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=args['scheduler_patience'])
            print(f"[SCHEDULER] ReduceLROnPlateau | factor={gamma} | patience={args['scheduler_patience']}")
        elif scheduler_type == 'CosineAnnealingLR':
            T_max = int(args['schedCos_Tmax'])
            if T_max < steps_per_epoch:
                T_max_steps = steps_per_epoch * T_max
                print(f"[SCHEDULER] CosineAnnealingLR: interpreting T_max={T_max} as epochs → {T_max_steps} steps")
            else: T_max_steps = T_max    
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max_steps, eta_min=args['schedCos_eta_min'], last_epoch=args['schedCos_last_epoch']) 
            print(f"[SCHEDULER] CosineAnnealingLR | T_max={T_max_steps} | eta_min={args['schedCos_eta_min']}")           
        elif scheduler_type == 'WarmupCosineSchedule':
            warmup_epochs = args['warmup_epochs']
            warmup_steps = steps_per_epoch * warmup_epochs
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps, cycles=args['warmup_cycles'], last_epoch=args['schedWarmupCos_last_epoch'])
            print(f"[SCHEDULER] WarmupCosineSchedule | warmup={warmup_epochs} epochs ({warmup_steps} steps) | t_total={total_steps} steps | cycles={args['warmup_cycles']}")
        elif scheduler_type == 'WarmupLinearSchedule':
            warmup_steps = steps_per_epoch * args['warmup_epochs']
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)
            print(f"[SCHEDULER] WarmupLinearSchedule | warmup={warmup_steps} steps | t_total={total_steps} steps")
        else: raise ValueError(f"Unsupported scheduler '{scheduler_type}'")
    else:
        scheduler = None
        print("[SCHED] No scheduler (constant LR)")
    print()

    if args['wandb_watch']:
        if getattr(saver, 'wandb_enabled', False):
            try:
                wandb_watch_freq = args['wandb_watch_freq']
                wandb.watch(model, log='all', log_freq=wandb_watch_freq, log_graph=True)
                print(f"[MAIN] WandB watch enabled | log=all | freq={wandb_watch_freq}\n")
            except Exception as e: print(f"[WARNING] wandb.watch failed: {type(e).__name__}: {e}")        
        else: print("[MAIN] --wandb_watch requested but W&B disabled — skipping.\n")  

    best = -1.0
    best_metric_name = args['best_metric']  
    saver.create_best_csv(best_metric_name)                    
    print(f"[MAIN] Tracking best model by '{best_metric_name}' in: {saver.best_csv_file}")

    device_is_cuda = getattr(args['device'], 'type', 'cpu') == 'cuda'
    use_amp = bool(use_cuda_amp) and cuda_available and device_is_cuda
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"[MAIN] AMP enabled: {use_amp} (requested={requested_cuda_amp}, cuda_available={cuda_available}, device={device})\n")

    train_context = {
    'dataloader': train_dataloader,
    'model': model,
    'sam': sam,
    'optimizer': optimizer,
    'device': device,
    'args': args,
    'saver': saver,
    'scheduler': scheduler,
    'scaler': scaler,
    'use_amp': use_amp,
}

    for epoch in range(num_epochs):
        train_metrics = train_single_epoch(train_context, epoch)
        train_loss        = train_metrics['loss']
        train_dice_excl   = train_metrics['dice_exclude_background']
        train_dice_incl   = train_metrics['dice_include_background']
        train_iou_excl    = train_metrics['iou_exclude_background']
        train_iou_incl    = train_metrics['iou_include_background']

        saver.log_scalar('train_loss',        train_loss,      epoch)
        saver.log_scalar('train_dice_excl',   train_dice_excl, epoch)
        saver.log_scalar('train_dice_incl',   train_dice_incl, epoch)
        saver.log_scalar('train_iou_excl',    train_iou_excl,  epoch)
        saver.log_scalar('train_iou_incl',    train_iou_incl,  epoch)
        
        print(f"[MAIN] [Epoch {epoch}] "
          f"train_loss={train_loss:.4f} | "
          f"train_dice_excl={train_dice_excl:.4f} | "
          f"train_dice_incl={train_dice_incl:.4f} | "
          f"train_iou_excl={train_iou_excl:.4f} | "
          f"train_iou_incl={train_iou_incl:.4f}\n")         
          

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
============================================================
            POLYP SEGMENTATION FRAMEWORK
============================================================
Method : SAM-Guided Prompt Learning for Polyp Segmentation
Task   : Automated Polyp Segmentation
Input  : Colonoscopy RGB Images (2D)
Output : Binary Polyp Segmentation Masks
============================================================

This framework leverages Segment Anything Model (SAM) during 
training to guide a dense prompt learner. At inference, SAM 
is replaced by a lightweight aggregator for efficient deployment.

For usage examples, see README.md
============================================================\n""", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--strict_determinism', type=str2bool, default=False, help='Enable strict determinism (slower but reproducible). Use True for final thesis experiments.')
    parser.add_argument('--experiment_name', required=True, help='Experiment name (required)')
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'], help='WandB mode')
    parser.add_argument('--wandb_project', default='SGPL_Polyp_Seg', help='WandB project name')
    parser.add_argument('--wandb_entity', default=None, help='WandB entity (username or team)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='Wandb tags')
    parser.add_argument('--use_sam', type=str2bool, default=True, help='Use SAM (phase 1) vs aggregator only (phase 2)')
    parser.add_argument('--sam_ckpt', type=str, default='sam_vit_b', choices = ['sam_vit_b', 'medsam_vit_b', 'sam_vit_h'], help='SAM checkpoint')
    parser.add_argument('--sam_version', type=str, default='vit_b', choices = ['vit_b', 'vit_h'], help='SAM version')
    parser.add_argument('--device', default='cuda', help="Device: 'cuda' or 'cpu'")
    parser.add_argument('--use_cuda_amp', type=str2bool, default=True, help='Use mixed precision training')
    parser.add_argument('--device_id', type=int, default=0, help='GPU index')
    parser.add_argument('--disable_batchnorm_running_stats', type=str2bool, default=False, help='Disable batchnorm running stats')
    parser.add_argument('-input_size', '--input_size', type=int, default=512, help='Input image size (resize resolution)')
    parser.add_argument('--use_aggregator', type=str2bool, default=False, help='Use HardNetSegmentor (backbone + decoder + aggregator) vs ModelEmb (backbone + decoder)')
    parser.add_argument('--backbone_type', type=str, default='HardNet', help='Prompt learner backbone')
    parser.add_argument('-depth_wise', '--depth_wise', type = str2bool, default=False, help='Use depthwise HarDNet')
    parser.add_argument('-order', '--order', default=68, type=int, help='HarDNet variant: 39 | 68 | 85')
    parser.add_argument('--pretrained', type=str2bool, default=True, help='Use ImageNet pretrained weights')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint to resume/init from')
    parser.add_argument('--net', type=str, default='HardNetSegmentor', help='Segmentor net (phase 2)')
    parser.add_argument('--segmentor_net_type', type=str, default='PointwiseConv', choices=['PointwiseConv', 'ConvUpsample', 'TransposeConvUpsample'], help='Aggregator head')
    parser.add_argument('--out_channels', type=int, default=2, help='Output channels')
    parser.add_argument('--upsample_factor_list', type=int, nargs='+', default=[2, 2], help='Upsample factors for ConvUpsample aggregator')
    parser.add_argument('--segmentor_finetune_backbone', type=str2bool, default=False, help='If we want to train the miniSegmentor finetuning also the backbone')
    parser.add_argument('--single_pl_trainable', type=str2bool, default=False, help='[single mode] Whether prompt learner (backbone+decoder) is trainable')
    parser.add_argument('--single_loss_type', type=str, choices=['gt', 'mixed'], default='gt', help='[single mode] Loss: gt (vs ground truth) or mixed (vs sam+gt)')
    parser.add_argument('--single_distill_weight', type=float, default=0.5, help='[single mode, mixed loss] alpha weight: loss = alpha*l(p,sam) + (1-alpha)*l(p,gt)')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1, help='[two_stage mode] Stage 1 (distillation) or Stage 2 (finetune)')
    parser.add_argument('--stage1_loss_type', type=str, choices=['sam', 'mixed'], default='sam', help='[two_stage, stage 1] Loss: sam (vs sam output) or mixed (vs sam+gt)')
    parser.add_argument('--stage1_distill_weight', type=float, default=0.5, help='[two_stage, stage 1, mixed loss] alpha weight')
    parser.add_argument('--phase_2_mode', type=str, choices=['single', 'two_stage'], default='two_stage', help='Phase 2 mode: single-stage or two-stage')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-WD', '--WD', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD', 'AdamW'], help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum (only used with SGD)')
    parser.add_argument('--image_key', type=str, default='image',help='Key for image in the data dict')
    parser.add_argument('--mask_key', type=str, default='mask', help='Key for mask in the data dict')
    parser.add_argument('--split_path', type=str, default=os.path.join(PROJECT_DIR, 'data', 'split.json'), help='Path to the JSON split file')
    parser.add_argument('--n_fold', type=int, default=0, help='Fold index')
    parser.add_argument('-bs', '--Batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('-nW', '--nW', type=int, default=0, help='Number of workers for train loader')
    parser.add_argument('-nW_eval', '--nW_eval', type=int, default=0, help='Number of workers for val loader')
    parser.add_argument('--use_scheduler', type=str2bool, default=False, help='Enable LR scheduler')
    parser.add_argument('--scheduler', type=str, default='StepLR', choices=['StepLR', 'MultiStepLR', 'ExponentialLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'WarmupCosineSchedule', 'WarmupLinearSchedule'], help='Scheduler type')
    parser.add_argument('--scheduler_step', type=int, default=100, help='Scheduler step for StepLR, and first step of MultiStepLR')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Gamma (LR decay factor)')
    parser.add_argument('-epochs', '--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--scheduler_patience', type=int, default=10, help='Patience for ReduceLROnPlateau')
    parser.add_argument('--schedCos_Tmax', type=int, default=10, help='T_max for CosineAnnealingLR')
    parser.add_argument('--schedCos_eta_min', type=float, default=0.0001, help='eta_min for CosineAnnealingLR')
    parser.add_argument('--schedCos_last_epoch', type=int, default=-1, help='last_epoch for CosineAnnealingLR')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--warmup_cycles', type=float, default=0.5, help='Warmup cycles')
    parser.add_argument('--schedWarmupCos_last_epoch', type=int, default=-1, help='WarmupCosineSchedule last_epoch (use only if you are resuming a training)')
    parser.add_argument('--wandb_watch', type=str2bool, default=False, help='Enable WandB watch')
    parser.add_argument('--wandb_watch_freq', type=int, default=10, help='Wandb watch frequency')
    parser.add_argument('--acc_steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--best_metric', type=str, default='dice_exclude_background', choices=['dice_exclude_background', 'dice_include_background', 'iou'], help='Best metric')
    parser.add_argument('--threshold_discretize', type=float, default=0.5, help='Threshold for binarization')
    parser.add_argument('--include_background', type=str2bool, default=False, help='Include background class in loss')
    parser.add_argument('--criterion', type=str, default='dice_ce', choices=['dice', 'dice_ce'], help='Loss function')
    parser.add_argument('--norm_mode', type=str, default='minmax', choices=['minmax', 'sigmoid'], help='Normalization mode for SAM output in Phase 1')
    parser.add_argument('--save_train_images', type=str2bool, default=True, help='Save training images to WandB')

    print(parser.description)
    args = vars(parser.parse_args())

    print('='*120)
    seed = args['seed']
    strict = args['strict_determinism']
    set_seed(seed=seed, strict=strict)

    output_root = os.path.join(PROJECT_DIR, 'results')
    saver = Saver(output_folder=output_root, args=args)
    cmd = f"{sys.executable} " + " ".join(shlex.quote(arg) for arg in sys.argv)
    saver.log_cmd(cmd)

    args['exp_folder'] = saver.experiment_dir
    args['last_ckpt_path'] = os.path.join(saver.ckpt_dir, 'net_last.pth')
    args['best_ckpt_path'] = os.path.join(saver.ckpt_dir, 'net_best.pth')
    args['vis_folder'] = saver.vis_dir
    args['wandb_info'] = saver.wandb_info

    if args['use_sam']: sam_needed = True
    elif args['phase_2_mode'] == 'single': sam_needed = (args['single_loss_type'] == 'mixed')
    else: sam_needed = (args['stage'] == 1)
    args['sam_needed'] = sam_needed

    if sam_needed:
        sam_ckpt = os.path.join(PROJECT_DIR, 'cp', f"{args['sam_ckpt']}.pth")
        if not os.path.exists(sam_ckpt): raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt}\n Please download {args['sam_ckpt']}.pth into the cp/ folder.")
    
        sam_args = {'sam_checkpoint': sam_ckpt, 'model_type': args['sam_version']}
        print(f"[MAIN] SAM checkpoint path: {sam_args['sam_checkpoint']}")
        print(f"[MAIN] SAM Model type: {sam_args['model_type']}\n")
    else:
        sam_args = None
        print(f"[MAIN] SAM is disabled.\n")

    args['sam_args'] = sam_args
    saver.log_hparams(args)    
    
    main(args=args, sam_args=sam_args, saver=saver)
    print('='*120)
