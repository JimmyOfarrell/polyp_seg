
import torch.optim as optim
import wandb
from utils.model_utils import get_model, get_standard_model
from utils.scheduler import WarmupCosineSchedule, WarmupLinearSchedule
from train import train_single_epoch
from inference import inference_dataloader
from utils.saver import Saver

import torch
import argparse
import os
import sys
import shlex
from segment_anything import sam_model_registry
from utils.utils import disable_batchnorm_running_stats, str2bool, set_seed
from utils.dataset_utils import get_train_val_dataloaders

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def main(args=None, sam_args=None, saver=None):
    device = args['device']
    cuda_available = torch.cuda.is_available()
    use_cuda_amp = args['use_cuda_amp']
    use_sam = args['use_sam']
    disable_bn = args['disable_batchnorm_running_stats']

    print("[MAIN] Device configuration:")
    if str(device).lower() == 'cuda':
        if not cuda_available:
            print(f"[WARNING] CUDA requested but not available. Falling back to CPU.")
            device = args['device'] = torch.device("cpu")
            print(f"[MAIN] Device         : CPU")
            print(f"[MAIN] PyTorch version: {torch.__version__}")
            if use_sam:
                print(f"[WARNING] Running Phase 1 with SAM on CPU will be extremely slow! Consider using a GPU for training.")
            if use_cuda_amp:
                print(f"[WARNING] AMP is enabled but device is CPU. Disabling AMP.")
                use_cuda_amp = args['use_cuda_amp'] = False
        else:
            num_gpus = torch.cuda.device_count()
            device_id = int(args['device_id'])
            if device_id >= num_gpus or device_id < 0:
                print(f"[WARNING] device_id={device_id} is out of range (only {num_gpus} GPU(s) available). Using device 0.")
                device_id = 0
                args['device_id'] = 0    
            device = args['device'] = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(device_id) 

            gpu_name = torch.cuda.get_device_name(device_id)
            gpu_props = torch.cuda.get_device_properties(device_id)
            total_vram_gb = gpu_props.total_memory / (1024**3)

            print(f"[MAIN] Device            : {args['device']}")
            print(f"[MAIN] GPU name          : {gpu_name}")
            print(f"[MAIN] Total VRAM        : {total_vram_gb:.2f} GB")
            print(f"[MAIN] CUDA version      : {torch.version.cuda}")
            print(f"[MAIN] PyTorch version   : {torch.__version__}")

            cc = (gpu_props.major, gpu_props.minor)
            print(f"[MAIN] Compute capability: {cc[0]}.{cc[1]}")

            if cc < (6, 0): print(f"[WARNING] GPU compute capability {cc[0]}.{cc[1]} is below 6.0. Mixed precision (AMP) may not work properly.")
            if cc < (7, 0) and use_cuda_amp: print(f"[WARNING] GPU compute capability {cc[0]}.{cc[1]} is below 7.0. Tensor Cores not available; AMP will be limited.")

            try:
                free_vram, total_vram = torch.cuda.mem_get_info(device_id)
                free_vram_gb = free_vram / (1024**3)
                print(f"[MAIN] Free VRAM     : {free_vram_gb:.2f} GB")
            
                if use_sam and free_vram_gb < 8:
                    print(f"[WARNING] Free VRAM ({free_vram_gb:.2f} GB) may be insufficient for Phase 1 with SAM (ViT-B). Consider:")
                    print(f"         - Smaller batch size (-bs)")
                    print(f"         - Using --use_cuda_amp True")
                    print(f"         - Already using smallest SAM (vit_b); reduce batch size instead")
            except Exception: print(f"[WARNING] Could not query VRAM: {type(e).__name__}: {e}") 
    else:
        device = args['device'] = torch.device("cpu")
        print(f"[MAIN] Device         : CPU")
        print(f"[MAIN] PyTorch version: {torch.__version__}")
        if use_sam:
            print(f"[WARNING] Running Phase 1 with SAM on CPU will be extremely slow! Consider using a GPU for training.")
        if use_cuda_amp:
            print(f"[WARNING] AMP is enabled but device is CPU. Disabling AMP.")
            use_cuda_amp = args['use_cuda_amp'] = False
    print()

    if use_sam:
        print("[MAIN] SAM model (Phase 1: SAM-guided prompt learning) Configuration:")
        try: sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
        except Exception as e: raise RuntimeError(f"Failed to load SAM '{sam_args['model_type']}' from '{sam_args['sam_checkpoint']}': {e}")
        sam.to(device=device)
        sam.eval()
        sam_params = sum(p.numel() for p in sam.parameters())
        sam_memory_mb = sum(p.numel() * p.element_size() for p in sam.parameters()) / (1024**2)
        img_dim = sam.image_encoder.img_size
        print(f"[MAIN] SAM input resolution: {img_dim}x{img_dim}")
        print(f"[MAIN] SAM Parameters      : {sam_params:,}")
        print(f"[MAIN] SAM Memory          : {sam_memory_mb:.2f} MB")
        print(f"[MAIN] SAM Device          : {args['device']}")
        print(f"[MAIN] SAM Mode            : Frozen (eval)")

        if disable_bn:
            disable_batchnorm_running_stats(sam)
            print("[MAIN] BatchNorm running stats disabled for SAM.")
    else:
        sam = None
        img_dim = int(args['Idim'])
        print("[MAIN] SAM is disabled. Using standard segmentor directly.")
        print(f"[MAIN] Using Idim = {img_dim}×{img_dim} for prompt learner input")
    print()

    if not args['use_standard_net']:
        print(f"[MAIN] Building ModelEmb (Phase 1: SAM-guided prompt learning):", '\n')
        model = get_model(args)
        input('stop')
    else:
        print('\n'+'='*120)
        print(f"[MAIN] Building HardNetSegmentor (Phase 2: aggregator-only)")
        model = get_standard_model(args)
        print('='*120)

    print('\n'+'='*60)
    model = model.to(args['device'])
    print(f"[MAIN] Model on device: {args['device']}")
    print('='*60)

    if args['disable_batchnorm_running_stats']: disable_bn(model)

    if not args['use_sam']:
        if not args['segmentor_finetune_backbone']:
            for param in model.parameters(): param.requires_grad = False
            for param in model.segmentor.parameters(): param.requires_grad = True
            params_to_optimize = list(model.segmentor.parameters())
        else:
            for param in model.parameters(): param.requires_grad = True
            params_to_optimize = list(model.parameters())
    else: params_to_optimize = list(model.parameters())

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen    = n_total - n_trainable
    train_ratio = 100.0 * n_trainable / max(n_total, 1)
    bytes_per_param = next(model.parameters()).element_size()
    param_mem_mb = n_total * bytes_per_param / (1024**2)
    grad_mem_mb  = n_trainable * bytes_per_param / (1024**2)
    n_tensors_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    n_tensors_total     = sum(1 for p in model.parameters())
    devices = {str(p.device) for p in model.parameters()}
    dtypes  = {str(p.dtype).replace('torch.','') for p in model.parameters()}
    n_bb_tr  = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad) if hasattr(model, 'backbone') else 0
    n_bb     = sum(p.numel() for p in model.backbone.parameters()) if hasattr(model, 'backbone') else 0
    n_dec_tr = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad) if hasattr(model, 'decoder') else 0
    n_dec    = sum(p.numel() for p in model.decoder.parameters()) if hasattr(model, 'decoder') else 0
    n_seg_tr = sum(p.numel() for p in model.segmentor.parameters() if p.requires_grad) if getattr(model, 'segmentor', None) is not None else 0
    n_seg    = sum(p.numel() for p in model.segmentor.parameters()) if getattr(model, 'segmentor', None) is not None else 0

    phase = 'Phase 1 (SAM-guided)' if args['use_sam'] else ('Phase 2 (full fine-tune)' if args['segmentor_finetune_backbone'] else 'Phase 2 (aggregator-only)')

    print('\n' + '='*60)
    print("[MAIN] Parameter Optimization Summary")
    print("-" * 58)
    print(f"{'Property':<22} | {'Value'}")
    print("-" * 58)
    print(f"{'Phase':<22} | {phase}")
    print(f"{'Model type':<22} | {type(model).__name__}")
    print(f"{'Total params':<22} | {n_total:,} ({n_total/1e6:.2f}M)")
    print(f"{'Trainable params':<22} | {n_trainable:,} ({n_trainable/1e6:.2f}M)")
    print(f"{'Frozen params':<22} | {n_frozen:,} ({n_frozen/1e6:.2f}M)")
    print(f"{'Trainable ratio':<22} | {train_ratio:.2f}%")
    print(f"{'Trainable tensors':<22} | {n_tensors_trainable} / {n_tensors_total}")
    print("-" * 58)
    if n_bb:  print(f"{'Backbone (HarDNet)':<28} | {n_bb_tr:,} / {n_bb:,} ({100.0*n_bb_tr/max(n_bb,1):.1f}%)")
    if n_dec: print(f"{'Decoder (SmallDecoder)':<28} | {n_dec_tr:,} / {n_dec:,} ({100.0*n_dec_tr/max(n_dec,1):.1f}%)")
    if n_seg: print(f"{'Aggregator':<28} | {n_seg_tr:,} / {n_seg:,} ({100.0*n_seg_tr/max(n_seg,1):.1f}%)")
    print("-" * 58)
    print(f"{'Param memory (fp32)':<22} | {param_mem_mb:.2f} MB")
    print(f"{'Grad memory (fp32)':<22} | {grad_mem_mb:.2f} MB")
    print(f"{'Total (param+grad)':<22} | {param_mem_mb + grad_mem_mb:.2f} MB")
    print("-" * 58)
    print(f"{'Param devices':<22} | {devices}")
    print(f"{'Param dtypes':<22} | {dtypes}")
    print('='*60 + '\n')

    lr, wd = float(args['learning_rate']), float(args['WD'])
    momentum = float(args.get('momentum', 0.9))
    if args['optim'] =='Adam':optimizer = optim.Adam(params_to_optimize, lr=lr, weight_decay=wd)
    elif args['optim'] == 'SGD':optimizer = optim.SGD(params_to_optimize, lr=lr, weight_decay=wd, momentum=momentum)
    elif args['optim'] == 'AdamW':optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=wd)
    else: raise ValueError(f"Unsupported optimizer '{args['optim']}'. Choose from: Adam, SGD, AdamW.")

    n_opt = sum(p.numel() for p in params_to_optimize)
    n_groups = len(optimizer.param_groups)
    print('\n' + '='*60)
    print("[MAIN] Optimizer Summary")
    print("-" * 58)
    print(f"{'Property':<22} | {'Value'}")
    print("-" * 58)
    print(f"{'Optimizer':<22} | {type(optimizer).__name__}")
    print(f"{'Learning rate':<22} | {lr}")
    print(f"{'Weight decay':<22} | {wd}")
    if args['optim'] == 'SGD':
        print(f"{'Momentum':<22} | {momentum}")
        print(f"{'Nesterov':<22} | {optimizer.param_groups[0].get('nesterov', False)}")
    elif args['optim'] in ('Adam', 'AdamW'):
            betas = optimizer.param_groups[0]['betas']
            print(f"{'Betas':<22} | {betas}")
            print(f"{'Eps':<22} | {optimizer.param_groups[0]['eps']}")    
    print(f"{'Param groups':<22} | {n_groups}")
    print(f"{'Total opt. params':<22} | {n_opt:,} ({n_opt/1e6:.2f}M)")
    print('='*60 + '\n')

    print('\n' + '='*60)
    train_dataloader, val_dataloader = get_train_val_dataloaders(args, img_dim)
    if args['use_scheduler']:
        steps_per_epoch = len(train_dataloader)
        scheduler_type = args['scheduler']
        gamma = float(args['scheduler_gamma'])
        num_epochs = int(args['epochs'])
        total_steps = steps_per_epoch * num_epochs
        print(f"[SCHED] steps_per_epoch = {steps_per_epoch}")
        if scheduler_type == 'StepLR':
            step_size = steps_per_epoch * args['scheduler_step']
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            print(f"[SCHED] StepLR | step_size={step_size} (={args['scheduler_step']} epochs) | gamma={gamma}")
        elif scheduler_type == 'MultiStepLR':
            milestones = [int(total_steps * 0.3), int(total_steps * 0.6), int(total_steps * 0.9)]
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
            print(f"[SCHED] MultiStepLR | milestones={milestones} (steps) ≈ epochs {[round(m/steps_per_epoch,1) for m in milestones]}")
        elif scheduler_type == 'ExponentialLR':
            print(f"[WARN] ExponentialLR decay per batch (gamma={args['scheduler_gamma']}) — likely too aggressive")
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=args['scheduler_patience'])
            print(f"[SCHED] ReduceLROnPlateau | factor={gamma} | patience={args['scheduler_patience']}")
        elif scheduler_type == 'CosineAnnealingLR':
            T_max = int(args['schedCos_Tmax'])
            if T_max < steps_per_epoch:
                T_max_steps = steps_per_epoch * T_max
                print(f"[SCHED] CosineAnnealingLR: interpreting T_max={T_max} as epochs → {T_max_steps} steps")
            else: T_max_steps = T_max    
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max_steps, eta_min=args['schedCos_eta_min'], last_epoch=args['schedCos_last_epoch']) 
            print(f"[SCHED] CosineAnnealingLR | T_max={T_max_steps} | eta_min={args['schedCos_eta_min']}")           
        elif scheduler_type == 'WarmupCosineSchedule':
            warmup_epochs = args['warmup_epochs']
            warmup_steps = steps_per_epoch * warmup_epochs
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps, cycles=args['warmup_cycles'], last_epoch=args['schedWarmupCos_last_epoch'])
            print(f"[SCHED] WarmupCosineSchedule | warmup={warmup_epochs} epochs ({warmup_steps} steps) | t_total={total_steps} steps | cycles={args['warmup_cycles']}")
        elif scheduler_type == 'WarmupLinearSchedule':
            warmup_steps = steps_per_epoch * args['warmup_epochs']
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)
            print(f"[SCHED] WarmupLinearSchedule | warmup={warmup_steps} steps | t_total={total_steps} steps")
        else: raise ValueError(f"Unsupported scheduler '{scheduler_type}'")
    else:
        scheduler = None
        print("[SCHED] No scheduler (constant LR)")
    print('='*60 + '\n')

    if args.get('wandb_watch', False):
        if getattr(saver, 'wandb_enabled', False):
            try:
                wandb.watch(model, log='all', log_freq=int(args.get('wandb_watch_freq', 10)), log_graph=True)
                print(f"[MAIN] WandB watch enabled | log=all | freq={args.get('wandb_watch_freq', 10)}")
            except Exception as e:
                print(f"[WARN] wandb.watch failed: {type(e).__name__}: {e}")        
        else: print("[MAIN] --wandb_watch requested but W&B disabled — skipping.")

    best = -1.0
    path_best = os.path.join(saver.path, 'best.csv')
    metric_name = args.get('best_metric', 'dice_f')
    f_best = open(path_best, 'w', encoding='utf-8', newline='')
    f_best.write(f'epoch,{metric_name}\n')                        
    f_best.flush()
    print(f"[MAIN] Tracking best model by '{metric_name}' in: {path_best}")

    cuda_available = torch.cuda.is_available()
    device_is_cuda = getattr(args['device'], 'type', 'cpu') == 'cuda'
    use_amp = args.get('use_cuda_amp', False) is True and cuda_available and device_is_cuda
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"[MAIN] AMP enabled: {use_amp} (requested={args.get('use_cuda_amp', False)}, cuda_available={cuda_available}, device={args['device']})")

    num_epochs = int(args['epochs'])
    for epoch in range(num_epochs):
        avg_loss, avg_dice, avg_iou= train_single_epoch(train_dataloader, model, sam, optimizer, epoch, args, saver, scheduler, scaler)
        saver.log_loss('train_loss', avg_loss, epoch)
        saver.log_loss('train_dice', avg_dice, epoch)
        saver.log_loss('train_iou', avg_iou, epoch)

        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f} | train_dice={avg_dice:.4f} | train_iou={avg_iou:.4f}")

        with torch.no_grad():
            dice_true, dice_false, iou_val, val_loss = inference_dataloader(val_dataloader, model, sam, epoch, args, saver)
            saver.log_loss('val_loss', val_loss, epoch)
            saver.log_loss('IoU', iou_val, epoch)
            saver.log_loss('Dice/include_true', dice_true, epoch)
            saver.log_loss('Dice/include_false', dice_false, epoch)

            best_metric_name = args.get('best_metric', 'dice_f')

            if best_metric_name == 'dice_f': metric = dice_false
            elif best_metric_name == 'dice_t': metric = dice_true
            elif best_metric_name == 'IoU': metric = iou_val
            else: raise ValueError(f"Unsupported best_metric '{best_metric_name}'. Choose from: 'dice_f', 'dice_t', 'IoU'.")
                    
            if metric > best:
                torch.save(model.state_dict(), args['best_ckpt_path'])
                best = metric
                print(f'[MAIN] ✓ New best {best_metric_name} = {best:.4f} @ epoch {epoch}')
                f_best.write(f'{epoch},{best}\n')
                f_best.flush()
            
        if epoch % int(args.get('save_every', 10)) == 0 or epoch == num_epochs-1: saver.save_model(model, 'net_last', epoch, fixed_name=True)
            
    f_best.close()
    print(f"[MAIN] ✓ Training complete. Best metric: {best:.4f}")
    
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
    parser.add_argument('--exp_name', required=True, help='Experiment name (required)')
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'], help='WandB mode')
    parser.add_argument('--wandb_project', default='SGPL_Polyp_Seg', help='WandB project name')
    parser.add_argument('--wandb_entity', default=None, help='WandB entity (username or team)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='Wandb tags')
    parser.add_argument('--best_metric', type=str, default='dice_f', choices=['dice_f', 'dice_t', 'iou'], help='Best metric')
    parser.add_argument('--use_sam', type=str2bool, default=True, help='Use SAM (phase 1) vs aggregator only (phase 2)')
    parser.add_argument('--sam_ckpt', type=str, default='sam_vit_b', choices = ['sam_vit_b', 'medsam_vit_b', 'sam_vit_h'], help='SAM checkpoint')
    parser.add_argument('--sam_version', type=str, default='vit_b', choices = ['vit_b', 'vit_h'], help='SAM version')
    parser.add_argument('--device_id', type=int, default=0, help='GPU index')
    parser.add_argument('--device', default='cuda', help="Device: 'cuda' or 'cpu'")
    parser.add_argument('--use_cuda_amp', type=str2bool, default=True, help='Use mixed precision training')
    parser.add_argument('--disable_batchnorm_running_stats', type=str2bool, default=False, help='Disable batchnorm running stats')
    parser.add_argument('-Idim', '--Idim', type=int, default=512, help='Input image size (resize resolution)')
    parser.add_argument('--image_key', type=str, default='image',help='Key for image in the data dict')
    parser.add_argument('--mask_key', type=str, default='mask', help='Key for mask in the data dict')
    parser.add_argument('--split_path', type=str, default=os.path.join(PROJECT_DIR, 'data', 'split.json'), help='Path to the JSON split file')
    parser.add_argument('--n_fold', type=int, default=0, help='Fold index')
    parser.add_argument('-bs', '--Batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('-nW', '--nW', type=int, default=0, help='Number of workers for train loader')
    parser.add_argument('-nW_eval', '--nW_eval', type=int, default=0, help='Number of workers for val loader')
    parser.add_argument('--use_standard_net', type=str2bool, default=False, help='Use standard segmentor (phase 2) vs prompt learner (phase 1)')
    parser.add_argument('--model_emb', type=str, default='HardNet', help='Prompt learner backbone')
    parser.add_argument('-depth_wise', '--depth_wise', type = str2bool, default=False, help='Use depthwise HarDNet')
    parser.add_argument('-order', '--order', default=68, type=int, help='HarDNet variant: 39 | 68 | 85')
    parser.add_argument('--pretrained', type=str2bool, default=True, help='Use ImageNet pretrained weights')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint to resume/init from')
    parser.add_argument('--net', type=str, default='HardNetSegmentor', help='Segmentor net (phase 2)')
    parser.add_argument('--net_segmentor', type=str, default='PointwiseConv', choices=['PointwiseConv', 'ConvUpsample', 'TransposeConvUpsample', 'UnetLike'], help='Aggregator head')
    parser.add_argument('--out_channels', type=int, default=2, help='Output channels')
    parser.add_argument('--upsample_factor_list', type=int, nargs='+', default=[2, 2], help='Upsample factors for ConvUpsample aggregator')
    parser.add_argument('--segmentor_finetune_backbone', type=str2bool, default=False, help='If we want to train the miniSegmentor finetuning also the backbone')
    parser.add_argument('-lr', '--learning_rate', default=0.01, help='Learning rate')
    parser.add_argument('-WD', '--WD', default=1e-4, help='Weight decay')
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD', 'AdamW'], help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (only used with SGD)')
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
    parser.add_argument('--theashold_discretize', type=float, default=0.5, help='Threshold for binarization')
    parser.add_argument('--include_background', type=str2bool, default=False, help='Include background class in loss')
    parser.add_argument('--criterion', type=str, default='dice_ce', choices=['dice', 'dice_ce'], help='Loss function')
    parser.add_argument('--norm_mode', type=str, default='minmax', choices=['minmax', 'sigmoid'], help='Normalization mode for SAM output in Phase 1')
    parser.add_argument('--save_train_images', type=str2bool, default=True, help='Save training images to WandB')
    parser.add_argument('--acc_steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--save_val_images', type=str2bool, default=True, help='Save validation images to WandB')
    parser.add_argument('--save_every', type=int, default=10, help='Save net_last checkpoint every N epochs (also saves at last epoch)')
    args = vars(parser.parse_args())
    print(parser.description)

    seed = args['seed']
    strict = args['strict_determinism']
    set_seed(seed=seed, strict=strict)

    args['experiment_name']= args['exp_name']
    output_root = os.path.join(PROJECT_DIR, 'results')

    saver = Saver(output_folder=output_root, args=args)
    cmd = f"{sys.executable} " + " ".join(shlex.quote(arg) for arg in sys.argv)
    saver.log_cmd(cmd)

    args['exp_folder'] = saver.experiment_dir
    args['last_ckpt_path'] = os.path.join(saver.ckpt_dir, 'net_last.pth')
    args['best_ckpt_path'] = os.path.join(saver.ckpt_dir, 'net_best.pth')
    args['vis_folder'] = saver.vis_dir
    args['wandb_info'] = saver.wandb_info

    if args['use_sam']:
        sam_ckpt = os.path.join(PROJECT_DIR, 'cp', f"{args['sam_ckpt']}.pth")
        if not os.path.exists(sam_ckpt): raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt}\n Please download {args['sam_ckpt']}.pth into the cp/ folder.")

        sam_args = {'sam_checkpoint': sam_ckpt, 'model_type': args['sam_version']}
        args['sam_args'] = sam_args
        print(f"[MAIN] SAM checkpoint path: {sam_ckpt}")
        print(f"[MAIN] SAM Model type : {args['sam_version']}", '\n')
    else:
        sam_args = None
        print("[MAIN] SAM is disabled.", '\n')

    args['sam_args'] = sam_args
    saver.log_hparams(args)    

    main(args=args, sam_args=sam_args, saver=saver)
