import wandb


from train import train_single_epoch
from inference import inference_dataloader
from utils.saver import Saver

import torch
import torch.optim as optim
import argparse
import os
import sys
import shlex

from utils.utils import disable_batchnorm_running_stats, str2bool, set_seed

from utils.dataset_utils import get_train_val_dataloaders
from utils.scheduler import WarmupCosineSchedule, WarmupLinearSchedule

def main(args=None, sam_args=None, saver=None):
    
    num_epochs = int(args['epochs'])
    

    if not use_sam:
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
    phase = 'Phase 1 (SAM-guided)' if use_sam else ('Phase 2 (full fine-tune)' if args['segmentor_finetune_backbone'] else 'Phase 2 (aggregator-only)')

    print("[MAIN] Parameter Optimization Summary:")
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
    if use_sam:
            print(f"{'SAM (not in optimizer)':<22} | {sam_params:,} ({sam_params/1e6:.2f}M) — frozen via no_grad")
            print("-" * 58)
    print(f"{'Param devices':<22} | {devices}")
    print(f"{'Param dtypes':<22} | {dtypes}", '\n')    

    optimizer_type = args['optim']
    lr, wd = float(args['learning_rate']), float(args['WD'])
    momentum = float(args.get('momentum', 0.9))
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
        print(f"{'Nesterov':<22} | {optimizer.param_groups[0].get('nesterov', False)}")
    elif optimizer_type in ('Adam', 'AdamW'):
            betas = optimizer.param_groups[0]['betas']
            print(f"{'Betas':<22} | {betas}")
            print(f"{'Eps':<22} | {optimizer.param_groups[0]['eps']}")    
    print(f"{'Param groups':<22} | {n_groups}")
    print(f"{'Total opt. params':<22} | {n_opt:,} ({n_opt/1e6:.2f}M)", '\n')

    train_dataloader, val_dataloader = get_train_val_dataloaders(args, img_dim)
    if args['use_scheduler']:
        steps_per_epoch = len(train_dataloader)
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

    if args.get('wandb_watch', False):
        if getattr(saver, 'wandb_enabled', False):
            try:
                wandb_watch_freq = args['wandb_watch_freq']
                wandb.watch(model, log='all', log_freq=wandb_watch_freq, log_graph=True)
                print(f"[MAIN] WandB watch enabled | log=all | freq={wandb_watch_freq}", '\n')
            except Exception as e: print(f"[WARN] wandb.watch failed: {type(e).__name__}: {e}")        
        else: print("[MAIN] --wandb_watch requested but W&B disabled — skipping.", '\n')
        
    best = -1.0
    best_metric_name = args['best_metric']  
    saver.create_best_csv(best_metric_name)                    
    print(f"[MAIN] Tracking best model by '{best_metric_name}' in: {saver.best_csv_file}")

    device_is_cuda = getattr(args['device'], 'type', 'cpu') == 'cuda'
    use_amp = bool(use_cuda_amp) and cuda_available and device_is_cuda
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"[MAIN] AMP enabled: {use_amp} (requested={requested_cuda_amp}, cuda_available={cuda_available}, device={device})", '\n')

    for epoch in range(num_epochs):
        avg_loss, avg_dice, avg_iou= train_single_epoch(train_dataloader, model, sam, optimizer, epoch, args, saver, scheduler, scaler, use_amp)
        saver.log_loss('train_loss', avg_loss, epoch)
        saver.log_loss('train_dice', avg_dice, epoch)
        saver.log_loss('train_iou', avg_iou, epoch)

        print(f"[MAIN] [Epoch {epoch}] train_loss={avg_loss:.4f} | train_dice={avg_dice:.4f} | train_iou={avg_iou:.4f}", '\n')

        with torch.no_grad():
            dice_true, dice_false, iou_val, val_loss = inference_dataloader(val_dataloader, model, sam, epoch, args, saver)
            saver.log_loss('val_loss', val_loss, epoch)
            saver.log_loss('iou', iou_val, epoch)
            saver.log_loss('dice/include_true', dice_true, epoch)
            saver.log_loss('dice/include_false', dice_false, epoch)
            print(f"[MAIN] [Epoch {epoch}] val_loss={val_loss:.4f} | val_dice/include_true={dice_true:.4f} | val_dice/include_false={dice_false:.4f} | val_iou={iou_val:.4f}", '\n')

            best_metric_name = args.get('best_metric', 'dice_f')

            if best_metric_name == 'dice_f': metric = dice_false
            elif best_metric_name == 'dice_t': metric = dice_true
            elif best_metric_name == 'iou': metric = iou_val
            else: raise ValueError(f"Unsupported best_metric '{best_metric_name}'. Choose from: 'dice_f', 'dice_t', 'IoU'.")
                    
            if metric > best:
                torch.save(model.state_dict(), args['best_ckpt_path'])
                best = metric
                print(f'[MAIN] ✓ New best {best_metric_name} = {best:.4f} @ epoch {epoch}')
                saver.log_best(epoch, best)
            print()
        if epoch % int(args.get('save_every', 10)) == 0 or epoch == num_epochs-1: saver.save_model(model, 'net_last', epoch, fixed_name=True)
            
    print(f"[MAIN] ✓ Training complete. Best metric: {best:.4f}")
    
if __name__ == '__main__':
    
    
    parser.add_argument('--best_metric', type=str, default='dice_f', choices=['dice_f', 'dice_t', 'iou'], help='Best metric')
    
    
   
    
    
    
    
    parser.add_argument('--image_key', type=str, default='image',help='Key for image in the data dict')
    parser.add_argument('--mask_key', type=str, default='mask', help='Key for mask in the data dict')
    parser.add_argument('--split_path', type=str, default=os.path.join(PROJECT_DIR, 'data', 'split.json'), help='Path to the JSON split file')
    parser.add_argument('--n_fold', type=int, default=0, help='Fold index')
    parser.add_argument('-bs', '--Batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('-nW', '--nW', type=int, default=0, help='Number of workers for train loader')
    parser.add_argument('-nW_eval', '--nW_eval', type=int, default=0, help='Number of workers for val loader')
    

    
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
    parser.add_argument('--threshold_discretize', type=float, default=0.5, help='Threshold for binarization')
    parser.add_argument('--include_background', type=str2bool, default=False, help='Include background class in loss')
    parser.add_argument('--criterion', type=str, default='dice_ce', choices=['dice', 'dice_ce'], help='Loss function')
    parser.add_argument('--norm_mode', type=str, default='minmax', choices=['minmax', 'sigmoid'], help='Normalization mode for SAM output in Phase 1')
    parser.add_argument('--save_train_images', type=str2bool, default=True, help='Save training images to WandB')
    parser.add_argument('--acc_steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--save_val_images', type=str2bool, default=True, help='Save validation images to WandB')
    parser.add_argument('--save_every', type=int, default=10, help='Save net_last checkpoint every N epochs (also saves at last epoch)')
    

    


    
    
