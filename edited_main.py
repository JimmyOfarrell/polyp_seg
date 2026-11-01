import wandb


from train import train_single_epoch
from inference import inference_dataloader
from utils.saver import Saver

import torch

import argparse
import os
import sys
import shlex

from utils.utils import disable_batchnorm_running_stats, str2bool, set_seed




def main(args=None, sam_args=None, saver=None):
    

    
        
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
    
    
   
    
    
    
    
    
    

    
    

    parser.add_argument('--threshold_discretize', type=float, default=0.5, help='Threshold for binarization')
    parser.add_argument('--include_background', type=str2bool, default=False, help='Include background class in loss')
    parser.add_argument('--criterion', type=str, default='dice_ce', choices=['dice', 'dice_ce'], help='Loss function')
    parser.add_argument('--norm_mode', type=str, default='minmax', choices=['minmax', 'sigmoid'], help='Normalization mode for SAM output in Phase 1')
    parser.add_argument('--save_train_images', type=str2bool, default=True, help='Save training images to WandB')
    
    parser.add_argument('--save_val_images', type=str2bool, default=True, help='Save validation images to WandB')
    parser.add_argument('--save_every', type=int, default=10, help='Save net_last checkpoint every N epochs (also saves at last epoch)')
    

    


    
    
