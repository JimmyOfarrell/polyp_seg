import wandb



from inference import inference_dataloader
from utils.saver import Saver

import torch

import argparse
import os
import sys
import shlex

from utils.utils import disable_batchnorm_running_stats, str2bool, set_seed




def main(args=None, sam_args=None, saver=None):
    

    
        
    

    

    
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


    
    
    
    
    
    parser.add_argument('--save_val_images', type=str2bool, default=True, help='Save validation images to WandB')
    parser.add_argument('--save_every', type=int, default=10, help='Save net_last checkpoint every N epochs (also saves at last epoch)')
    

    


    
    
