import argparse




import os


from utils.saver import Saver



import torch.utils.data
import torch

    from segment_anything.utils.transforms import ResizeLongestSide


import numpy as np
from utils.utils import get_input_dict, norm_batch



from utils.metrics_utils import stratified_lesion_analysis, compute_lesionwise_metrics







@torch.no_grad()
def inference_dataloader(test_context, epoch):

    
    
    

        



       
                
        
    
    
    
    
    
    


    
    
    
    
    return np.nanmean(dice_list_include_background), np.nanmean(dice_list_exclude_background), np.nanmean(iou_list_include_background), np.nanmean(iou_list_exclude_background)

def main(args=None, sam_args= None, saver=None):
    
        test_results[dataset_name] = {
        'dice_include_background': dice_inc,
        'dice_exclude_background': dice_exc,
        'iou_include_background':  iou_inc,
        'iou_exclude_background':  iou_exc,
    }
    print(f"\n{'='*60}")
    print(f"[TEST] Final Summary (all datasets):")
    print(f"{'='*60}")
    print(f"{'Dataset':<22} | {'Dice (polyp)':>12} | {'IoU (polyp)':>12}")
    print(f"{'-'*22}-+-{'-'*12}-+-{'-'*12}")
    for name, r in test_results.items():
        print(f"{name:<22} | {r['dice_exclude_background']:>12.4f} | {r['iou_exclude_background']:>12.4f}")
    print(f"{'='*60}")
    
if __name__ == '__main__':
    

    
    
    
    

    
    parser.add_argument('-rotate', '--rotate', default=22, help='image size', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='image size', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='image size', required=False)
    
    parser.add_argument('--data_dir', type=str, default='', help='Path to the dataset directory', required=False)
    parser.add_argument('--save_every', type=int, default=10, help='Save model every n epochs')
    parser.add_argument('--spatial_crop_size', type=int, default=224, help='Spatial crop size')
    parser.add_argument('--pos_rand_crop', type=float, default=0.75, help='Positive ratio for random crop')
    parser.add_argument('--neg_rand_crop', type=float, default=0.25, help='Negative ratio for random crop')
    parser.add_argument('--axcodes', type=str, default='RAS', help='Axial codes')
    parser.add_argument('--num_slices', type=int, default=1, help='Number of slices')
    parser.add_argument('--criterion', type=str, default='dice', choices=['dice', 'dice_ce'], help='Criterion')
    
    parser.add_argument('--acc_steps', type=int, default=4, help='Number of accumulation steps')
    parser.add_argument('--cache_rate', type=float, default=.7, help='Cache rate')

    
    
    
    
    
    parser.add_argument('--remove_maxpool', type=str2bool, default=False, help='Remove first maxpool when ResNet is used as embedding model')

    parser.add_argument('--sam_zeroshot', type=str2bool, default=False, help='SAM zero shot')

    parser.add_argument('--decoder_channels', type=int, nargs='+', default=[], help='Decoder channels for UnetLike segmentor/aggregator')

    

    
