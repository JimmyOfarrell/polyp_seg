import torch.optim as optim
import torch
import os
from utils.saver import Saver
import sys
from inference import inference_dataloader
from train import train_single_epoch, train_single_epoch
from utils.model_utils import get_model, get_standard_model
from utils.dataset_utils import get_dataset
from segment_anything.utils.transforms import ResizeLongestSide

import wandb


def main(args=None, sam_args=None, saver=None):
    

    
    
    
    
    
    

    
    
    
if __name__ == '__main__':
    

    parser.add_argument('-task', '--task', default='mslesseg', help='Task type (e.g., pancreas, spleen, prostate, brats, mslesseg, tooth)', required=False)
    parser.add_argument('-rotate', '--rotate', default=22, help='image size', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='image size', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='image size', required=False)
    
    
    
    
    parser.add_argument('--data_dir', type=str, default='', help='Path to the dataset directory', required=False)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--spatial_crop_size', type=int, default=224, help='Spatial crop size')
    parser.add_argument('--pos_rand_crop', type=float, default=0.75, help='Positive ratio for random crop')
    parser.add_argument('--neg_rand_crop', type=float, default=0.25, help='Negative ratio for random crop')
    parser.add_argument('--axcodes', type=str, default='RAS', help='Axial codes')
    parser.add_argument('--num_slices', type=int, default=1, help='Number of slices')
    parser.add_argument('--acc_steps', type=int, default=2, help='Number of accumulation steps')
    parser.add_argument('--cache_rate', type=float, default=.7, help='Cache rate')

    





    parser.add_argument('--remove_maxpool', type=str2bool, default=False, help='Remove first maxpool when ResNet is used as embedding model')
    
    
    parser.add_argument('--sam_zeroshot', type=str2bool, default=False, help='SAM zero shot')
    
    


    parser.add_argument('--label_values', type=int, nargs='+', default=[1, 2, 4], help='Label values to discretize mask (primarily for the BRATS dataset; may not apply to other datasets)')
    parser.add_argument('--center_crop', type=str2bool, default=False, help='Center crop the input images')
    parser.add_argument('--train_with_crop', type=str2bool, default=True, help='Train with crop')

    parser.add_argument('--decoder_channels', type=int, nargs='+', default=[], help='Decoder channels for UnetLike segmentor/aggregator')
    
    
    

    
    

