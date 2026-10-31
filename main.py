import argparse
import os
from utils.utils import str2bool, set_seed
from utils.saver import Saver

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
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
    print(parser.description)
    args = vars(parser.parse_args())

    print('='*120)
    seed = args['seed']
    strict = args['strict_determinism']
    set_seed(seed=seed, strict=strict)

    output_root = os.path.join(PROJECT_DIR, 'results')
    saver = Saver(output_folder=output_root, args=args)