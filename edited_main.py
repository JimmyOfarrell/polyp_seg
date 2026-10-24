import os
import argparse
import sys
import shlex
from utils.utils import disable_batchnorm_running_stats, str2bool, set_seed
from utils.saver import Saver
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
============================================================
            POLYP SEGMENTATION FRAMEWORK
============================================================
Method : SAM-Guided Prompt Learning for Polyp Segmentation
Task   : Automated Polyp Detection and Segmentation
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
    parser.add_argument('--wandb_project', default='PS_PROJECT', help='WandB project name')
    parser.add_argument('--wandb_entity', default=None, help='WandB entity (username or team)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='Wandb tags')
    parser.add_argument('--use_sam', type=str2bool, default=True, help='Use SAM (phase 1) vs aggregator only (phase 2)')
    parser.add_argument('--sam_ckpt', type=str, default='sam_vit_b', choices = ['sam_vit_b', 'medsam_vit_b', 'sam_vit_h'], help='SAM checkpoint')
    parser.add_argument('--sam_version', type=str, default='vit_b', choices = ['vit_b', 'vit_h'], help='SAM version')
    parser.add_argument('--device_id', type=int, default=0, help='GPU index')
    args = vars(parser.parse_args())
    print(parser.description)

    set_seed(args['seed'], strict=args['strict_determinism'])

    args['experiment_name']= args['exp_name']
    output_root = os.path.join(PROJECT_DIR, 'results')

    saver = Saver(output_folder=output_root, args=args)
    cmd = f"{sys.executable} " + " ".join(shlex.quote(arg) for arg in sys.argv)
    saver.log_cmd(cmd)

    args['exp_folder'] = saver.path
    args['last_ckpt_path'] = os.path.join(saver.ckpt_path, 'net_last.pth')
    args['best_ckpt_path'] = os.path.join(saver.ckpt_path, 'net_best.pth')
    args['vis_folder'] = saver.vis_path
    saver.log_hparams(args)

    if args['use_sam']:
        sam_ckpt = os.path.join(PROJECT_DIR, 'cp', f"{args['sam_ckpt']}.pth")
        if not os.path.exists(sam_ckpt): raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt}\n Please download {args['sam_ckpt']}.pth into the cp/ folder.")
        sam_args = {
            'sam_checkpoint': sam_ckpt,
            'model_type': args['sam_version'],
            'generator_args': {
                'points_per_side': 8,
                'pred_iou_thresh': 0.95,
                'stability_score_thresh': 0.7,
                'crop_n_layers': 0,
                'crop_n_points_downscale_factor': 2,
                'min_mask_region_area': 0,
                'point_grids': None,
                'box_nms_thresh': 0.7,
            },
            'gpu_id': args['device_id']}
        print(f"[MAIN] SAM checkpoint path: {sam_ckpt}")
        print(f"[MAIN] SAM Model type : {args['sam_version']}")
    else:
        sam_args = None
        print("[MAIN] SAM is disabled.")
    print('='*60)

    main(args=args, sam_args=sam_args, saver=saver)
