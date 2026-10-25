import torch
import os
import argparse
import sys
import shlex
from segment_anything import sam_model_registry
from utils.utils import disable_batchnorm_running_stats, str2bool, set_seed
from utils.dataset_utils import get_train_val_dataloaders
from utils.model_utils import get_model, get_standard_model
from utils.saver import Saver

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def main(args=None, sam_args=None, saver=None):
    print('\n'+'='*60)
    print("[MAIN] Device configuration")
    if str(args['device']).lower() == 'cuda':
        if not torch.cuda.is_available():
            print(f"[WARNING] CUDA requested but not available. Falling back to CPU.")
            args['device'] = torch.device("cpu")
        else:
            num_gpus = torch.cuda.device_count()
            device_id = int(args.get('device_id', 0))
            if device_id >= num_gpus or device_id < 0:
                print(f"[WARNING] device_id={device_id} is out of range (only {num_gpus} GPU(s) available). Using device 0.")
                device_id = 0
                args['device_id'] = 0    
            args['device'] = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(device_id) 

            gpu_name = torch.cuda.get_device_name(device_id)
            gpu_props = torch.cuda.get_device_properties(device_id)
            total_vram_gb = gpu_props.total_memory / (1024**3)

            print(f"[MAIN] Device        : {args['device']}")
            print(f"[MAIN] GPU name      : {gpu_name}")
            print(f"[MAIN] Total VRAM    : {total_vram_gb:.2f} GB")
            print(f"[MAIN] CUDA version  : {torch.version.cuda}")
            print(f"[MAIN] PyTorch version: {torch.__version__}")

            cc = (gpu_props.major, gpu_props.minor)
            print(f"[MAIN] Compute capability: {cc[0]}.{cc[1]}")

            if cc < (6, 0): print(f"[WARNING] GPU compute capability {cc[0]}.{cc[1]} is below 6.0. Mixed precision (AMP) may not work properly.")
            if cc < (7, 0) and args.get('use_cuda_amp', False): print(f"[WARNING] GPU compute capability {cc[0]}.{cc[1]} is below 7.0. Tensor Cores not available; AMP will be limited.")

            try:
                free_vram, total_vram = torch.cuda.mem_get_info(device_id)
                free_vram_gb = free_vram / (1024**3)
                print(f"[MAIN] Free VRAM     : {free_vram_gb:.2f} GB")
            
                if args.get('use_sam', False) and free_vram_gb < 8:
                    print(f"[WARNING] Free VRAM ({free_vram_gb:.2f} GB) may be insufficient for Phase 1 with SAM (ViT-B). Consider:")
                    print(f"         - Smaller batch size (-bs)")
                    print(f"         - Using --use_cuda_amp True")
                    print(f"         - Already using smallest SAM (vit_b); reduce batch size instead")
            except Exception: pass 
    else:
        args['device'] = torch.device("cpu")
        print(f"[MAIN] Device        : CPU")
        print(f"[MAIN] PyTorch version: {torch.__version__}")
        if args.get('use_sam', False):
            print(f"[WARNING] Running Phase 1 with SAM on CPU will be extremely slow!")
            print(f"         Consider using a GPU for training.")
        if args.get('use_cuda_amp', False):
            print(f"[WARNING] AMP is enabled but device is CPU. Disabling AMP.")
            args['use_cuda_amp'] = False
    print('='*60)

    print('\n'+'='*60)
    if args['use_sam']:
        print("[MAIN] SAM model (Phase 1: SAM-guided prompt learning) Configuration")
        try: sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
        except Exception as e: raise RuntimeError(f"Failed to load SAM '{sam_args['model_type']}' from '{sam_args['sam_checkpoint']}': {e}")
        sam.to(device=args['device'])
        sam.eval()
        for param in sam.parameters(): param.requires_grad = False
        sam_params = sum(p.numel() for p in sam.parameters())
        sam_memory_mb = sum(p.numel() * p.element_size() for p in sam.parameters()) / (1024**2)
        img_dim = sam.image_encoder.img_size
        print(f"[MAIN] SAM input resolution: {img_dim}x{img_dim}")
        print(f"[MAIN] SAM Parameters: {sam_params:,}")
        print(f"[MAIN] SAM Memory: {sam_memory_mb:.2f} MB")
        print(f"[MAIN] SAM Device: {args['device']}")
        print(f"[MAIN] SAM Mode: Frozen (eval)")
        if args['disable_batchnorm_running_stats']:
            disable_batchnorm_running_stats(sam)
            print("[MAIN] BatchNorm running stats disabled for SAM.")
    else:
        sam = None
        img_dim = int(args['Idim'])
        print("[MAIN] SAM is disabled. Using standard segmentor directly.")
        print(f"[MAIN] Using Idim = {img_dim}×{img_dim} for prompt learner input")
    print('='*60)

    train_dataloader, val_dataloader = get_train_val_dataloaders(args, img_dim)
    if not args['use_standard_net']:
        model = get_model(args)
    else:
        model = get_standard_model(args)


    """if not args['use_standard_net']:
        model = get_model(args)
        print(f"[MAIN] Phase 1 model: {type(model).__name__} (prompt learner)")
    else:
        model = get_standard_model(args)
        print(f"[MAIN] Phase 2 model: {type(model).__name__} (aggregator)")
    model = model.to(args['device'])
    print(f"[MAIN] Model moved to {args['device']}")"""
    input('STOP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


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
