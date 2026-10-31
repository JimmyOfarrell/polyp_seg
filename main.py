import torch
import argparse
import os
import sys
import shlex
from segment_anything import sam_model_registry
from utils.utils import disable_batchnorm_running_stats, str2bool, set_seed
from utils.model_utils import get_model, get_standard_model
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
        print(f"[MAIN] SAM input resolution          : {img_dim}x{img_dim}")
        print(f"[MAIN] SAM Parameters                : {sam_params:,} ({sam_params/1e6:.2f}M)")
        print(f"[MAIN] SAM Memory                    : {sam_memory_mb:.2f} MB")
        print(f"[MAIN] SAM Device                    : {args['device']}")
        print(f"[MAIN] SAM Mode                      : Frozen (eval)")
        print(f"[MAIN] Image Encoder Parameters      : {sam_image_encoder_params:,} ({sam_image_encoder_params/1e6:.2f}M)")
        print(f"[MAIN] SAM Prompt Encoder Parameters : {sam_prompt_encoder_params:,} ({sam_prompt_encoder_params/1e6:.2f}M)")
        print(f"[MAIN] SAM Mask Decoder Parameters   : {mask_decoder_params:,} ({mask_decoder_params/1e6:.2f}M)")
        if disable_bn: disable_batchnorm_running_stats(sam)
        print()    
    else:
        sam = None
        img_dim = int(args['input_size'])
        print(f"[MAIN] SAM is disabled. Using standard segmentor directly.")
        print(f"[MAIN] Using Idim = {img_dim}×{img_dim} for prompt learner input")
        print()

    if not args['use_aggregator']:
        print('*'*80)
        print(f"[MAIN] Building prompt learner (backbone + decoder) [ModelEmb]:")
        model = get_model(args)
        print('*'*80)
        input('stop')
    else:
        print(f"[MAIN] Building prompt learner (backbone + decoder) + aggregator [HardNetSegmentor]:")
        model = get_standard_model(args)

    model = model.to(device)
    model_params = sum(p.numel() for p in model.parameters())
    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

    print(f"[MAIN] Model Parameters      : {model_params:,}")
    print(f"[MAIN] Trainable Parameters  : {model_trainable_params:,}")
    print(f"[MAIN] Model Memory          : {model_memory_mb:.2f} MB")
    print(f"[MAIN] Model on device       : {device}\n")
    if disable_bn: disable_batchnorm_running_stats(model)          
          

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

    if args['use_sam']:
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
