import argparse
import os
import sys
import shlex
from utils.utils import str2bool
from utils.saver import Saver

from utils.dataset_utils import get_all_test_dataloaders


import torch.utils.data
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm.auto import tqdm
import torch.nn.functional as F
import numpy as np
from utils.utils import get_input_dict, norm_batch
from monai.transforms import AsDiscrete, Activations
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from dataset.polyp_dataset import get_test_mslesseg_dataset
from utils.model_utils import get_model, sam_call, get_standard_model
from utils.metrics_utils import stratified_lesion_analysis, compute_lesionwise_metrics

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

@torch.no_grad()
def inference_dataloader(test_dataloader, model, sam, device, epoch, args, saver):
    model.eval()
    if sam is not None: sam.eval()

    pbar = tqdm(test_dataloader, desc=f"Test")
    threshold  = args['threshold_discretize']
    use_standard_net = args['use_standard_net']
    input_resolution = int(args['Idim'])
    apply_sigm = bool(getattr(model, "is_segmentor", False) and use_standard_net)
    image_key, mask_key = args['image_key'], args['mask_key']
    norm_mode = args['norm_mode'] 
    save_embeddings = args['save_embeddings']


    one_hot = AsDiscrete(to_onehot=2, dim=1)
    discretize_labels = AsDiscrete(threshold=threshold)
    sigmoid = Activations(sigmoid=True)

    mae_list = []
    dice_list_include_background, dice_list_exclude_background = [], []
    iou_list_include_background, iou_list_exclude_background = [], [] 
    hd_list_include_background, hd_list_exclude_background = [], []  
    hd95_list_include_background, hd95_list_exclude_background = [], []
    surface_distance_list_include_background, surface_distance_list_exclude_background = [], []

    metrics_list_include_background, metrics_list_exclude_background = {}, {}
    metrics_list_lesionwise = {}
    metrics_list_stratified = {}
    metrics_per_el = {}
    
    dice_metric_include_background = DiceMetric(include_background=True, reduction='mean')
    dice_metric_exclude_background = DiceMetric(include_background=False, reduction='mean')
    iou_metric_include_background = MeanIoU(include_background=True, reduction='mean')
    iou_metric_exclude_background = MeanIoU(include_background=False, reduction='mean')
    hd_metric_include_background = HausdorffDistanceMetric(include_background=True, reduction='mean')
    hd_metric_exclude_background = HausdorffDistanceMetric(include_background=False, reduction='mean')
    hd95_metric_include_background = HausdorffDistanceMetric(include_background=True, reduction='mean', percentile=95)
    hd95_metric_exclude_background = HausdorffDistanceMetric(include_background=False, reduction='mean', percentile=95)
    surface_distance_metric_include_background = SurfaceDistanceMetric(include_background=True, reduction='mean', symmetric=True)
    surface_distance_metric_exclude_background = SurfaceDistanceMetric(include_background=False, reduction='mean', symmetric=True)

    metric_names = ('sensitivity', 'specificity', 'accuracy', 'precision', 'f1_score', 'false_negative_rate', 'false_positive_rate', 'threat_score')
    metric_names_mapping = {'sensitivity': "Sens(Rec)", 
                           'specificity': 'Spec-TNR', 
                           'accuracy': 'Acc', 
                           'precision': 'Prec',
                           'f1_score': 'F1',
                           'false_negative_rate': 'FNR',
                           'false_positive_rate': 'FPR', 
                           'threat_score': 'Jaccard'}
    conf_matr_metric_include_background = ConfusionMatrixMetric(include_background=True, metric_name = metric_names, compute_sample=True)
    conf_matr_metric_exclude_background = ConfusionMatrixMetric(include_background=False, metric_name = metric_names, compute_sample=True)
    
    for m in metric_names:
        metrics_list_include_background[m] = []
        metrics_list_exclude_background[m] = []
    
    for ix,  sample in enumerate(pbar):
        imgs, gts = sample[image_key], sample[mask_key]
        batch_size = imgs.shape[0]
        if batch_size != 1: raise ValueError(f"Val batch_size must be 1, got {imgs.shape[0]}. Check val_dataloader.")
        original_sz_batch, img_sz = sample['original_shape'], torch.tensor(imgs.shape[2:]).repeat(batch_size, 1)
        orig_imgs = imgs.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)
        
        spacing = (1.0, 1.0)
        filename = os.path.splitext(os.path.basename(str(imgs.meta['filename_or_obj'])))[0]
        metrics_per_el[filename] = {'include_background_true' : {}, 'include_background_false' : {}, 'lesionwise' : {}, 'stratified' : {}}
        tensor_per_el = {}

        if sam is not None:
            orig_imgs_small = F.interpolate(orig_imgs, size=(input_resolution, input_resolution), mode='bilinear', align_corners=False)
            dense_embeddings = model(orig_imgs_small)
            batched_input = get_input_dict(orig_imgs, original_sz_batch, img_sz)
            masks = norm_batch(sam_call(batched_input, sam, dense_embeddings), mode=norm_mode)

            if save_embeddings:
                dense_embs_mean = dense_embeddings.mean(dim=1)
                saver.save_batch(dense_embeddings.detach().cpu(), f'embs', epoch, subfolder=filename)
                saver.save_batch(dense_embs_mean.detach().cpu(), f'embs_mean', epoch, subfolder=filename, remove_batch_dim=False, normalize=True)
        else: masks = model(orig_imgs)

        size = masks.shape[2:]
        imgs_resized = F.interpolate(orig_imgs, size=size, mode='bilinear', align_corners=False)
        gts_resized = F.interpolate(gts, size=size, mode='nearest')
        onehot_gts=one_hot(discretize_labels(gts_resized))

        if masks.shape[1] == 1: onehot_masks = one_hot(discretize_labels(masks))
        elif masks.shape[1] == 2:
            probs = sigmoid(masks) if apply_sigm else masks
            pred_polyp = discretize_labels(probs[:, 1:2])
            onehot_masks = torch.cat([1.0 - pred_polyp, pred_polyp], dim=1)
        else: raise ValueError(f"Unexpected mask channels: {masks.shape[1]}")

        if save_embeddings:
            if masks.shape[1] == 2: save_pred = masks[:,1,:,:].unsqueeze(1)
            elif masks.shape[1] == 1: save_pred = masks
            else: raise ValueError(f"Unexpected mask channels: {masks.shape[1]}")

            if apply_sigm: save_pred = discretize_labels(sigmoid(save_pred))
            else: save_pred = discretize_labels(save_pred)

            if orig_imgs.shape[1] == 3:
                mean = torch.tensor([0.485, 0.456, 0.406], device=orig_imgs.device).view(1, 3, 1, 1)
                std  = torch.tensor([0.229, 0.224, 0.225], device=orig_imgs.device).view(1, 3, 1, 1)
                orig_imgs_viz = (orig_imgs * std + mean).clamp(0, 1)
            else: orig_imgs_viz = orig_imgs    

            saver.save_batch(orig_imgs_viz.detach().cpu(), f'image', epoch, subfolder=filename, remove_batch_dim=False)
            saver.save_batch(gts_resized.detach().cpu(), f'gt', epoch, subfolder=filename, remove_batch_dim=False)
            saver.save_batch(save_pred.detach().cpu(), f'pred_discrete', epoch, subfolder=filename, remove_batch_dim=False)

        if args['save_tensors']:
            tensor_per_el['masks'] = onehot_masks.detach().cpu()
            tensor_per_el['gts'] = gts_resized.detach().cpu()
            tensor_per_el['imgs'] = orig_imgs.detach().cpu()
            saver.save_data(tensor_per_el, filename, subfolder='tensors')
        
        if args['save_test_images']:
            if onehot_masks.shape[1] == 2: pred_polyp = onehot_masks[:, 1:2].detach()
            elif onehot_masks.shape[1] == 1: pred_polyp = onehot_masks.detach()
            else: raise ValueError(f"Unexpected pred_binary channels: {onehot_masks.shape[1]}")

            saver.log_samples_2d(images=imgs_resized.detach(), masks=pred_polyp, ground_truth=gts_resized.detach(), step=epoch, name=filename, phase='test')
        
        dice_metric_include_background(onehot_masks, onehot_gts)
        dice_metric_exclude_background(onehot_masks, onehot_gts)
        dice_include_background = dice_metric_include_background.aggregate().item()
        dice_exclude_background = dice_metric_exclude_background.aggregate().item()

        iou_metric_include_background(onehot_masks, onehot_gts)
        iou_metric_exclude_background(onehot_masks, onehot_gts)
        iou_include_background = iou_metric_include_background.aggregate().item()
        iou_exclude_background = iou_metric_exclude_background.aggregate().item()

        iou_list_include_background.append(iou_include_background)
        iou_list_exclude_background.append(iou_exclude_background)
        dice_list_include_background.append(dice_include_background)
        dice_list_exclude_background.append(dice_exclude_background)
        
        hd_metric_include_background(onehot_masks, onehot_gts, spacing=spacing)
        hd_metric_exclude_background(onehot_masks, onehot_gts, spacing=spacing)
        hd_include_background = hd_metric_include_background.aggregate().item()
        hd_exclude_background = hd_metric_exclude_background.aggregate().item()
        
        hd95_metric_include_background(onehot_masks, onehot_gts, spacing=spacing)
        hd95_metric_exclude_background(onehot_masks, onehot_gts, spacing=spacing)
        hd95_include_background = hd95_metric_include_background.aggregate().item()
        hd95_exclude_background = hd95_metric_exclude_background.aggregate().item()

        hd_list_include_background.append(hd_include_background)
        hd_list_exclude_background.append(hd_exclude_background)
        hd95_list_include_background.append(hd95_include_background)
        hd95_list_exclude_background.append(hd95_exclude_background)
        
        surface_distance_metric_include_background(onehot_masks, onehot_gts, spacing=spacing)
        surface_distance_metric_exclude_background(onehot_masks, onehot_gts, spacing=spacing)
        surface_distance_include_background = surface_distance_metric_include_background.aggregate().item()
        surface_distance_exclude_background = surface_distance_metric_exclude_background.aggregate().item()

        surface_distance_list_include_background.append(surface_distance_include_background)
        surface_distance_list_exclude_background.append(surface_distance_exclude_background)
        
        conf_matr_metric_include_background(onehot_masks, onehot_gts)
        conf_matr_metric_exclude_background(onehot_masks, onehot_gts)
        metrics_true = conf_matr_metric_include_background.aggregate()
        metrics_false = conf_matr_metric_exclude_background.aggregate()
        
        metrics_per_el[filename]['include_background_true']['dice_score'] = dice_include_background
        metrics_per_el[filename]['include_background_false']['dice_score'] = dice_exclude_background
        metrics_per_el[filename]['include_background_true']['iou_score'] = iou_include_background
        metrics_per_el[filename]['include_background_false']['iou_score'] = iou_exclude_background
        metrics_per_el[filename]['include_background_true']['hausdorff_distance'] = hd_include_background
        metrics_per_el[filename]['include_background_false']['hausdorff_distance'] = hd_exclude_background
        metrics_per_el[filename]['include_background_true']['hausdorff_distance_95'] = hd95_include_background
        metrics_per_el[filename]['include_background_false']['hausdorff_distance_95'] = hd95_exclude_background
        metrics_per_el[filename]['include_background_true']['average_surface_distance'] = surface_distance_include_background
        metrics_per_el[filename]['include_background_false']['average_surface_distance'] = surface_distance_exclude_background

        pred_bin = onehot_masks[:, 1, ...].cpu().numpy().squeeze()
        gt_bin = onehot_gts[:, 1, ...].cpu().numpy().squeeze()

        lesion_metrics = compute_lesionwise_metrics(pred_bin, gt_bin)
        for k, v in lesion_metrics.items():
            metrics_per_el[filename]['lesionwise'][k] = v
            if k not in metrics_list_lesionwise.keys():
                metrics_list_lesionwise[k] = []
            metrics_list_lesionwise[k].append(v)

        strat_metrics = stratified_lesion_analysis(pred_bin, gt_bin) 
        for k, v in strat_metrics.items():
            metrics_per_el[filename]['stratified'][k] = v
            if k not in metrics_list_stratified.keys():
                metrics_list_stratified[k] = []
            metrics_list_stratified[k].append(v)
    
        for i,m in enumerate(metric_names):       
            metrics_per_el[filename]['include_background_true'][m]= metrics_true[i].item()
            metrics_per_el[filename]['include_background_false'][m]= metrics_false[i].item()
            metrics_list_include_background[m].append(metrics_true[i].item())
            metrics_list_exclude_background[m].append(metrics_false[i].item())
                
        pbar.set_description(
            '(Inference | {task}) Epoch {epoch} :: Dice True {dice_true:.4f} :: Dice False {dice_false:.4f} :: IoU True {iou_true:.4f} :: IoU False {iou_false:.4f}'.format(
                task=args['task'],
                epoch=epoch,
                dice_true=np.mean(dice_list_include_background),
                dice_false=np.mean(dice_list_exclude_background),
                iou_true=np.mean(iou_list_include_background), 
                iou_false=np.mean(iou_list_exclude_background)))
        
        iou_metric_include_background.reset()
        iou_metric_exclude_background.reset()
        dice_metric_include_background.reset()
        dice_metric_exclude_background.reset()
        hd_metric_include_background.reset()
        hd_metric_exclude_background.reset()
        hd95_metric_include_background.reset()
        hd95_metric_exclude_background.reset()
        conf_matr_metric_include_background.reset()
        conf_matr_metric_exclude_background.reset()
        surface_distance_metric_include_background.reset()
        surface_distance_metric_exclude_background.reset()
    
    if args['log_histograms']:
        saver.log_histogram('dice_include', dice_list_include_background, epoch)
        saver.log_histogram('dice_exclude', dice_list_exclude_background, epoch)
        saver.log_histogram('iou_include', iou_list_include_background, epoch)
        saver.log_histogram('iou_exclude', iou_list_exclude_background, epoch)
        saver.log_histogram('hd_include', hd_list_include_background, epoch)
        saver.log_histogram('hd_exclude', hd_list_exclude_background, epoch)
        saver.log_histogram('hd95_include', hd95_list_include_background, epoch)
        saver.log_histogram('hd95_exclude', hd95_list_exclude_background, epoch)
        saver.log_histogram('asd_include', surface_distance_list_include_background, epoch)
        saver.log_histogram('asd_exclude', surface_distance_list_exclude_background, epoch)
        saver.log_histogram('mae', mae_list, epoch)
        for m in metric_names:
            saver.log_histogram(f'{m}_include', metrics_list_include_background[m], epoch)
            saver.log_histogram(f'{m}_exclude', metrics_list_exclude_background[m], epoch)
    
    saver.save_json(metrics_per_el, 'metrics_per_el')
    saver.log_loss('dice_include', np.nanmean(dice_list_include_background), epoch)
    saver.log_loss('dice_exclude', np.nanmean(dice_list_exclude_background), epoch)
    saver.log_loss('iou_include', np.nanmean(iou_list_include_background), epoch)
    saver.log_loss('iou_exclude', np.nanmean(iou_list_exclude_background), epoch)
    saver.log_loss('hd_include', np.nanmean(hd_list_include_background), epoch)
    saver.log_loss('hd_exclude', np.nanmean(hd_list_exclude_background), epoch)
    saver.log_loss('hd95_include', np.nanmean(hd95_list_include_background), epoch)
    saver.log_loss('hd95_exclude', np.nanmean(hd95_list_exclude_background), epoch)
    saver.log_loss('asd_include', np.nanmean(surface_distance_list_include_background), epoch)
    saver.log_loss('asd_exclude', np.nanmean(surface_distance_list_exclude_background), epoch)
    saver.log_loss('mae', np.nanmean(mae_list), epoch)
    for m in metric_names:
        saver.log_loss(f'{m}_include', np.nanmean(metrics_list_include_background[m]), epoch)
        saver.log_loss(f'{m}_exclude', np.nanmean(metrics_list_exclude_background[m]), epoch)
    
    m_dict_log = {
        'ice_include': np.nanmean(dice_list_include_background),
        'ice_exclude': np.nanmean(dice_list_exclude_background),
        'iou_include': np.nanmean(iou_list_include_background),
        'iou_exclude': np.nanmean(iou_list_exclude_background),
        'hd_include': np.nanmean(hd_list_include_background), 
        'hd_exclude': np.nanmean(hd_list_exclude_background), 
        'hd95_include': np.nanmean(hd95_list_include_background), 
        'hd95_exclude': np.nanmean(hd95_list_exclude_background), 
        'asd_include': np.nanmean(surface_distance_list_include_background), 
        'asd_exclude': np.nanmean(surface_distance_list_exclude_background),
        'mae': np.nanmean(mae_list),
    }
    for m in metric_names:
        m_dict_log[f'{metric_names_mapping[m]}_include'] = np.mean(metrics_list_include_background[m])
        m_dict_log[f'{metric_names_mapping[m]}_exclude'] = np.mean(metrics_list_exclude_background[m])

    for m in metrics_list_lesionwise.keys():
        saver.log_loss(f'{m}_lesionwise', np.nanmean(metrics_list_lesionwise[m]), epoch)
        m_dict_log[f'{m}_lesionwise'] = np.nanmean(metrics_list_lesionwise[m])
    for m in metrics_list_stratified.keys():
        saver.log_loss(f'{m}_stratified', np.nanmean(metrics_list_stratified[m]), epoch)
        m_dict_log[f'{m}_stratified'] = np.nanmean(metrics_list_stratified[m])

    m_dict_log['Exp_name'] = args['exp_name']
    m_dict_log['Step'] = epoch + 1
    
    if saver.metrics_table is None:
        columns = list(m_dict_log.keys())
        saver.create_table(columns)
    saver.log_table_row(epoch +1, m_dict_log)
    
    return np.nanmean(dice_list_include_background), np.nanmean(dice_list_exclude_background), np.nanmean(iou_list_include_background), np.nanmean(iou_list_exclude_background)

def main(args=None, sam_args= None, saver=None):
    requested_device = args['device']
    cuda_available = torch.cuda.is_available()

    use_sam = args['use_sam']
    
    print("[TEST] Device configuration:")
    if str(requested_device).lower() == 'cuda' and cuda_available:
        num_gpus = torch.cuda.device_count()
        device_id = int(args['device_id'])

        if device_id >= num_gpus or device_id < 0:
            print(f"[WARNING] device_id={device_id} is out of range (only {num_gpus} GPU(s) available). Using device 0.")
            device_id = 0
            args['device_id'] = 0 
              
        args['device'] = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id) 

        gpu_name = torch.cuda.get_device_name(device_id)
        gpu_props = torch.cuda.get_device_properties(device_id)
        total_vram_gb = gpu_props.total_memory / (1024**3)
        cc = (gpu_props.major, gpu_props.minor)

        print(f"[TEST] Device            : {args['device']}")
        print(f"[TEST] GPU name          : {gpu_name}")
        print(f"[TEST] Total VRAM        : {total_vram_gb:.2f} GB")
        print(f"[TEST] CUDA version      : {torch.version.cuda}")
        print(f"[TEST] PyTorch version   : {torch.__version__}")
        print(f"[TEST] Compute capability: {cc[0]}.{cc[1]}")

        try:
            free_vram, total_vram = torch.cuda.mem_get_info(device_id)
            free_vram_gb = free_vram / (1024**3)
            print(f"[TEST] Free VRAM     : {free_vram_gb:.2f} GB")
        except Exception as e: print(f"[WARNING] Could not query VRAM: {type(e).__name__}: {e}") 
    else:
        if str(requested_device).lower() == 'cuda' and not cuda_available: print(f"[WARNING] CUDA requested but not available. Falling back to CPU.")
        args['device'] = torch.device("cpu")
        print(f"[TEST] Device         : CPU")
        print(f"[TEST] PyTorch version: {torch.__version__}")
        if use_sam: print(f"[WARNING] Running Phase 1 with SAM on CPU will be extremely slow! Consider using a GPU for training.")
    device = args['device']
    print()

    if use_sam:
        print("[TEST] SAM model (Phase 1: SAM-guided prompt learning) Configuration:")
        try: sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
        except Exception as e: raise RuntimeError(f"Failed to load SAM '{sam_args['model_type']}' from '{sam_args['sam_checkpoint']}': {e}")
        sam.to(device=device)
        sam.eval()
        sam_params = sum(p.numel() for p in sam.parameters())
        sam_memory_mb = sum(p.numel() * p.element_size() for p in sam.parameters()) / (1024**2)
        img_dim = sam.image_encoder.img_size
        print(f"[TEST] SAM input resolution: {img_dim}x{img_dim}")
        print(f"[TEST] SAM Parameters      : {sam_params:,}")
        print(f"[TEST] SAM Memory          : {sam_memory_mb:.2f} MB")
        print(f"[TEST] SAM Device          : {args['device']}")
        print(f"[TEST] SAM Mode            : Frozen (eval)\n")   
    else:
        sam = None
        img_dim = int(args['Idim'])
        print("[TEST] SAM is disabled. Using standard segmentor directly.")
        print(f"[TEST] Using Idim = {img_dim}×{img_dim} for prompt learner input\n")
    

    if not args['use_standard_net']:
        print(f"[TEST] Building ModelEmb (Phase 1: SAM-guided prompt learning):")
        model = get_model(args)
    else:
        print(f"[TEST] Building HardNetSegmentor (Phase 2: aggregator-only):")
        model = get_standard_model(args)
    
    model = model.to(device)
    print(f"[TEST] Model on device: {device}\n")

    test_dataloaders = get_all_test_dataloaders(args, img_dim)
        
    test_results = {}
    for dataset_idx, (dataset_name, test_dataloader) in enumerate(test_dataloaders.items()):
        print(f"\n{'='*60}")
        print(f"[TEST] Dataset {dataset_idx+1}/{len(test_dataloader)}: {dataset_name}")
        print(f"{'='*60}")
        dice_inc, dice_exc, iou_inc, iou_exc = inference_dataloader(test_dataloader, model, sam, device, epoch=dataset_idx, args=args, saver=saver)
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
    parser.add_argument('--exp_name', required=True, help='Experiment name (required)')
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'], help='WandB mode')
    parser.add_argument('--wandb_project', default='SGPL_Polyp_Seg', help='WandB project name')
    parser.add_argument('--wandb_entity', default=None, help='WandB entity (username or team)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=['test'], help='Wandb tags')
    parser.add_argument('--best_metric', type=str, default='dice_f', choices=['dice_f', 'dice_t', 'iou'], help='Best metric')
    parser.add_argument('--sam_ckpt', type=str, default='sam_vit_b', choices = ['sam_vit_b', 'medsam_vit_b', 'sam_vit_h'], help='SAM checkpoint')
    parser.add_argument('--device_id', type=int, default=0, help='GPU index')
    parser.add_argument('--sam_version', type=str, default='vit_b', choices = ['vit_b', 'vit_h'], help='SAM version')
    parser.add_argument('--use_sam', type=str2bool, default=True, help='Use SAM')
    parser.add_argument('--device', default='cuda', help="Device: 'cuda' or 'cpu'")
    parser.add_argument('-Idim', '--Idim', type=int, default=512, help='Input image size (resize resolution)')
    parser.add_argument('--use_standard_net', type=str2bool, default=False, help='Use standard segmentor (phase 2) vs prompt learner (phase 1)')
    parser.add_argument('--model_emb', type=str, default='HardNet', help='Prompt learner backbone')
    parser.add_argument('-depth_wise', '--depth_wise', type = str2bool, default=False, help='Use depthwise HarDNet')
    parser.add_argument('-order', '--order', default=68, type=int, help='HarDNet variant: 39 | 68 | 85')
    parser.add_argument('--pretrained', type=str2bool, default=True, help='Use ImageNet pretrained weights')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to trained checkpoint (required for testing)')
    parser.add_argument('--net', type=str, default='HardNetSegmentor', help='Segmentor net (phase 2)')
    parser.add_argument('--net_segmentor', type=str, default='PointwiseConv', choices=['PointwiseConv', 'ConvUpsample', 'TransposeConvUpsample'], help='Aggregator head')
    parser.add_argument('--out_channels', type=int, default=2, help='Output channels')
    parser.add_argument('--upsample_factor_list', type=int, nargs='+', default=[2, 2], help='Upsample factors for ConvUpsample aggregator')
    parser.add_argument('--segmentor_finetune_backbone', type=str2bool, default=False, help='If we want to train the miniSegmentor finetuning also the backbone')
    parser.add_argument('--image_key', type=str, default='image',help='Key for image in the data dict')
    parser.add_argument('--mask_key', type=str, default='mask', help='Key for mask in the data dict')
    parser.add_argument('--split_path', type=str, default=os.path.join(PROJECT_DIR, 'data', 'split.json'), help='Path to the JSON split file')
    parser.add_argument('--n_fold', type=int, default=0, help='Fold index')
    parser.add_argument('--train_batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('-nW_eval', '--nW_eval', type=int, default=0, help='Number of workers for val loader')
    parser.add_argument('--save_embeddings', type=str2bool, default=True, help='Save embeddings') 
    parser.add_argument('--save_tensors', type=str2bool, default=False, help='If save the tensors of imgs, prediciton and gts of all the samples during test')
    parser.add_argument('--save_test_images', type=str2bool, default=True, help='Save images')

    
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
    parser.add_argument('--include_background', type=str2bool, default=False, help='Include background in loss')
    parser.add_argument('--acc_steps', type=int, default=4, help='Number of accumulation steps')
    parser.add_argument('--cache_rate', type=float, default=.7, help='Cache rate')

    parser.add_argument('--threshold_discretize', type=float, default=0.01, help='Threshold for discretization')
    
    
    parser.add_argument('--log_histograms', type=str2bool, default=False, help='Log histograms to wandb')
    
    parser.add_argument('--remove_maxpool', type=str2bool, default=False, help='Remove first maxpool when ResNet is used as embedding model')

    parser.add_argument('--sam_zeroshot', type=str2bool, default=False, help='SAM zero shot')

    parser.add_argument('--decoder_channels', type=int, nargs='+', default=[], help='Decoder channels for UnetLike segmentor/aggregator')
    args = vars(parser.parse_args())
    print(parser.description)

    print('='*120)
    args['experiment_name']= args['exp_name']
    output_root = os.path.join(PROJECT_DIR, 'results')

    saver = Saver(output_folder=output_root, args=args)
    cmd = f"{sys.executable} " + " ".join(shlex.quote(arg) for arg in sys.argv)
    saver.log_cmd(cmd)

    args['vis_folder'] = saver.vis_dir

    if args['use_sam']:
        sam_ckpt = os.path.join(PROJECT_DIR, 'cp', f"{args['sam_ckpt']}.pth")
        if not os.path.exists(sam_ckpt): raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt}\n Please download {args['sam_ckpt']}.pth into the cp/ folder.")
        sam_args = {'sam_checkpoint': sam_ckpt, 'model_type': args['sam_version']}
        args['sam_args'] = sam_args
        print(f"[TEST] SAM checkpoint path: {sam_ckpt}")
        print(f"[TEST] SAM Model type : {args['sam_version']}", '\n')
    else:
        sam_args = None
        args['sam_args'] = None
        print("[TEST] SAM is disabled (aggregator-only testing).", '\n')  

    saver.log_hparams(args)
    main(args=args, sam_args=sam_args, saver=saver)
    print('='*120)
