import numpy as np
from tqdm.auto import tqdm
import torch
from utils.utils import norm_batch, get_input_dict
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import AsDiscrete, Activations
from utils.model_utils import sam_call
import torch.nn.functional as F
from torch.amp import autocast

@torch.no_grad()
def inference_dataloader(val_context, epoch):
    val_dataloader = val_context['dataloader']
    model = val_context['model']
    sam = val_context['sam']
    device = val_context['device']
    args = val_context['args']
    saver = val_context['saver']
    use_amp = val_context['use_amp']

    model.eval()
    if sam is not None: sam.eval()

    threshold  = args['threshold_discretize']
    include_bg = args['include_background']
    use_aggregator = args['use_aggregator']
    input_size = args['input_size']
    image_key, mask_key = args['image_key'], args['mask_key']

    apply_sigm = True

    one_hot = AsDiscrete(to_onehot=2, dim=1)     
    post_trans = AsDiscrete(threshold=threshold)         
    sigmoid = Activations(sigmoid=True)

    dice_metric_include_background = DiceMetric(include_background=True, reduction='mean')
    dice_metric_exclude_background = DiceMetric(include_background=False, reduction='mean')
    iou_metric_include_background = MeanIoU(include_background=True, reduction='mean')
    iou_metric_exclude_background = MeanIoU(include_background=False, reduction='mean')

    criterion_name = args['criterion']
    if criterion_name == 'dice': criterion = DiceLoss(include_background=include_bg, sigmoid=apply_sigm)
    elif criterion_name == 'dice_ce': criterion = DiceCELoss(include_background=include_bg, sigmoid=apply_sigm)
    else: raise ValueError(f"Unsupported criterion '{criterion_name}'.")


    dice_list_incl, dice_list_excl = [], []
    iou_list_incl, iou_list_excl = [], []
    loss_list = []

    pbar = tqdm(val_dataloader, desc=f"Val E{epoch}")
    
    for ix,  sample in enumerate(pbar):
        imgs = sample[image_key]
        gts  = sample[mask_key]
        batch_size = imgs.shape[0]
        if batch_size != 1: raise ValueError(f"Val batch_size must be 1, got {imgs.shape[0]}. Check val_dataloader.")
        original_sz_batch = sample['original_shape']
        img_sz = torch.tensor(imgs.shape[2:]).repeat(batch_size, 1)
        orig_imgs = imgs.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)

        with autocast('cuda', enabled=use_amp):
            orig_imgs_small = F.interpolate(orig_imgs, size=(input_size, input_size), mode='bilinear', align_corners=False)
            if not use_aggregator:
                dense_embeddings = model(orig_imgs_small)
                batched_input = get_input_dict(orig_imgs, original_sz_batch, img_sz)
                masks = sam_call(batched_input, sam, dense_embeddings)
            else: masks = model(orig_imgs_small)

            size = masks.shape[2:]
            gts_resized = F.interpolate(gts, size=size, mode='nearest')
            onehot_gts = one_hot(post_trans(gts_resized))
            
            if masks.shape[1] == 1:
                masks_prob = sigmoid(masks)
                onehot_masks = one_hot(post_trans(masks_prob))
            elif masks.shape[1] == 2:
                probs = sigmoid(masks) if apply_sigm else masks
                pred_polyp = post_trans(probs[:, 1:2])
                onehot_masks = torch.cat([1.0 - pred_polyp, pred_polyp], dim=1)
            else: raise ValueError(f"Unexpected mask channels: {masks.shape[1]}")

            
            if (getattr(saver, 'wandb_enabled', False)) and args['save_val_images'] and ix < 5 and epoch % 5 == 0:
                imgs_resized = F.interpolate(orig_imgs, size=size, mode='bilinear', align_corners=False)
                pred_2d = onehot_masks[:, 1].detach()
                gt_2d   = gts_resized.squeeze(1).detach()
                saver.log_samples_2d(imgs_resized.detach(), pred_2d, gt_2d, step=epoch, name=f"batch_{ix}", phase='val')
                
            dice_metric_include_background(onehot_masks, onehot_gts)
            dice_include_background = dice_metric_include_background.aggregate().item()

            dice_metric_exclude_background(onehot_masks, onehot_gts)
            dice_exclude_background = dice_metric_exclude_background.aggregate().item()

            iou_metric_include_background(onehot_masks, onehot_gts)
            iou_include_background = iou_metric_include_background.aggregate().item()

            iou_metric_exclude_background(onehot_masks, onehot_gts)
            iou_exclude_background = iou_metric_exclude_background.aggregate().item()

            dice_list_incl.append(dice_include_background)
            dice_list_excl.append(dice_exclude_background)
            iou_list_incl.append(iou_include_background)
            iou_list_excl.append(iou_exclude_background)

            dice_metric_include_background.reset()
            dice_metric_exclude_background.reset()
            iou_metric_include_background.reset()
            iou_metric_exclude_background.reset()
            
            
            if masks.shape[1] == 1: loss=criterion(masks, gts_resized)
            elif masks.shape[1] == 2: loss=criterion(masks, onehot_gts)
            else: raise ValueError(f"Unexpected mask channels: {masks.shape[1]}")

            loss_list.append(loss.item())
            
        pbar.set_description(f'(val | polyp) E{epoch} | '
                             f'loss: {np.nanmean(loss_list):.4f} | '
                             f'dice_incl: {np.nanmean(dice_list_incl):.4f} | '
                             f'dice_excl: {np.nanmean(dice_list_excl):.4f} | '
                             f'iou_incl: {np.nanmean(iou_list_incl):.4f} | '
                             f'iou_excl: {np.nanmean(iou_list_excl):.4f}')
        
    return {
    'loss':                    float(np.nanmean(loss_list)),
    'dice_exclude_background': float(np.nanmean(dice_list_excl)),
    'dice_include_background': float(np.nanmean(dice_list_incl)),
    'iou_exclude_background':  float(np.nanmean(iou_list_excl)),
    'iou_include_background':  float(np.nanmean(iou_list_incl)),}