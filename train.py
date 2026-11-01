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

def train_single_epoch(train_context, epoch):
    train_dataloader = train_context['dataloader']
    model = train_context['model']
    sam = train_context['sam']
    optimizer = train_context['optimizer']
    args = train_context['args']
    saver = train_context['saver']
    scheduler = train_context['scheduler']
    scaler = train_context['scaler']
    use_amp = train_context['use_amp']
    device = train_context['device']

    model.train()
    if sam is not None: sam.eval()

    threshold = args['threshold_discretize']
    include_bg = args['include_background']
    use_aggregator = args['use_aggregator']
    loss_type = args['loss_type']
    distill_alpha = args['distill_alpha']
    norm_mode = args['norm_mode']
    input_size = args['input_size']
    use_scheduler = args['use_scheduler']
    image_key = args['image_key']
    mask_key = args['mask_key']
    acc_steps = args['acc_steps']

    apply_sigm = bool(getattr(model, "is_segmentor", False) and use_aggregator)

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
    else: raise ValueError(f"Unsupported criterion '{criterion_name}'. Choose from: 'dice', 'dice_ce'.")

    total_batches = len(train_dataloader)
    dice_list_incl, dice_list_excl = [], []
    iou_list_incl, iou_list_excl = [], []
    loss_list = []

    pbar = tqdm(train_dataloader, desc=f"Train E{epoch}")
    
    if use_scheduler : saver.log_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch, log_to_csv=False)
    optimizer.zero_grad()

    for ix,  sample in enumerate(pbar):      
        imgs = sample[image_key]
        gts = sample[mask_key]
        original_sz = sample['original_shape']
        batch_size = imgs.shape[0]
        img_sz =torch.tensor(imgs.shape[2:]).repeat(batch_size, 1)
        orig_imgs = imgs.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)
        
        with autocast('cuda', enabled=use_amp):
            if sam is not None and not use_aggregator:
                orig_imgs_small = F.interpolate(orig_imgs, size=(input_size, input_size), mode='bilinear', align_corners=False)
                dense_embeddings = model(orig_imgs_small)
                batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
                masks = norm_batch(sam_call(batched_input, sam, dense_embeddings), mode=norm_mode)
                sam_target = None
            elif sam is not None and use_aggregator:
                orig_imgs_small = F.interpolate(orig_imgs, size=(input_size, input_size), mode='bilinear', align_corners=False)
                masks, dec = model(orig_imgs_small, return_features=True)
                with torch.no_grad():
                    dense_embeddings = F.interpolate(dec, size=(64, 64), mode='bilinear', align_corners=False) 
                    batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
                    sam_logits = sam_call(batched_input, sam, dense_embeddings)
                    sam_target = norm_batch(sam_logits, mode=norm_mode)  
            else:
                masks = model(orig_imgs)
                sam_target = None

            size = masks.shape[2:]
            gts_resized = F.interpolate(gts, size=size, mode='nearest')
            onehot_gts=one_hot(post_trans(gts_resized))

            if masks.shape[1] == 1: onehot_masks = one_hot(post_trans(masks))
            elif masks.shape[1] == 2:
                probs = sigmoid(masks) if apply_sigm else masks
                pred_polyp = post_trans(probs[:, 1:2])
                onehot_masks = torch.cat([1.0 - pred_polyp, pred_polyp], dim=1)
            else: raise ValueError(f"Unexpected mask channels: {masks.shape[1]}. Expected 1 or 2.")

            sam_onehot = None
            if sam_target is not None:
                sam_target_resized = F.interpolate(sam_target, size=masks.shape[2:], mode='bilinear', align_corners=False)
                sam_binary = post_trans(sam_target_resized)
                sam_onehot = one_hot(sam_binary)    
            
            if getattr(saver, 'wandb_enabled', False) and args.get('save_train_images', False) and ix < 10 and epoch % 5 == 0:
                imgs_resized = F.interpolate(orig_imgs, size=size, mode='bilinear', align_corners=False)
                pred_2d = onehot_masks[:, 1].detach()
                gt_2d   = gts_resized.squeeze(1).detach()
                max_images_to_log = 4
                n_log = min(max_images_to_log, imgs_resized.shape[0])
                saver.log_samples_2d(imgs_resized[:n_log].detach(), pred_2d[:n_log].detach(), gt_2d[:n_log].detach(), step=epoch, name=f"batch_{ix}", phase='train')

            
            dice_metric_include_background(onehot_masks, onehot_gts)
            dice_metric_exclude_background(onehot_masks, onehot_gts)
            dice_incl = dice_metric_include_background.aggregate().item()
            dice_excl = dice_metric_exclude_background.aggregate().item()

            iou_metric_include_background(onehot_masks, onehot_gts)
            iou_metric_exclude_background(onehot_masks, onehot_gts)
            iou_incl = iou_metric_include_background.aggregate().item()
            iou_excl = iou_metric_exclude_background.aggregate().item()

            dice_list_incl.append(dice_incl)
            dice_list_excl.append(dice_excl)
            iou_list_incl.append(iou_incl)
            iou_list_excl.append(iou_excl)

            dice_metric_include_background.reset()
            dice_metric_exclude_background.reset()
            iou_metric_include_background.reset()
            iou_metric_exclude_background.reset()

            if masks.shape[1] == 1: loss = criterion(masks, gts_resized)
            elif masks.shape[1] == 2:
                if sam_onehot is None: loss = criterion(masks, onehot_gts)
                elif loss_type == 'gt': loss = criterion(masks, onehot_gts)
                elif loss_type == 'sam': loss = criterion(masks, sam_onehot)
                elif loss_type == 'mixed':
                    alpha = distill_alpha if distill_alpha is not None else 0.5
                    loss_gt = criterion(masks, onehot_gts)
                    loss_sam = criterion(masks, sam_onehot)
                    loss = alpha * loss_sam + (1 - alpha) * loss_gt
                else: raise ValueError(f"Unsupported loss_type '{loss_type}'. Choose from: 'gt', 'sam', 'mixed'.")
            else: raise ValueError(f"Unexpected mask channels: {masks.shape[1]}")
            
            loss_list.append(loss.item())

            accum_loss = loss / acc_steps
        
        scaler.scale(accum_loss).backward()

        if ((ix + 1) % acc_steps == 0) or ((ix + 1) == total_batches):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if use_scheduler:
                if args.get('scheduler') == 'ReduceLROnPlateau': scheduler.step(loss.item())
                else: scheduler.step()
        
        pbar.set_description(
            f'(train | polyp) E{epoch} | '
            f'loss: {np.nanmean(loss_list):.4f} | '
            f'dice_incl: {np.nanmean(dice_list_incl):.4f} | '
            f'dice_excl: {np.nanmean(dice_list_excl):.4f} | '
            f'iou_incl: {np.nanmean(iou_list_incl):.4f} | '
            f'iou_excl: {np.nanmean(iou_list_excl):.4f}')

    if use_scheduler: saver.log_scalar('learning_rate_end', optimizer.param_groups[0]['lr'], epoch, log_to_csv=False)

    return {
    'loss':                    float(np.nanmean(loss_list)),
    'dice_exclude_background': float(np.nanmean(dice_list_excl)),
    'dice_include_background': float(np.nanmean(dice_list_incl)),
    'iou_exclude_background':  float(np.nanmean(iou_list_excl)),
    'iou_include_background':  float(np.nanmean(iou_list_incl))}