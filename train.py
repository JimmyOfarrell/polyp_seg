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
    input('Press Enter to start training...')

    for ix,  sample in enumerate(pbar):      
        imgs = sample[image_key]
        gts = sample[mask_key]
        original_sz = sample['original_shape']

        batch_size = imgs.shape[0]
        img_sz =torch.tensor(imgs.shape[2:]).repeat(batch_size, 1)

        orig_imgs = imgs.to(args['device'], non_blocking=True)
        gts = gts.to(args['device'], non_blocking=True)
        
        with autocast('cuda', enabled=use_amp):
            if sam is not None:
                orig_imgs_small = F.interpolate(orig_imgs, size=(input_size, input_size), mode='bilinear', align_corners=False)
                dense_embeddings = model(orig_imgs_small)
                batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
                
                masks = norm_batch(sam_call(batched_input, sam, dense_embeddings), mode=norm_mode)
            else: masks = model(orig_imgs)

            size = masks.shape[2:]
            gts_resized = F.interpolate(gts, size=size, mode='nearest')

            onehot_gts=one_hot(post_trans(gts_resized))

            if masks.shape[1] == 1: onehot_masks = one_hot(post_trans(masks))
            elif masks.shape[1] == 2:
                probs = sigmoid(masks) if apply_sigm else masks
                pred_polyp = post_trans(probs[:, 1:2])
                onehot_masks = torch.cat([1.0 - pred_polyp, pred_polyp], dim=1)
            else: raise ValueError(f"Unexpected mask channels: {masks.shape[1]}. Expected 1 or 2.")    
            
            if getattr(saver, 'wandb_enabled', False) and args.get('save_train_images', False) and ix < 10 and epoch % 5 == 0:
                imgs_resized = F.interpolate(orig_imgs, size=size, mode='bilinear', align_corners=False)
                pred_2d = onehot_masks[:, 1].detach()
                gt_2d   = gts_resized.squeeze(1).detach()
                saver.log_samples_2d(imgs_resized.detach(), pred_2d, gt_2d, step=epoch, name=f"batch{ix}", phase='train')

            
            diceMetric(onehot_masks, onehot_gts)
            iouMetric(onehot_masks, onehot_gts)
            dice = diceMetric.aggregate().item()
            iou = iouMetric.aggregate().item()

            if masks.shape[1] == 1: loss = criterion(masks, gts_resized)
            elif masks.shape[1] == 2: loss = criterion(masks, onehot_gts)
            else: raise ValueError(f"Unexpected mask channels: {masks.shape[1]}")
            
            loss_list.append(loss.item())
            iou_list.append(iou)
            dice_list.append(dice)

            accum_loss = loss / acc_steps
        
        scaler.scale(accum_loss).backward()

        if ((ix + 1) % acc_steps == 0) or ((ix + 1) == total_batches):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if use_scheduler:
                if args.get('scheduler') == 'ReduceLROnPlateau': scheduler.step(loss.item())
                else: scheduler.step()
        
        pbar.set_description(f'(train | polyp) epoch {epoch} | (loss: {np.mean(loss_list):.4f}) | (dice: {np.mean(dice_list):.4f}) | (iou: {np.mean(iou_list):.4f})')
        diceMetric.reset()
        iouMetric.reset()

    if use_scheduler: saver.log_loss('lr_end', optimizer.param_groups[0]['lr'], epoch)

    return np.mean(loss_list), np.mean(dice_list), np.mean(iou_list)