import numpy as np
from tqdm import tqdm
import torch
from utils.utils import norm_batch, get_input_dict
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import AsDiscrete, Activations
from utils.model_utils import sam_call
import torch.nn.functional as F
from torch.amp import autocast

@torch.no_grad()
def inference_dataloader(val_dataloader, model, sam, epoch, args, saver):
    print('\n' + '=' * 60)
    print(f"[INFERENCE] Validation | Epoch {epoch}")
    print('=' * 60)

    pbar = tqdm(val_dataloader, desc=f"Val E{epoch}")

    model.eval()
    if sam is not None: sam.eval()

    threshold  = args.get('threshold_discretize', 0.5)
    include_bg = args.get('include_background', False)
    use_std_net = args.get('use_standard_net', False)
    net_name = args.get('net', 'HardNetSegmentor')
    Idim = int(args.get('Idim', 512))
    norm_mode = args.get('norm_mode', 'minmax')
    image_key = args.get('image_key', 'image')
    mask_key = args.get('mask_key', 'mask')
    use_amp = args.get('use_cuda_amp', False) and torch.cuda.is_available()

    one_hot     = AsDiscrete(to_onehot=2, dim=1)     
    post_trans  = AsDiscrete(threshold=threshold)         
    sigmoid     = Activations(sigmoid=True)

    diceMetric_true  = DiceMetric(include_background=True,  reduction='mean')
    diceMetric_false = DiceMetric(include_background=False, reduction='mean')
    iouMetric = MeanIoU(include_background=include_bg, reduction='mean')
    
    apply_sigm = bool(getattr(model, "is_segmentor", False) and use_std_net)

    criterion_name = args.get('criterion', 'dice_ce')
    if criterion_name == 'dice': criterion = DiceLoss(include_background=include_bg, sigmoid=apply_sigm)
    elif criterion_name == 'dice_ce': criterion = DiceCELoss(include_background=include_bg, sigmoid=apply_sigm)
    else: raise ValueError(f"Unsupported criterion '{criterion_name}'.")

    if epoch == 0: print(f"[INFERENCE] Config | net={net_name} | apply_sigm={apply_sigm} | threshold={threshold} | include_bg={include_bg}")

    iou_list, dice_list_true, dice_list_false, loss_list = [], [], [], []
    
    for ix,  sample in enumerate(pbar):
        imgs = sample[image_key]
        gts  = sample[mask_key]
        batch_size = imgs.shape[0]
        if batch_size != 1: raise ValueError(f"Val batch_size must be 1, got {imgs.shape[0]}. Check val_dataloader.")
        original_sz_batch = sample['original_shape']
        img_sz = torch.tensor(imgs.shape[2:]).repeat(batch_size, 1)
        orig_imgs = imgs.to(args['device'], non_blocking=True)
        gts       = gts.to(args['device'], non_blocking=True)

        with autocast('cuda', enabled=use_amp):
            if sam is not None:
                orig_imgs_small = F.interpolate(orig_imgs, size=(Idim, Idim), mode='bilinear', align_corners=False)
                dense_embeddings = model(orig_imgs_small)
                batched_input = get_input_dict(orig_imgs, original_sz_batch, img_sz)
                masks = norm_batch(sam_call(batched_input, sam, dense_embeddings), mode=norm_mode)
            else: masks = model(orig_imgs)
            size = masks.shape[2:]
            gts_resized = F.interpolate(gts, size=size, mode='nearest')

            onehot_gts = one_hot(post_trans(gts_resized))
            if masks.shape[1] == 1: onehot_masks = one_hot(post_trans(masks))
            elif masks.shape[1] == 2:
                probs = sigmoid(masks) if apply_sigm else masks
                pred_polyp = post_trans(probs[:, 1:2])
                onehot_masks = torch.cat([1.0 - pred_polyp, pred_polyp], dim=1)
            else: raise ValueError(f"Unexpected mask channels: {masks.shape[1]}")

            
            if (getattr(saver, 'wandb_enabled', False)) and args.get('save_val_images', False) and ix < 5 and epoch % 5 == 0:
                imgs_resized = F.interpolate(orig_imgs, size=size, mode='bilinear', align_corners=False)
                pred_2d = onehot_masks[:, 1].detach()
                gt_2d   = gts_resized.squeeze(1).detach()
                saver.log_samples_2d(imgs_resized.detach(), pred_2d, gt_2d, step=epoch, name=f"sample{ix}", phase='val')
                
            diceMetric_true(onehot_masks, onehot_gts)
            dice_true = diceMetric_true.aggregate().item()

            diceMetric_false(onehot_masks, onehot_gts)
            dice_false = diceMetric_false.aggregate().item()

            iouMetric(onehot_masks, onehot_gts)
            iou = iouMetric.aggregate().item()

            iou_list.append(iou)
            dice_list_true.append(dice_true)
            dice_list_false.append(dice_false)
            
            
            if masks.shape[1] == 1: loss=criterion(masks, gts_resized)
            elif masks.shape[1] == 2: loss=criterion(masks, onehot_gts)
            else: raise ValueError(f"Unexpected mask channels: {masks.shape[1]}")

            loss_list.append(loss.item())
            
        pbar.set_description(f'(val | polyp) epoch {epoch} :: loss {np.mean(loss_list):.4f} :: DiceT {np.mean(dice_list_true):.4f} :: DiceF {np.mean(dice_list_false):.4f} :: IoU {np.mean(iou_list):.4f}')
        
        iouMetric.reset()
        diceMetric_true.reset()
        diceMetric_false.reset()
        
    return np.mean(dice_list_true),np.mean(dice_list_false), np.mean(iou_list), np.mean(loss_list)