import argparse
import random
import numpy as np
import torch




from monai.utils import set_determinism

import os
import torch.nn.functional as F

import torch.nn as nn
 
def disable_batchnorm_running_stats(model):
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            count += 1       
    print(f"[disable_batchnorm_running_stats] Disabled BN running stats on {count} layer(s) in {type(model).__name__}")

def str2bool(v):
    if isinstance(v, bool): return v
    if isinstance(v, int): return bool(v)
    if isinstance(v, str):
        v_lower = v.strip().lower()
        if v_lower in ('yes', 'true', 't', 'y', '1', 'on'): return True
        elif v_lower in ('no', 'false', 'f', 'n', '0', 'off'): return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {v!r} of type {type(v).__name__}')

def set_seed(seed, strict=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed, use_deterministic_algorithms=strict)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try: torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError: torch.use_deterministic_algorithms(True)
        print(f"[set_seed] Strict determinism enabled (seed={seed}). Training may be slower.\n") 
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try: torch.use_deterministic_algorithms(False)  
        except Exception as e: print(f"[set_seed] Warning: could not disable deterministic algorithms: {e}")
        print(f"[set_seed] Fast non-deterministic mode (seed={seed}).\n")           

def get_input_dict(imgs, original_sz, img_sz, point_coords=None, point_labels=None, boxes=None, mask_inputs=None):
    batched_input = []
    for i, img in enumerate(imgs):
        input_size    = tuple(int(x) for x in img_sz[i].squeeze().tolist())
        original_size = tuple(int(x) for x in original_sz[i].squeeze().tolist())
        single_input = {'image': img, 'original_size': original_size, 'image_size': input_size}

        if point_coords is not None: single_input['point_coords'] = point_coords
        if point_labels is not None: single_input['point_labels'] = point_labels
        if boxes is not None: single_input['boxes'] = boxes
        if mask_inputs is not None: single_input['mask_inputs'] = mask_inputs

        batched_input.append(single_input)

    return batched_input

def norm_batch(x, mode='minmax'):
    if mode == 'minmax':
        B = x.shape[0]
        flat = x.view(B, -1)
        min_v = flat.min(dim=1)[0].view(B, 1, 1, 1)
        max_v = flat.max(dim=1)[0].view(B, 1, 1, 1)
        return (x - min_v) / (max_v - min_v + 1e-6)
    elif mode == 'sigmoid': return torch.sigmoid(x)
    else: raise ValueError(f"Unsupported mode '{mode}'. Choose from: 'minmax', 'sigmoid'.")


def Dice_loss(y_true, y_pred, smooth=1):
    alpha = 0.5
    beta = 0.5
    tp = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    fn = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3))
    fp = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3))
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - torch.mean(tversky_class)


def get_dice_ji(predict, target):
    predict = predict + 1
    target = target + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji


def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return str(len(a))


def gen_step(optimizer, gts, masks, criterion, accumulation_steps, step):
    size = masks.shape[2:]
    gts_sized = F.interpolate(gts.unsqueeze(dim=1), size, mode='nearest')
    loss = criterion(masks, gts_sized) + Dice_loss(masks, gts_sized)
    loss.backward()
    if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()

def get_input_dict_OLD(imgs, original_sz, img_sz):
    batched_input = []
    for i, img in enumerate(imgs):
        input_size = tuple([int(x) for x in img_sz[i].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[i].squeeze().tolist()])
        singel_input = {
            'image': img,
            'original_size': original_size,
            'image_size': input_size, 
            'point_coords': None,
            'point_labels': None,
        }
        batched_input.append(singel_input)
    return batched_input

def postprocess_masks(masks_dict):
    masks = torch.zeros((len(masks_dict), *masks_dict[0]['low_res_logits'].squeeze().shape)).unsqueeze(dim=1).cuda()
    ious = torch.zeros(len(masks_dict)).cuda()
    for i in range(len(masks_dict)):
        cur_mask = masks_dict[i]['low_res_logits'].squeeze()
        cur_mask = (cur_mask - cur_mask.min()) / (cur_mask.max() - cur_mask.min())
        masks[i, 0] = cur_mask.squeeze()
        ious[i] = masks_dict[i]['iou_predictions'].squeeze()
    return masks, ious


