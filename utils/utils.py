import torch.nn as nn
import argparse
import random
import numpy as np
import torch
from monai.utils import set_determinism

 
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

def safe_mean(lst):
    arr = np.asarray(lst, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if len(finite) > 0 else float('nan')
