import copy
import torch.nn.functional as F
import torch
import cv2
import numpy as np
import random
from monai.transforms import (
    MapTransform, Compose, LoadImageD, EnsureChannelFirstD, ScaleIntensityRangeD,
    NormalizeIntensityD, ResizeWithPadOrCropd, RandFlipD, RandRotate90D, EnsureTyped,
    RandRotated, RandZoomd, Rand2DElasticd, RandGaussianSmoothd,
    RandScaleIntensityd)

class ResizeWithRatio(MapTransform):
    def __init__(self, image_key, mask_key, image_size, image_mode="bilinear", mask_mode="nearest", antialias=False):
        super().__init__(keys=[image_key, mask_key])
        if image_mode not in ("bilinear", "bicubic", "area", "nearest"): raise ValueError(f"Unsupported image_mode '{image_mode}'. Choose from 'bilinear', 'bicubic', 'area', 'nearest'.")
        if mask_mode != "nearest": raise ValueError(f"Unsupported mask_mode '{mask_mode}'. Use 'nearest' to preserve binary values.")

        self.image_key = image_key
        self.mask_key = mask_key
        self.image_size = int(image_size)
        self.image_mode = image_mode
        self.mask_mode = mask_mode
        self.antialias = bool(antialias)
        self._image_align_corners = (None if image_mode in ("nearest", "area") else False)

    def __call__(self, data):
        if self.image_key not in data: raise KeyError(f"Image key '{self.image_key}' not found in the input sample. Available keys: {list(data.keys())}")
        if self.mask_key not in data: raise KeyError(f"Mask key '{self.mask_key}' not found in the input sample. Available keys: {list(data.keys())}")

        d = copy.deepcopy(data)
        image = d[self.image_key]
        mask = d[self.mask_key]

        if image.ndim != 3:raise ValueError(f"Expected 3D tensor (C, H, W) for image, got shape {tuple(image.shape)}")
        if mask.ndim != 3: raise ValueError(f"Expected 3D tensor (C, H, W) for mask, got shape {tuple(mask.shape)}")

        h, w = int(image.shape[-2]), int(image.shape[-1])
        if h == w: new_h, new_w = self.image_size, self.image_size
        elif h > w:
            new_h = self.image_size
            new_w = max(1, round(w * (self.image_size / h)))
        else:
            new_w = self.image_size
            new_h = max(1, round(h * (self.image_size / w)))    
        
        if not image.is_floating_point(): image = image.float()
        if not mask.is_floating_point(): mask = mask.float()

        img_4d = image.unsqueeze(0)
        mask_4d = mask.unsqueeze(0)

        img_kwargs = {"size": (new_h, new_w), "mode": self.image_mode}
        if self.image_mode in ("bilinear", "bicubic"):
            img_kwargs["align_corners"] = self._image_align_corners
            if self.antialias: img_kwargs["antialias"] = True
        resized_img = F.interpolate(img_4d, **img_kwargs)

        mask_kwargs = {"size": (new_h, new_w), "mode": self.mask_mode}
        resized_mask = F.interpolate(mask_4d, **mask_kwargs)


        d[self.image_key] = resized_img.squeeze(0)
        d[self.mask_key] = resized_mask.squeeze(0)

        return d

class SaveOriginalShapeD(MapTransform):
    def __init__(self, image_key, out_key='original_shape'):
        super().__init__(keys=[image_key])
        self.image_key = image_key
        self.out_key = out_key

    def __call__(self, data):
        d = dict(data)
        img = d[self.image_key]                       
        d[self.out_key] = torch.tensor(img.shape[-2:], dtype=torch.int64)
        return d

class CLAHETransformd(MapTransform):
    def __init__(self, keys, clip_limit=4.0, grid_size=(8, 8), p=0.3):
        super().__init__(keys)
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        self.p = p
        self._do_transform = False

    def randomize(self):
        self._do_transform = random.random() < self.p

    def _apply_clahe(self, img_np):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def __call__(self, data):
        self.randomize()
        d = dict(data)
        if not self._do_transform: return d
        for key in self.keys:
            img = d[key]
            if isinstance(img, torch.Tensor): img_np = img.permute(1, 2, 0).numpy()
            else: img_np = img
            result = self._apply_clahe(img_np)
            if isinstance(img, torch.Tensor): d[key] = torch.from_numpy(result).permute(2, 0, 1)
            else: d[key] = result
        return d

class HSVTransformd(MapTransform):
    def __init__(self, keys, hue_shift=20, sat_shift=30, val_shift=0, p=0.3):
        super().__init__(keys)
        self.hue_shift = hue_shift
        self.sat_shift = sat_shift
        self.val_shift = val_shift
        self.p = p
        self._do_transform = False
        self._hue = 0
        self._sat = 0
        self._val = 0

    def randomize(self):
        self._do_transform = random.random() < self.p
        if self._do_transform:
            self._hue = random.uniform(-self.hue_shift, self.hue_shift)
            self._sat = random.uniform(-self.sat_shift, self.sat_shift)
            self._val = random.uniform(-self.val_shift, self.val_shift)

    def _apply_hsv(self, img_np):
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + self._hue / 2) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + self._sat, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + self._val, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def __call__(self, data):
        self.randomize()
        d = dict(data)
        if not self._do_transform: return d
        for key in self.keys:
            img = d[key]
            if isinstance(img, torch.Tensor): img_np = img.permute(1, 2, 0).numpy()
            else: img_np = img
            result = self._apply_hsv(img_np)
            if isinstance(img, torch.Tensor): d[key] = torch.from_numpy(result).permute(2, 0, 1)
            else: d[key] = result
        return d

class UnsharpMaskTransformd(MapTransform):
    def __init__(self, keys, sigma=1.0, strength=0.5, p=0.2):
        super().__init__(keys)
        self.sigma = sigma
        self.strength = strength
        self.p = p
        self._do_transform = False

    def randomize(self):
        self._do_transform = random.random() < self.p

    def _apply_unsharp(self, img_np):
        blurred = cv2.GaussianBlur(img_np, (0, 0), self.sigma)
        result = cv2.addWeighted(img_np, 1.0 + self.strength, blurred, -self.strength, 0)
        return np.clip(result, 0, 255).astype(np.uint8)

    def __call__(self, data):
        self.randomize()
        d = dict(data)
        if not self._do_transform: return d
        for key in self.keys:
            img = d[key]
            if isinstance(img, torch.Tensor): img_np = img.permute(1, 2, 0).numpy()
            else: img_np = img
            result = self._apply_unsharp(img_np)
            if isinstance(img, torch.Tensor): d[key] = torch.from_numpy(result).permute(2, 0, 1)
            else: d[key] = result
        return d

def get_polyp_transforms(args, sam_image_dim = None, get_bbox=False):
    KEYS = [args['image_key'], args['mask_key']]
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transforms = Compose([
        LoadImageD(KEYS, reader="PILReader", reverse_indexing=False),
        EnsureChannelFirstD(keys=KEYS),
        SaveOriginalShapeD(image_key=args['image_key']),
        CLAHETransformd(keys=[args['image_key']], clip_limit=4.0, grid_size=(8, 8), p=0.3),
        HSVTransformd(keys=[args['image_key']], hue_shift=20, sat_shift=30, val_shift=0, p=0.3),
        UnsharpMaskTransformd(keys=[args['image_key']], sigma=1.0, strength=0.3, p=0.2),
        ScaleIntensityRangeD(keys=[args['image_key']], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        RandScaleIntensityd(keys=[args['image_key']], factors=0.2, channel_wise=True, prob=0.3),
        NormalizeIntensityD(keys=[args['image_key']], subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
        ResizeWithRatio(image_key=KEYS[0], mask_key=KEYS[1], image_size=sam_image_dim, image_mode="bilinear", mask_mode='nearest', antialias=True),
        ResizeWithPadOrCropd(keys=KEYS, spatial_size=(sam_image_dim, sam_image_dim), mode='constant'),
        RandFlipD(keys=KEYS, prob=0.5, spatial_axis=0),
        RandFlipD(keys=KEYS, prob=0.5, spatial_axis=1),
        RandRotate90D(keys=KEYS, prob=0.5, spatial_axes=(0, 1)),
        RandRotated(keys=KEYS, range_x=0.44, mode=('bilinear', 'nearest'), padding_mode='zeros', keep_size=True, prob=0.4),
        RandZoomd(keys=KEYS, min_zoom=0.9, max_zoom=1.1, mode=('bilinear', 'nearest'), padding_mode='constant', prob=0.3),
        Rand2DElasticd(keys=KEYS, sigma_range=(5, 8), magnitude_range=(20, 40), mode=('bilinear', 'nearest'), padding_mode='zeros', spatial_size=(sam_image_dim, sam_image_dim), prob=0.3),
        RandGaussianSmoothd(keys=[args['image_key']], sigma_x=(0.25, 1.0), sigma_y=(0.25, 1.0), sigma_z=0, prob=0.2),
        EnsureTyped(keys=KEYS)])

    val_test_transforms = Compose([
        LoadImageD(KEYS, reader="PILReader", reverse_indexing=False),
        EnsureChannelFirstD(keys=KEYS),
        SaveOriginalShapeD(image_key=args['image_key']),
        ScaleIntensityRangeD(keys=[args['image_key']], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityD(keys=[args['image_key']], subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
        ResizeWithRatio(image_key=KEYS[0], mask_key=KEYS[1], image_size=sam_image_dim, image_mode="bilinear", mask_mode='nearest', antialias=True),
        ResizeWithPadOrCropd(keys=KEYS, spatial_size=(sam_image_dim, sam_image_dim), mode='constant'),
        EnsureTyped(keys=KEYS)])

    return train_transforms, val_test_transforms