import copy
import torch.nn.functional as F
import torch
from monai.transforms import (
    MapTransform,
    Compose,
    LoadImageD,
    EnsureChannelFirstD,
    ScaleIntensityRangeD,
    NormalizeIntensityD,
    ResizeWithPadOrCropd,
    RandFlipD,
    RandRotate90D,
    EnsureTyped)

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

def get_polyp_transforms(args, sam_image_dim = None, get_bbox=False):
    KEYS = [args['image_key'], args['mask_key']]
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transforms = Compose([
        LoadImageD(KEYS, reader="PILReader", reverse_indexing=False),
        EnsureChannelFirstD(keys=KEYS),
        SaveOriginalShapeD(image_key=args['image_key']),
        ScaleIntensityRangeD(keys=[args['image_key']], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityD(keys=[args['image_key']], subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
        ResizeWithRatio(image_key=KEYS[0], mask_key=KEYS[1], image_size=sam_image_dim, image_mode="bilinear", mask_mode='nearest', antialias=True),
        ResizeWithPadOrCropd(keys=KEYS, spatial_size=(sam_image_dim, sam_image_dim),mode='constant'),
        RandFlipD(keys=KEYS, prob=0.5, spatial_axis=0),
        RandFlipD(keys=KEYS, prob=0.5, spatial_axis=1),
        RandRotate90D(keys=KEYS, prob=0.5, spatial_axes=(0, 1)),
        EnsureTyped(keys=KEYS)])
        
    val_test_transforms = Compose([
        LoadImageD(KEYS, reader="PILReader", reverse_indexing=False),
        EnsureChannelFirstD(keys=KEYS),
        SaveOriginalShapeD(image_key=args['image_key']),
        ScaleIntensityRangeD(keys=[args['image_key']], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityD(keys=[args['image_key']], subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
        ResizeWithRatio(image_key=KEYS[0], mask_key=KEYS[1], image_size=sam_image_dim, image_mode="bilinear", mask_mode='nearest', antialias=True),
        ResizeWithPadOrCropd(keys=KEYS, spatial_size=(sam_image_dim, sam_image_dim), mode='constant'),
        EnsureTyped(keys=KEYS)
    ])

    return train_transforms, val_test_transforms