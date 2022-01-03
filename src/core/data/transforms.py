import math

import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms as T
import torchvision.transforms.functional as TF

from .bbox import BBox


__all__ = ['get_transforms', 'padcrop', 'SquareCrop']


def get_transforms(*, size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Create basic image transforms for training or validation dataset."""
    train_tfms = T.Compose([
        T.RandomResizedCrop((size, size), scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply(torch.nn.ModuleList([
            T.ColorJitter(brightness=0.2, contrast=0.2)
        ]), p=0.2),  # random brightness contrast
        T.ToTensor(),
        T.Normalize(mean, std)])
    valid_tfms = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean, std)])
    return train_tfms, valid_tfms


def padcrop(img, top, left, height, width, fill=0, padding_mode='constant'):
    """Crop input image and apply padding to keep same width and height."""
    # apply crop
    img = TF.crop(img, top, left, height, width)

    # apply padding if needed
    _, h, w = img.shape
    if h > w:
        padding = [(h - w)//2, 0]  # [left/right, top/bottom]
        img = TF.pad(img, padding, fill, padding_mode)
    elif w > h:
        padding = [0, (w - h)//2]  # [left/right, top/bottom]
        img = TF.pad(img, padding, fill, padding_mode)

    return img


class SquareCrop(nn.Module):
    """Crop square region based on the given bounding box."""
    def __init__(self, size=None, p=1.0):
        super().__init__()
        if size is not None:
            assert isinstance(size, (list, tuple)) and len(size) == 2
        self.size = size
        self.p = p
    
    def crop(self, img, bbox):
        xmin, ymin, xmax, ymax = bbox
        img = TF.crop(img, top=int(ymin), left=int(xmin),
                      height=int(ymax-ymin), width=int(xmax-xmin))
        return img

    def forward(self, img, bbox):
        if not isinstance(bbox, BBox):
            bbox = BBox(*bbox)

        w, h = img.size
        # if bbox is not None and bbox != (0, 0, w, h):
        bbox = bbox.make_square(h, w)
        if self.size is not None:
            bbox = bbox.make_min_size(*self.size, h, w)

        if self.p >= np.random.rand(1):
            img = self.crop(img, bbox)
        if self.size is not None:
            img = TF.resize(img, size=self.size)
        return img


# class PadCrop(Crop):
#     """
#     Crop region based on the given bounding box
#     and include padding to make image square.
#     """
#     def forward(self, img, bbox):
#         bbox_dict = self.get_bbox_dict(bbox)
#         return padcrop(img, **bbox_dict)


# class ResizedCrop(Crop):
#     """Crop and resize to square region based on the given bounding box."""
#     def __init__(self, size):
#         super().__init__()
#         self.size = size

#     def forward(self, img, bbox):
#         bbox_dict = self.get_bbox_dict(bbox)
#         return TF.resized_crop(img, **bbox_dict, size=self.size)
