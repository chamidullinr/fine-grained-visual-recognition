import os
from typing import Iterable

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import PIL

from src.utils import visualization as viz
from .bbox import BBox
from .transforms import SquareCrop


__all__ = ['ClassificationBaseDataset', 'ImageDataset', 'ImageCroppingDataset', 'get_dataloader']


def truncate_str(x, maxlen=20):
    return f'{x[:maxlen].strip()}...' if len(x) > maxlen else x


class ClassificationBaseDataset(Dataset):
    def __init__(self, df, label_col, labels=None, encode_labels=True):
        assert label_col in df
        self.df = df
        self.label_col = label_col

        # set labels and ids
        self.labels = None
        if labels is not None:
            self.labels = labels
        elif (np.all([isinstance(x, str) for x in df[label_col]]) or
             pd.api.types.is_integer_dtype(df[label_col])):
            self.labels = np.unique(df[label_col])

        if encode_labels and self.labels is not None:
            self._label2id = {label: i for i, label in enumerate(self.labels)}
            self._id2label = {i: label for label, i in self._label2id.items()}
            self.ids = list(self._label2id.values())
        else:
            self._label2id = None
            self._id2label = None
            self.ids = self.labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raise NotImplementedError

    def label2id(self, x):
        if self._label2id is None:
            raise ValueError('The dataset does not encode labels.')

        if isinstance(x, Iterable) and not isinstance(x, str):
            out = np.array([self._label2id[itm] for itm in x])
        else:
            out = self._label2id[x]
        return out

    def id2label(self, x):
        if self._id2label is None:
            raise ValueError('The dataset does not encode labels.')

        if isinstance(x, Iterable) and not isinstance(x, str):
            out = np.array([self._id2label[itm] for itm in x])
        else:
            out = self._id2label[x]
        return out


class ImageDataset(ClassificationBaseDataset):
    def __init__(self, df, img_path_col, label_col, transforms=None,
                 path='.', labels=None, encode_labels=True):
        assert img_path_col in df
        super().__init__(df, label_col, labels, encode_labels)
        self.df = df
        self.img_path_col = img_path_col
        self.path = path
        self.transforms = transforms

    def __getitem__(self, idx):
        img, label = self.get_item(idx)
        img = self.transform_image(img)
        if self._label2id:
            label = self._label2id[label]
        return img, label
        
    def transform_image(self, img: PIL.Image, ignore=[]):
        if self.transforms is not None:
            for t in self.transforms.transforms:
                if not isinstance(t, tuple(ignore)):
                    img = t(img)
        return img

    def get_item(self, idx):
        _file_path = self.df[self.img_path_col].values[idx]
        if _file_path[0] == '/':
            _file_path = _file_path[1:]
        file_path = os.path.join(self.path, _file_path)
        img = PIL.Image.open(file_path).convert('RGB')
        label = self.df[self.label_col].values[idx]
        return img, label

    def show_item(self, idx=0, ax=None, apply_transforms=False):
        img, label = self.get_item(idx)
        if apply_transforms:
            img = self.transform_image(img, ignore=[T.Normalize])
            img = TF.to_pil_image(img)
        else:
            img = img.resize((224, 224), resample=PIL.Image.BILINEAR)
        img = np.array(img)
        ax = viz.imshow(
            img, title=truncate_str(f'{label}', maxlen=20),
            ax=ax, axis_off=True)
        return ax

    def show_items(self, idx=None, apply_transforms=False, *,
                   ncols=3, nrows=3, colsize=3, rowsize=3, **kwargs):
        if isinstance(idx, Iterable):
            params = dict(ntotal=len(idx))
        else:
            params = dict(nrows=nrows)
        fig, axs = viz.create_fig(ncols=ncols, colsize=colsize, rowsize=rowsize, **params)
        for i, ax in enumerate(axs):
            if idx is None:
                _idx = np.random.randint(len(self))
            elif isinstance(idx, Iterable):
                if i >= len(idx):
                    break
                _idx = idx[i]
            else:
                _idx = i + idx
            self.show_item(_idx, ax=ax, apply_transforms=apply_transforms, **kwargs)


def _find_transforms(transforms: T.Compose, pat: str, inverse=False):
    import re

    def contains(pat, x, inverse):
        out = re.search(pat, x, flags=re.IGNORECASE)
        if inverse:
            out = not out
        return out

    return [x for x in transforms.transforms
            if contains(pat, x.__class__.__name__, inverse)]


class ImageCroppingDataset(ImageDataset):
    def __init__(self, df, img_path_col, label_col, bbox_col,
                 transforms, path='.', labels=None, encode_labels=True, crop_p=1.0):
        assert bbox_col in df

        # create crop transform
        resize_tfms = _find_transforms(transforms, 'resize')
        assert len(resize_tfms) > 0, 'Transforms do not contain Resize method.'
        self.size = resize_tfms[-1].size
        self.crop_tfm = SquareCrop(self.size, p=crop_p)

        # remove cropping and resize transforms
        transforms = T.Compose(_find_transforms(transforms, 'crop|resize', inverse=True))

        super().__init__(df, img_path_col, label_col, transforms,
                         path, labels, encode_labels)
        self.bbox_col = bbox_col

    def get_item(self, idx, *, crop=True, return_bbox=False):
        img, label = super().get_item(idx)
        bbox = BBox(*self.df[self.bbox_col].iloc[idx])
        if np.array(bbox).max() <= 1:
            w, h = img.size
            bbox = bbox.denormalize(h, w)
        if crop:
            img = self.crop_tfm(img, bbox)
        return (img, label, bbox) if return_bbox else (img, label)

    def show_item(self, idx=0, ax=None, apply_transforms=False, square_image=True):
        if apply_transforms:
            img, label = self.get_item(idx, crop=True)
            img = self.transform_image(img, ignore=[T.Normalize])
            img = TF.to_pil_image(img)
        else:
            img, label, bbox = self.get_item(idx, crop=False, return_bbox=True)
            w, h = img.size

            # create square bbox
            square_bbox = bbox.make_square(h, w)
            square_bbox = square_bbox.make_min_size(224, 224, h, w)

            # adjust bbox to 224 x 224 image dimenstions and resize image
            if square_image:
                bbox = bbox.normalize(h, w)
                bbox = bbox.denormalize(224, 224)
                square_bbox = square_bbox.normalize(h, w)
                square_bbox = square_bbox.denormalize(224, 224)
                img = img.resize((224, 224), resample=PIL.Image.BILINEAR)
        img = np.array(img)
        ax = viz.imshow(img, title=truncate_str(f'{label}', maxlen=20), ax=ax, axis_off=True)
        if not apply_transforms:
            viz.plot_bbox(bbox, ax=ax)
            viz.plot_bbox(square_bbox, ax=ax, linestyle=':')
        return ax


def get_dataloader(df, img_path_col, label_col, path='.', transforms=None,
                   batch_size=32, shuffle=True, num_workers=4, sampler=None,
                   bbox_col=None, labels=None, encode_labels=True, **kwargs):
    if bbox_col is None:
        dataset = ImageDataset(
            df, img_path_col, label_col, path=path, transforms=transforms,
            labels=labels, encode_labels=encode_labels)
    else:
        dataset = ImageCroppingDataset(
            df, img_path_col, label_col, path=path, bbox_col=bbox_col,
            transforms=transforms, labels=labels, encode_labels=encode_labels)
    if callable(sampler):
        sampler = sampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, sampler=sampler, **kwargs)
    return dataloader
