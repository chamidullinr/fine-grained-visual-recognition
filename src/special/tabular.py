from typing import NamedTuple

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import data



"""Dataset and Dataloader."""


# class ImageMetadataTuple(NamedTuple):
#     img: torch.Tensor
#     cat_metadata: torch.Tensor
#     cont_metadata: torch.Tensor

#     def to(self, device):
#         return ImageMetadataTuple(
#             self.img.to(device),
#             self.cat_metadata.to(device),
#             self.cont_metadata.to(device))


# class ImageMetadataDataset(data.ImageDataset):
#     def __init__(self, df, img_path_col, label_col,
#                  cat_metadata_cols, cont_metadata_cols,
#                  transforms, path='.', labels=None, encode_labels=True):
#         if not isinstance(cat_metadata_cols, list):
#             cat_metadata_cols = [cat_metadata_cols]
#         if not isinstance(cont_metadata_cols, list):
#             cont_metadata_cols = [cont_metadata_cols]
#         for col in cat_metadata_cols + cont_metadata_cols:
#             assert col in df
#         df = df.copy()
#         super().__init__(
#             df, img_path_col, label_col, transforms,
#             path, labels, encode_labels)
#         self.cat_metadata_cols = cat_metadata_cols
#         self.cont_metadata_cols = cont_metadata_cols

#         # encode categorical metadata cols
#         self.encoders = {}
#         for col in self.cat_metadata_cols:
#             self.df[col] = self.df[col].fillna('nan')
#             self.encoders[col] = {x: i for i, x in enumerate(np.unique(self.df[col]))}
#             self.df[col + '_enc'] = self.df[col].apply(self.encoders[col].get)

#         # fill missing values in continuous metadata cols
#         for col in self.cont_metadata_cols:
#             self.df[col] = self.df[col].fillna(0)

#     def __getitem__(self, idx):
#         img, label = super().__getitem__(idx)
#         cat_metadata_cols = [x + '_enc' for x in self.cat_metadata_cols]
#         cat_metadata = self.df[cat_metadata_cols].iloc[idx].values.astype(int)
#         cont_metadata = self.df[self.cont_metadata_cols].iloc[idx].values.astype(np.float32)
#         return ImageMetadataTuple(img, cat_metadata, cont_metadata), label


# def get_dataloader(df, img_path_col, label_col, cat_metadata_cols, cont_metadata_cols,
#                    path='.', transforms=None, batch_size=32, shuffle=True, num_workers=4,
#                    sampler=None, labels=None, encode_labels=True, **kwargs):
#     dataset = ImageMetadataDataset(
#         df, img_path_col, label_col, cat_metadata_cols, cont_metadata_cols,
#         path=path, transforms=transforms,
#         labels=labels, encode_labels=encode_labels)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
#                             num_workers=num_workers, sampler=sampler, **kwargs)
#     return dataloader


class MetadataTuple(NamedTuple):
    cat_metadata: torch.Tensor
    cont_metadata: torch.Tensor

    def to(self, device):
        return MetadataTuple(
            self.cat_metadata.to(device),
            self.cont_metadata.to(device))


class MetadataDataset(data.ClassificationBaseDataset):
    def __init__(self, df, cat_cols, cont_cols, label_col,
                 labels=None, encode_labels=True):
        if not isinstance(cat_cols, list):
            cat_cols = [cat_cols]
        if not isinstance(cont_cols, list):
            cont_cols = [cont_cols]
        for col in cat_cols + cont_cols:
            assert col in df
        df = df.copy()
        super().__init__(df, label_col, labels, encode_labels)

        self.cat_cols = cat_cols
        self.cont_cols = cont_cols

        # encode categorical metadata cols
        self.encoders = {}
        for col in self.cat_cols:
            self.df[col] = self.df[col].fillna('nan')
            self.encoders[col] = {x: i for i, x in enumerate(np.unique(self.df[col]))}
            self.df[col + '_enc'] = self.df[col].apply(self.encoders[col].get)

        # fill missing values in continuous metadata cols
        for col in self.cont_cols:
            self.df[col] = self.df[col].fillna(0)

    def __getitem__(self, idx):
        label = self.df[self.label_col].values[idx]
        if self._label2id:
            label = self._label2id[label]
        cat_cols = [x + '_enc' for x in self.cat_cols]
        cat_metadata = self.df[cat_cols].iloc[idx].values.astype(int)
        cont_metadata = self.df[self.cont_cols].iloc[idx].values.astype(np.float32)
        return MetadataTuple(cat_metadata, cont_metadata), label


def get_dataloader(df, cat_cols, cont_cols, label_col,
                   batch_size=32, shuffle=True, num_workers=4,
                   sampler=None, labels=None, encode_labels=True, **kwargs):
    dataset = MetadataDataset(
        df, cat_cols, cont_cols, label_col, labels=labels, encode_labels=encode_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, sampler=sampler, **kwargs)
    return dataloader


"""Tabular Network."""


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.nonlin2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.linear1(x)
        out = self.nonlin1(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.nonlin2(out)
        out = x + out
        return out


class OneHot(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embedding_dim = num_classes

    def forward(self, x):
        return F.one_hot(x, num_classes=self.embedding_dim).to(torch.float32)


class TabularNet(nn.Module):
    def __init__(self, categories, no_continuous, out_dim, *,
                 depth=2, hidden_size=256, use_onehot=False):
        assert len(categories) + no_continuous > 0
        super().__init__()
        self.categories = categories
        self.no_continuous = no_continuous

        # embedding layers with dropout for categorical features
        if use_onehot:
            self.embeddings = nn.ModuleList([OneHot(x) for x in categories])
        else:
            self.embeddings = nn.ModuleList([
                nn.Embedding(x, min(100, (x+1)//2)) for x in categories])  
        self.emb_out = sum(x.embedding_dim for x in self.embeddings)  # length of all embeddings combined
        self.emb_drop = nn.Dropout(0.5)

        # batch normalization for continuous features
        self.bn_cont = nn.BatchNorm1d(self.no_continuous)

        # linear layers
        self.tabular_fc = nn.Sequential(
            nn.Linear(self.emb_out + self.no_continuous, hidden_size),
            nn.ReLU(inplace=True),
            *(ResidualBlock(hidden_size),)*depth
        )

        # classification head
        self.classifier = nn.Linear(hidden_size, out_dim)

    def forward_metadata(self, cat_meta, cont_meta):
        # encode categorical features
        if cat_meta.shape[1] > 0:
            cat_meta = torch.cat([emb(cat_meta[:,i]) for i, emb in enumerate(self.embeddings)], dim=1)
            cat_meta = self.emb_drop(cat_meta)

        # process continuous features
        if cont_meta.shape[1] > 0:
            cont_meta = self.bn_cont(cont_meta)

        # combine features and apply fully connected layers
        if cat_meta.shape[1] > 0 and cont_meta.shape[1] > 0:
            out = torch.cat([cat_meta, cont_meta], 1)
        elif cat_meta.shape[1] > 0:
            out = cat_meta
        elif cont_meta.shape[1] > 0:
            out = cont_meta
        else:
            raise ValueError('Both categorical and continuous metadata are empty.')
        out = self.tabular_fc(out)
        return out

    def forward(self, x):
        cat_meta, cont_meta = x
        x = self.forward_metadata(cat_meta, cont_meta)
        x = self.classifier(x)
        return x


class TabTransformer(nn.Module):
    def __init__(self, categories, no_continuous, out_dim, *,
                 dim=32, depth=6, heads=8):
        # !pip install tab-transformer-pytorch
        from tab_transformer_pytorch import TabTransformer

        assert len(categories) + no_continuous > 0
        super().__init__()
        self.net = TabTransformer(
            categories=categories,
            num_continuous=no_continuous,
            dim_out=out_dim,
            dim=dim,  # dimension, paper set at 32
            depth=depth,  # depth, paper recommended 6
            heads=heads,  # heads, paper recommends 8
            attn_dropout=0.1,  # post-attention dropout
            ff_dropout=0.1,  # feed forward dropout
            mlp_hidden_mults=(4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
            mlp_act=nn.ReLU(),  # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            # continuous_mean_std=cont_mean_std # (optional) - normalize the continuous values before layer norm
        )

    def forward(self, x):
        cat_meta, cont_meta = x
        return self.net(cat_meta, cont_meta)
