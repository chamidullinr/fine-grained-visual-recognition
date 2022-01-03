import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


"""Loss functions."""


class F1Loss(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps
        self.reduction = 'mean'

    def f1_loss(self, preds, targs, macro=True):
        assert preds.ndim == 1 or preds.ndim == 2
        assert targs.ndim == 1

        if preds.ndim == 2:
            preds = F.softmax(preds, dim=1)

        # count true positives and false positives and negatives
        labels = torch.arange(preds.shape[1], device=preds.device).reshape(-1, 1)

        # # create binary versions of targets and predictions
        targs_bin = (targs == labels).to(targs.dtype)
        preds = preds.T

        # count true positives, false positives and negatives
        tp_sum = (targs_bin * preds).sum(axis=1)  # true positive
        fp_sum = ((1 - targs_bin) * preds).sum(axis=1)  # false positive
        fn_sum = (targs_bin * (1 - preds)).sum(axis=1)  # false negative

        # compute precision and recall
        precision = tp_sum / (tp_sum + fp_sum + self.eps)
        recall = tp_sum / (tp_sum + fn_sum + self.eps)

        # compute f1 loss
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        f1 = f1.clamp(min=self.eps, max=1 - self.eps)

        if macro:
            f1 = f1.mean()

        return 1 - f1

    def forward(self, preds, targs):
        return self.f1_loss(preds, targs)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = 'mean'

    def forward(self, preds, targs):
        ce_loss = F.cross_entropy(preds, targs, weight=self.weight, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t)**self.gamma * ce_loss
        loss = loss.mean()
        return loss


def ce_loss(*args, weight=None, **kwargs):
    return nn.CrossEntropyLoss(weight=weight, reduction='mean')


def focal_loss(*args, gamma=2.0, weight=None, **kwargs):
    return FocalLoss(gamma, weight)


def f1_loss(*args, **kwargs):
    return F1Loss()


def mae_loss(*args, **kwargs):
    return nn.L1Loss()


def mse_loss(*args, **kwargs):
    return nn.MSELoss()


LOSSES = {
    'ce': ce_loss,
    'focal': focal_loss,
    'f1': f1_loss,
    'mae': mae_loss,
    'mse': mse_loss
}


"""Functions for computing class weights for loss functions."""


def inverse_class_frequency_weights(freq):
    return 1 / freq


def linear_class_weights(freq):
    return freq.max() / freq


def modified_linear_class_weights(freq):
    # thanks to http://ceur-ws.org/Vol-2936/paper-126.pdf
    weights = 1 - 1 / np.sqrt(freq.max() / freq + 0.5)
    return weights


def class_balanced_weights(freq, beta=0.99):
    # thanks to https://github.com/vandit15/Class-balanced-loss-pytorch
    # https://arxiv.org/abs/1901.05555v1
    # suggested beta values 0.9, 0.99, 0.999, 0.9999
    no_classes = len(freq)
    # compute effective number of samples
    weights = (1 - beta) / (1 - beta ** freq)
    # normalize
    weights = weights / weights.sum() * no_classes
    return weights


WEIGHTING = {
    'none': lambda *args, **kwargs: None,
    'inverse_class_frequency': inverse_class_frequency_weights,
    'linear_class': linear_class_weights,
    'modified_linear_class': modified_linear_class_weights,
    'class_balanced': class_balanced_weights
}
