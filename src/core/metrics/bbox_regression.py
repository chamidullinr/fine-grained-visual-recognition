from typing import Iterable

import numpy as np


__all__ = ['iou']


def iou(bbox1: Iterable, bbox2: Iterable, macro=True):
    """
    Compute Intersection over Union (IoU).
    
    The method accepts either single bounding box
    or numpy array of muplitple bounding boxes.
    The computation uses vectorized operations bringing high performance.
    """
    bbox1, bbox2 = np.array(bbox1), np.array(bbox2)
    assert bbox1.shape == bbox2.shape
    assert bbox1.shape[0] == 4 or bbox1.shape[1] == 4

    # extract bounding box coordinates
    if bbox1.shape[0] != 4:
        bbox1 = bbox1.T
        bbox2 = bbox2.T
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    # compute middle steps
    w1, h1 = xmax1 - xmin1, ymax1 - ymin1
    w2, h2 = xmax2 - xmin2, ymax2 - ymin2
    w_intersection = np.min([xmax1, xmax2], 0) - np.max([xmin1, xmin2], 0)
    h_intersection = np.min([ymax1, ymax2], 0) - np.max([ymin1, ymin2], 0)

    # create empty output array
    out = np.zeros(len(w_intersection))

    # compute IoU in overlapping records
    cond = (w_intersection > 0) & (h_intersection > 0)
    intersection = w_intersection[cond] * h_intersection[cond]
    union = w1[cond] * h1[cond] + w2[cond] * h2[cond] - intersection
    out[cond] = intersection / union

    # post-process output
    if macro is True:
        out = np.mean(out)
    elif len(out) == 1:
        out = out[0]

    return out


# def iou(bbox1, bbox2):
#     xmin1, ymin1, xmax1, ymax1 = bbox1
#     xmin2, ymin2, xmax2, ymax2 = bbox2
#     w1, h1 = xmax1 - xmin1, ymax1 - ymin1
#     w2, h2 = xmax2 - xmin2, ymax2 - ymin2

#     w_intersection = min(xmax1, xmax2) - max(xmin1, xmin2)
#     h_intersection = min(ymax1, ymax2) - max(ymin1, ymin2)

#     if w_intersection <= 0 or h_intersection <= 0: 
#         out = 0
#     else:
#         intersection = w_intersection * h_intersection
#         union = w1 * h1 + w2 * h2 - intersection
#         out = intersection / union
#     return out


# def average_precision(bbox1, bbox2):
#     bbox1, bbox2 = np.array(bbox1), np.array(bbox2)
#     return average_precision_score(bbox1, bbox2)
