import numpy as np
import torch


def get_saliency_map(model, img, label=None, device=None):
    if len(img.shape) == 4:
        bs, ch, h, w = img.shape
        assert bs == 1, 'Function accepts only one image as an input.'
    elif len(img.shape) == 3:
        img = img.unsqueeze(0)
        bs, ch, h, w = img.shape
    else:
        raise ValueError('Invalid shape on input image')

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)  # this could slow the model
    model = model.eval()
    img = img.to(device)
    img = img.requires_grad_()

    # run inference
    output = model(img)
    idx = output.argmax() if label is None else label
    val = output[:, idx]

    # do backpropagation and get the gradients
    val.backward()

    # create saliency map
    img_grad = img.grad.data.abs()
    saliency, _ = img_grad.max(dim=1)
    saliency = saliency.cpu().numpy().reshape(h, w)

    return saliency


def binarize_saliency_map(saliency_map, percentile=95):
    x = saliency_map.reshape(-1)
    p95 = np.percentile(x, 95)
    out = (saliency_map > p95).astype(np.uint8)
    return out


def get_bounding_box(saliency_map, bin_percentile=95, clust_pixel_th=10):
    from scipy import ndimage

    # binarize saliency map
    saliency_map_bin = binarize_saliency_map(
        saliency_map, percentile=bin_percentile)

    # identify clusters in the binary saliency map
    cluster_map, num_features = ndimage.label(saliency_map_bin)

    # count labels in the clustered map
    _cluster_map = cluster_map[cluster_map != 0]  # remove background labels
    labels, counts = np.unique(
        _cluster_map.reshape(-1), return_counts=True)

    # clean binary saliency map by removing small clusters
    labels = labels[counts > clust_pixel_th]
    saliency_map_cleaned = np.isin(cluster_map, labels).astype(np.uint8)

    # compute bounding box
    y, x = np.where(saliency_map_cleaned)
    bbox = (x.min(), y.min(), x.max(), y.max())
    # bbox = np.array([[xmin, ymin], [xmax, ymax]])

    return bbox, saliency_map_cleaned
