import numpy as np


def get_top1_ensemble(preds):
    def max_voting(arr):
        vals, counts = np.unique(arr, return_counts=True)
        return vals[np.argmax(counts)]

    # get most frequent predictions
    top1_preds = np.concatenate([x.argmax(1)[..., None] for x in preds], axis=1)
    top1_ensemble = np.apply_along_axis(max_voting, 1, top1_preds)

    # create one-hot matrix
    no_classes = preds[0].shape[1]
    top1_ensemble = np.float64(top1_ensemble[..., None] == np.arange(no_classes)[None, ...])

    return top1_ensemble


def get_product_ensemble(preds):
    prod_ensemble = np.exp(np.sum([np.log(x) for x in preds], 0))
    prod_ensemble /= prod_ensemble.sum(1)[..., None]
    return prod_ensemble
