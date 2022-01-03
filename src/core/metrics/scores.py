import numpy as np

from sklearn.metrics import (f1_score, accuracy_score, top_k_accuracy_score,
                             precision_score, recall_score,
                             mean_absolute_error, mean_squared_error)

__all__ = ['classification_scores', 'regression_scores']


def classification_scores(preds, targs, *, top_k=3, precision_recall=False):
    preds_argmax = preds.argmax(1)
    labels = np.arange(preds.shape[1])
    scores = {}
    scores['accuracy'] = accuracy_score(targs, preds_argmax)
    if top_k is not None and preds.shape[1] > 2:
        scores[f'top_{top_k}'] = top_k_accuracy_score(targs, preds, k=top_k, labels=labels)
    if precision_recall is True:
        scores['precision'] = precision_score(targs, preds_argmax, labels=labels, average='macro')
        scores['recall'] = recall_score(targs, preds_argmax, labels=labels, average='macro')
    scores['f1_score'] = f1_score(targs, preds_argmax, labels=labels, average='macro')
    return scores


def regression_scores(preds, targs):
    scores = {
        'mae': mean_absolute_error(targs, preds),
        'mse': mean_squared_error(targs, preds)}
    return scores
