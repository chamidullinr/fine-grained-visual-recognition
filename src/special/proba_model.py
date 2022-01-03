import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity


def estimate_relative_freq(arr) -> np.ndarray:
    _, counts = np.unique(arr, return_counts=True)
    dist = counts / counts.sum()
    return dist


def combine_predictions(img_pred, metadata_preds, class_priors):
    combined_preds = np.ones(img_pred.shape)
    for metadata_pred in metadata_preds:
        combined_preds *= metadata_pred / class_priors

    # create predictions combined with metadata-target distributions
    pred_adj = img_pred * combined_preds
    return pred_adj


class HistogramClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, m=1):
        self.m = m

    def fit(self, X, y):
        assert len(X.shape) == 1 or X.shape[1] == 1

        # estimate conditional probability
        cond = pd.notnull(X)
        self.model_ = pd.crosstab(X[cond], y[cond]) + self.m
        self.model_ = self.model_.divide(self.model_.sum(1), axis=0)
        self.classes_ = np.array(self.model_.columns)
        return self

    def predict_proba(self, X):
        # set defaults values for unknow records
        probs = np.ones((len(X), len(self.classes_))) / len(self.classes_)
        cond = np.isin(X, self.model_.index)
        probs[cond] = self.model_.loc[X[cond]].values
        return probs    

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]    


class KDEClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian generative classification based on KDE.

    thanks to: https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian', metric='euclidean', **kwargs):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric
        self._kwargs = kwargs

    def fit(self, X, y):
        cond = pd.notnull(X)
        if len(cond.shape) > 1:
            cond = cond.all(1)
        X, y = X[cond], y[cond]

        self.classes_ = np.unique(y)
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel,
                                      metric=self.metric,
                                      **self._kwargs).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(len(Xi) / len(X)) for Xi in training_sets]
        return self
        
    def predict_proba(self, X):
        # set defaults values for nan records
        probs = np.ones((len(X), len(self.classes_))) / len(self.classes_)
        cond = pd.notnull(X)
        if len(cond.shape) > 1:
            cond = cond.all(1)
        logprobs = np.array([model.score_samples(X[cond])
                             for model in self.models_]).T
        probs[cond] = np.exp(logprobs + self.logpriors_)
        return probs / probs.sum(1, keepdims=True)
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]
