"""Losses + Quadratic Weighted Kappa threshold optimiser.

We train an ordinal *regressor* (one scalar). To turn scores into grades we fit
rounding thresholds that maximise QWK on the validation fold - the classic
Abhishek-Thakur `OptimizedRounder`. This decouples the metric from the loss and
generalises better across datasets than a fixed round().
"""
import numpy as np
import torch
import torch.nn as nn
from functools import partial

import scipy.optimize as opt
from sklearn.metrics import cohen_kappa_score


def get_loss(cfg):
    if cfg.loss == "mse":
        return nn.MSELoss()
    return nn.SmoothL1Loss(beta=1.0)


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


class OptimizedRounder:
    """Find thresholds t0<t1<t2<t3 that map a scalar score to grades 0..4."""

    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.coef_ = [0.5, 1.5, 2.5, 3.5]

    def _kappa_loss(self, coef, X, y):
        preds = self.predict(X, coef)
        return -quadratic_weighted_kappa(y, preds)

    def fit(self, X, y):
        loss_fn = partial(self._kappa_loss, X=X, y=y)
        init = [0.5, 1.5, 2.5, 3.5]
        res = opt.minimize(loss_fn, init, method="nelder-mead",
                           options={"maxiter": 1000, "xatol": 1e-3})
        self.coef_ = sorted(res.x.tolist())
        return self

    def predict(self, X, coef=None):
        coef = self.coef_ if coef is None else coef
        X = np.asarray(X).ravel()
        return np.digitize(X, coef).astype(int).clip(0, self.num_classes - 1)
