import torch as t
import numpy as np

def cloud_mean_absolute_error(y_true, y_pred):
    """Computes the MAE over the full image."""
    return t.mean(t.abs(y_pred[:, 0:3, :, :] - y_true[:, 0:3, :, :]))


def cloud_mean_squared_error(y_true, y_pred):
    """Computes the MSE over the full image."""
    return t.mean(t.square(y_pred[:, 0:3, :, :] - y_true[:, 0:3, :, :]))


def cloud_root_mean_squared_error(y_true, y_pred):
    """Computes the RMSE over the full image."""
    return t.sqrt(t.mean(t.square(y_pred[:, 0:3, :, :] - y_true[:, 0:3, :, :])))


def cloud_bandwise_root_mean_squared_error(y_true, y_pred):
    return t.mean(t.sqrt(t.mean(t.square(y_pred[:, 0:3, :, :] - y_true[:, 0:3, :, :]), axis=[2, 3])))



