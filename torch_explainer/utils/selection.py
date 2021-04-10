import torch
import numpy as np

from .base import collect_parameters, to_categorical


def rank_pruning(model: torch.nn.Module,
                 x_sample: torch.Tensor,
                 y: torch.Tensor,
                 device: torch.device = torch.device('cpu'),
                 num_classes: int = None) -> np.ndarray:
    """
    Feature ranking by pruning.

    :param model: torch model
    :param x_sample: input sample
    :param y: input label
    :param device: cpu or cuda device
    :param num_classes: override the number of classes
    :return: best features
    """
    # get the model prediction on the individual sample
    y_pred_sample = model(x_sample)
    pred_class = to_categorical(y_pred_sample)
    y = to_categorical(y)

    # identify non-pruned features
    w, b = collect_parameters(model, device)
    feature_weights = w[0]

    n_classes = len(torch.unique(y)) if num_classes is None else num_classes
    if n_classes < 2 or (num_classes is None and n_classes <= 2):
        feature_used_bool = np.sum(np.abs(feature_weights), axis=0) > 0

    else:
        block_size = feature_weights.shape[0] // n_classes
        feature_used_bool = np.sum(np.abs(feature_weights[pred_class * block_size:(pred_class + 1) * block_size]), axis=0) > 0

    feature_used = np.sort(np.nonzero(feature_used_bool)[0])
    return feature_used
