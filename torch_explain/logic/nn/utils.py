from typing import Tuple, List

import numpy as np
import torch


def _collect_parameters(model: torch.nn.Module,
                        device: torch.device = torch.device('cpu')) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Collect network parameters in two lists of numpy arrays.

    :param model: pytorch model
    :param device: cpu or cuda device
    :return: list of weights and list of biases
    """
    weights, bias = [], []
    for module in model.children():
        if isinstance(module, torch.nn.Linear):
            if device.type == 'cpu':
                weights.append(module.weight.detach().numpy())
                try:
                    bias.append(module.bias.detach().numpy())
                except:
                    pass

            else:
                weights.append(module.weight.cpu().detach().numpy())
                try:
                    bias.append(module.bias.cpu().detach().numpy())
                except:
                    pass

    return weights, bias
