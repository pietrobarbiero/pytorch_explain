import torch
from torch.nn import Linear

from torch_explain.nn.logic import EntropyLinear


def entropy_logic_loss(model: torch.nn.Module):
    """
    Entropy loss function to get simple logic explanations.

    :param model: pytorch model.
    :return: entropy loss.
    """
    loss = 0
    for module in model.children():
        if isinstance(module, EntropyLinear):
            loss -= torch.sum(module.alpha * torch.log(module.alpha))
            break
    return loss


def l1_loss(model: torch.nn.Module):
    """
    L1 loss function to get simple logic explanations.

    :param model: pytorch model.
    :return: L1 loss.
    """
    loss = 0
    for module in model.children():
        if isinstance(module, Linear):
            loss += torch.norm(module.weight, 1)
            break
    return loss
