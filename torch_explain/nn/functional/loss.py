import torch
from torch import Tensor
from torch.nn import Linear

from torch_explain.nn.logic import EntropyLinear, LogDiffPool

EPS = 1e-15


def _entropy_loss(alpha: Tensor):
    return -torch.sum(alpha * torch.log(alpha + EPS))
    # ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()


def entropy_logic_loss(model: torch.nn.Module):
    """
    Entropy loss function to get simple logic explanations.

    :param model: pytorch model.
    :return: entropy loss.
    """
    loss = 0
    for module in model.children():
        if isinstance(module, EntropyLinear):
            loss += _entropy_loss(module.alpha)
            break

        elif isinstance(module, LogDiffPool):
            module.entropy_loss = 0
            module.link_loss = 0
            module.entropy_loss = _entropy_loss(module.alpha_r) + _entropy_loss(module.alpha_c)
            # add link loss
            if module.compute_link_loss:
                link_loss = module.adj - torch.matmul(module.gamma, module.gamma.transpose(1, 2))
                link_loss = torch.norm(link_loss, p=2)
                module.link_loss = link_loss / module.adj.numel()

            loss += module.entropy_loss + module.link_loss

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
