import torch

from torch_explain.nn import Logic
from torch_explain.nn.logic import LogicAttention


def l1_loss(model: torch.nn.Module):
    loss = 0
    for module in model.children():
        if isinstance(module, LogicAttention) and module.shrink:
            loss += torch.norm(module.weight, 1, dim=1).norm(p=1) + torch.norm(module.bias, 1)
    return loss


def concept_activation_loss(model: torch.nn.Module):
    activation_loss = 0
    for module in model.modules():
        if isinstance(module, Logic) and not module.top:
            activation_loss += torch.norm(module.conceptizator.concepts.view(-1) - 0.5, 1)
            # activation_loss += torch.norm(module.conceptizator.concepts.view(-1).abs() - 1, 1)
    return activation_loss


def whitening_loss(model: torch.nn.Module, device: torch.device = torch.device('cpu')):
    loss = 0
    cov = None
    for module in model.children():
        if isinstance(module, Logic):
            # the target covariance matrix is diagonal
            n_concepts = module.conceptizator.concepts.shape[1]
            cov_objective = torch.eye(n_concepts).to(device)
            # compute covariance matrix of activations
            cov = 1 / (n_concepts - 1) * torch.matmul(module.conceptizator.concepts.T, module.conceptizator.concepts)
            loss += torch.norm(cov - cov_objective, p=2)
            break
    return loss, cov
