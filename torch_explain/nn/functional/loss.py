import torch

from torch_explain.nn.logic import LogicAttention


def l1_loss(model: torch.nn.Module):
    loss = 0
    for module in model.children():
        if isinstance(module, LogicAttention) and module.shrink:
            loss += torch.norm(module.beta, 1)
            break
    return loss


def whitening_loss(model: torch.nn.Module, device: torch.device = torch.device('cpu')):
    loss = 0
    cov = None
    for module in model.children():
        if isinstance(module, LogicAttention):
            # the target covariance matrix is diagonal
            n_concepts = module.conceptizator.concepts.shape[1]
            cov_objective = torch.eye(n_concepts).to(device)
            # compute covariance matrix of activations
            cov = 1 / (n_concepts - 1) * torch.matmul(module.conceptizator.concepts.T, module.conceptizator.concepts)
            loss += torch.norm(cov - cov_objective, p=2)
            break
    return loss, cov
