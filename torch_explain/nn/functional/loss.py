import torch

from torch_explain.nn.logic import ConceptAwareness


def l1_loss(model: torch.nn.Module):
    loss = 0
    for module in model.children():
        if isinstance(module, ConceptAwareness) and module.shrink:
            loss += torch.norm(module.weight, 1)
            break
    return loss


def concept_awareness_loss(model: torch.nn.Module):
    loss = 0
    for module in model.children():
        if isinstance(module, ConceptAwareness) and module.shrink:
            for alpha in module.alpha:
                loss -= torch.sum(alpha * torch.log(alpha))
            # loss += torch.norm(module.beta.squeeze(), p=1) + 0.01 * torch.norm(module.weight, 2)
            break
    return loss


def whitening_loss(model: torch.nn.Module, device: torch.device = torch.device('cpu')):
    loss = 0
    cov = None
    for module in model.children():
        if isinstance(module, ConceptAwareness):
            # the target covariance matrix is diagonal
            n_concepts = module.conceptizator.concepts.shape[1]
            cov_objective = torch.eye(n_concepts).to(device)
            # compute covariance matrix of activations
            cov = 1 / (n_concepts - 1) * torch.matmul(module.conceptizator.concepts.T, module.conceptizator.concepts)
            loss += torch.norm(cov - cov_objective, p=2)
            break
    return loss, cov
