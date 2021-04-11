import torch
from ..nn import Logic, Conv2Concepts


def prune_logic_layers(model: torch.nn.Module, current_epoch: int, prune_epoch: int,
                       fan_in: int = None, device: torch.device = torch.device('cpu')) -> torch.nn.Module:
    """
    Prune the inputs of the model.

    :param model: pytorch model
    :param fan_in: number of features to retain
    :param device: cpu or cuda device
    :return: pruned model
    """
    if current_epoch != prune_epoch:
        return model

    model.eval()
    for i, module in enumerate(model.children()):
        # prune only Linear layers
        if isinstance(module, Logic):
            if not module.top:
                if hasattr(module, 'weight_orig'):
                    pass
                _prune(module, fan_in, device=device)

        if isinstance(module, Conv2Concepts):
            _prune(module, fan_in, device=device)
        # break

    model.train()
    return model


def _prune(module: torch.nn.Module, fan_in: int, device: torch.device = torch.device('cpu')):
    # pruning
    w_size = (module.weight.shape[0], module.weight.shape[1])

    # identify weights with the lowest absolute values
    w_abs = torch.norm(module.weight, dim=0)

    w_sorted = torch.argsort(w_abs, descending=True)

    if fan_in:
        w_to_prune = w_sorted[fan_in:]
    else:
        w_max = torch.max(w_abs)
        w_to_prune = (w_abs / w_max) < 0.5

    mask = torch.ones(w_size)
    mask[:, w_to_prune] = 0

    # prune
    torch.nn.utils.prune.custom_from_mask(module, name="weight", mask=mask.to(device))
    return


def l1_loss(model: torch.nn.Module):
    loss = 0
    for module in model.children():
        if isinstance(module, Logic):
            loss += torch.norm(module.weight, 1) + torch.norm(module.bias, 1)
            break
    return loss


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
