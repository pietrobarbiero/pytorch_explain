import torch

from torch_explain.nn import Logic, Conv2Concepts


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
