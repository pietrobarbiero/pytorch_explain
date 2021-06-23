import torch
from torch.nn.utils import prune


def prune_equal_fanin(model: torch.nn.Module, epoch: int, prune_epoch: int, k: int = 2,
                      device: torch.device = torch.device('cpu')) -> torch.nn.Module:
    """
    Prune the linear layers of the network such that each neuron has the same fan-in.

    :param model: pytorch model.
    :param epoch: current training epoch.
    :param prune_epoch: training epoch when pruning needs to be applied.
    :param k: fan-in.
    :param device: cpu or cuda device.
    :return: Pruned model
    """
    if epoch != prune_epoch:
        return model

    model.eval()
    for i, module in enumerate(model.children()):
        # prune only Linear layers
        if isinstance(module, torch.nn.Linear):
            # create mask
            mask = torch.ones(module.weight.shape)
            # identify weights with the lowest absolute values
            param_absneg = -torch.abs(module.weight)
            idx = torch.topk(param_absneg, k=param_absneg.shape[1] - k, dim=1)[1]
            for j in range(len(idx)):
                mask[j, idx[j]] = 0
            # prune
            mask = mask.to(device)
            prune.custom_from_mask(module, name="weight", mask=mask)
            # print(f"Pruned {k}/{module.weight.shape[1]} weights")

    return model
