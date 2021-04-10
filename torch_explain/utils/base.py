from typing import List, Tuple

import numpy as np
import torch
from sklearn.tree import _tree, DecisionTreeClassifier
from sympy import simplify_logic


def to_categorical(y: torch.Tensor) -> torch.Tensor:
    """
    Transform input tensor to categorical.

    :param y: input tensor.
    :return: Categorical tensor
    """
    if y.min() < 0:
        y = torch.nn.Sigmoid()(y)
    if len(y.shape) == 2 and y.shape[1] > 1:
        # one hot encoding to categorical
        yc = torch.argmax(y, dim=1)
    elif y.long().sum() == y.sum():
        # already argmax passed
        yc = y
    elif torch.max(y) > 1 or torch.min(y) < 0:
        # logits tensor
        yc = y.squeeze() > 0
    else:
        # binary/probabilities to categorical
        yc = y.squeeze() > 0.5

    if len(yc.size()) == 0:
        yc = yc.unsqueeze(0)

    return yc


def collect_parameters(model: torch.nn.Module,
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


def tree_to_formula(tree: DecisionTreeClassifier, concept_names: List[str], target_class: int) -> str:
    """
    Translate a decision tree into a set of decision rules.

    :param tree: sklearn decision tree
    :param concept_names: concept names
    :param target_class: target class
    :return: decision rule
    """
    tree_ = tree.tree_
    feature_name = [
        concept_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    pathto = dict()

    global k
    global explanation
    explanation = ''
    k = 0

    def recurse(node, depth, parent):
        global k
        global explanation
        indent = "  " * depth

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            s = f'{name} <= {threshold:.2f}'
            if node == 0:
                pathto[node] = s
            else:
                pathto[node] = pathto[parent] + ' & ' + s

            recurse(tree_.children_left[node], depth + 1, node)
            s = f'{name} > {threshold:.2f}'
            if node == 0:
                pathto[node] = s
            else:
                pathto[node] = pathto[parent] + ' & ' + s
            recurse(tree_.children_right[node], depth + 1, node)
        else:
            k = k + 1
            if tree_.value[node].squeeze().argmax() == target_class:
                explanation += f'({pathto[parent]}) | '

    recurse(0, 1, 0)
    return explanation[:-3]


def brl_extracting_formula(model) -> str:
    min_terms = []
    for i, j in enumerate(model.d_star):
        if model.itemsets[j] != 'null' and model.theta[i]:  # > 0.5:
            min_term = (" & ".join([str(model.itemsets[j][k])
                                    for k in range(len(model.itemsets[j]))]))
            min_terms.append(min_term)

    formula = ""
    for i, min_term in enumerate(min_terms):
        if model.theta[i] >= 0.5:
            part_formula = min_term
            # Taking into consideration all the previous terms negated
            for j, min_term2 in enumerate(min_terms[:i]):
                part_formula += f" & ~({min_term2})"
            formula += f"({part_formula}) | "

    # Taking into consideration the ELSE (only in case it implies the class)
    i = len(min_terms)
    if model.theta[i] >= 0.5:
            formula += f" & ".join([f"~({min_term2})" for min_term2 in min_terms])
    else:
        formula = formula[:-3]
    if formula == "":
        formula = "false"

    simplified_formula = str(simplify_logic(formula, form="dnf"))

    return simplified_formula
