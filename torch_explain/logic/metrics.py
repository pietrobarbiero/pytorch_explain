from typing import List, Tuple

import sympy

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sympy import to_dnf, lambdify


def test_explanation(explanation: str, target_class: int, x: torch.Tensor, y: torch.Tensor,
                     give_local: bool = False, metric: callable = accuracy_score, concept_names: list = None) \
        -> Tuple[float, np.ndarray]:
    """
    Test explanation

    :param explanation: formula
    :param target_class: class ID
    :param x: input data
    :param y: input labels (categorical, NOT one-hot encoded)
    :param give_local: if true will return local predictions
    :return: Accuracy of the explanation and predictions
    """

    assert concept_names is not None or "feature" in explanation or explanation == "", \
        "Concept names must be given when present in the formula"

    if explanation == '' or explanation == None:
        local_predictions = [torch.empty_like(y)]
        predictions = torch.cat(local_predictions).eq(target_class).cpu().detach().numpy()
        accuracy = 0.0
        return accuracy, torch.stack(local_predictions, dim=0).sum(dim=0) > 0 if give_local else predictions
    if explanation == "(True)" or explanation == "True":
        local_predictions = [torch.tensor(np.ones_like(y))]
        predictions = torch.cat(local_predictions).eq(target_class).cpu().detach().numpy()
    elif explanation == "(False)" or explanation == "False":
        local_predictions = [torch.tensor(np.zeros_like(y))]
        predictions = torch.cat(local_predictions).eq(target_class).cpu().detach().numpy()
    else:

        concept_list = [f"feature{i:010}" for i in range(x.shape[1])]
        if concept_names is not None:
            for i, concept_name in enumerate(concept_names):
                explanation = explanation.replace(concept_name, f"feature{i:010}")

        explanation = to_dnf(explanation)
        fun = lambdify(concept_list, explanation, 'numpy')
        x = x.cpu().detach().numpy()
        predictions = fun(*[x[:, i] > 0.5 for i in range(x.shape[1])])

    accuracy = metric(y, predictions)
    return accuracy, predictions


def complexity(formula: str, to_dnf=False) -> float:
    if formula != "" and formula is not None:
        if to_dnf:
            formula = str(sympy.to_dnf(formula))
        return np.array([len(f.split(' & ')) for f in formula.split(' | ')]).sum()
    return 0


def concept_consistency(formula_list: List[str]) -> dict:
    concept_dict = _generate_consistency_dict(formula_list)
    return {k: v / len(formula_list) for k, v in concept_dict.items()}


def formula_consistency(formula_list: List[str]) -> float:
    concept_dict = _generate_consistency_dict(formula_list)
    concept_consistency = np.array([c for c in concept_dict.values()]) / len(formula_list)
    return concept_consistency.mean()


def _generate_consistency_dict(formula_list: List[str]) -> dict:
    concept_dict = {}
    for i, formula in enumerate(formula_list):
        concept_dict_i = {}
        for minterm_list in formula.split(' | '):
            for term in minterm_list.split(' & '):
                concept = term.replace('(', '').replace(')', '').replace('~', '')
                if concept in concept_dict_i:
                    continue
                elif concept in concept_dict:
                    concept_dict_i[concept] = 1
                    concept_dict[concept] += 1
                else:
                    concept_dict_i[concept] = 1
                    concept_dict[concept] = 1
    return concept_dict
