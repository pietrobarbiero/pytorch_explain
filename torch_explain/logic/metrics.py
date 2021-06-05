from typing import List, Tuple

import sympy

import torch
import numpy as np
from sklearn.metrics import f1_score
from sympy import to_dnf, lambdify


def test_explanation(formula: str, x: torch.Tensor, y: torch.Tensor, target_class: int):
    """
    Tests a logic formula.

    :param formula: logic formula
    :param x: input data
    :param y: input labels (MUST be one-hot encoded)
    :param target_class: target class
    :return: Accuracy of the explanation and predictions
    """

    if formula in ['True', 'False', ''] or formula is None:
        return 0.0, None

    else:
        assert len(y.shape) == 2
        y = y[:, target_class]
        concept_list = [f"feature{i:010}" for i in range(x.shape[1])]
        # get predictions using sympy
        explanation = to_dnf(formula)
        fun = lambdify(concept_list, explanation, 'numpy')
        x = x.cpu().detach().numpy()
        predictions = fun(*[x[:, i] > 0.5 for i in range(x.shape[1])])
        # get accuracy
        accuracy = f1_score(y, predictions, average='macro')
        return accuracy, predictions


def complexity(formula: str, to_dnf: bool = False) -> float:
    """
    Estimates the complexity of the formula.

    :param formula: logic formula.
    :param to_dnf: whether to convert the formula in disjunctive normal form.
    :return: The complexity of the formula.
    """
    if formula != "" and formula is not None:
        if to_dnf:
            formula = str(sympy.to_dnf(formula))
        return np.array([len(f.split(' & ')) for f in formula.split(' | ')]).sum()
    return 0


def concept_consistency(formula_list: List[str]) -> dict:
    """
    Computes the frequency of concepts in a list of logic formulas.

    :param formula_list: list of logic formulas.
    :return: Frequency of concepts.
    """
    concept_dict = _generate_consistency_dict(formula_list)
    return {k: v / len(formula_list) for k, v in concept_dict.items()}


def formula_consistency(formula_list: List[str]) -> float:
    """
    Computes the average frequency of concepts in a list of logic formulas.

    :param formula_list: list of logic formulas.
    :return: Average frequency of concepts.
    """
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
