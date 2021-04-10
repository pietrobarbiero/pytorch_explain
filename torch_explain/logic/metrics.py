from typing import List

import sympy
import torch
import numpy as np
from sklearn.metrics import accuracy_score

from torch_explain.logic import test_explanation


def predictions(explanation: str, target_class: int, x: torch.Tensor, y: torch.Tensor,
                give_local: bool = False, metric: callable = accuracy_score,
                concept_names: list = None) -> np.ndarray:
    return test_explanation(explanation, target_class, x, y, give_local, metric, concept_names)[1]


def fidelity(y_formula: torch.Tensor, y_pred: torch.Tensor, metric: callable = accuracy_score) -> float:
    return metric(y_formula, y_pred)


def accuracy_score(y_formula: torch.Tensor, y_true: torch.Tensor, metric: callable = accuracy_score) -> float:
    return metric(y_formula, y_true)


def complexity(formula: str, to_dnf=False) -> float:
    if formula != "":
        if to_dnf:
            formula = str(sympy.to_dnf(formula))
        return np.array([len(f.split(' & ')) for f in formula.split(' | ')]).sum()
    return 1


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
