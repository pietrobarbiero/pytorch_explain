from typing import List

import sympy
import numpy as np


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
