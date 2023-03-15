from typing import List
import torch
from sympy import lambdify, sympify
import copy



def replace_names(explanation: str, concept_names: List[str]) -> str:
    """
    Replace names of concepts in a formula.
    :param explanation: formula
    :param concept_names: new concept names
    :return: Formula with renamed concepts
    """
    feature_abbreviations = [f'feature{i:010}' for i in range(len(concept_names))]
    mapping = []
    for f_abbr, f_name in zip(feature_abbreviations, concept_names):
        mapping.append((f_abbr, f_name))

    for k, v in mapping:
        explanation = explanation.replace(k, v)

    return explanation


def get_predictions(formula: str, x: torch.Tensor, threshold: float = 0.5):
    """
    Tests a logic formula.
    :param formula: logic formula
    :param x: input data
    :param target_class: target class
    :return: Accuracy of the explanation and predictions
    """

    if formula in ['True', 'False', ''] or formula is None:
        return None

    else:
        concept_list = [f"feature{i:010}" for i in range(x.shape[1])]
        # get predictions using sympy
        # explanation = to_dnf(formula)
        explanation = sympify(formula)
        fun = lambdify(concept_list, explanation, 'numpy')
        x = x.cpu().detach().numpy()
        predictions = fun(*[x[:, i] > threshold for i in range(x.shape[1])])
        return predictions

def get_the_good_and_bad_terms(
    model, c, edge_index, sample_pos, explanation, target_class, concept_names=None, threshold=0.5
):
    def perturb_inputs_rem(inputs, target):
        if threshold == 0.5:
            inputs[:, target] = 0.0
        elif threshold == 0.:
            inputs[:, target] = -1.0
        return inputs

    def perturb_inputs_add(inputs, target):
        # inputs[:, target] += inputs.sum(axis=1) / (inputs != 0).sum(axis=1)
        # inputs[:, target] += inputs.max(axis=1)[0]
        inputs[:, target] = 1
        # inputs[:, target] += 1
        return inputs

    explanation = explanation.split(" & ")

    good, bad = [], []

    if edge_index is None:
        base = model(c)[sample_pos].view(1, -1)
    else:
        base = model(c, edge_index)[sample_pos].view(1, -1)

    for term in explanation:
        atom = term
        remove = True
        if atom[0] == "~":
            remove = False
            atom = atom[1:]

        if concept_names is not None:
            idx = concept_names.index(atom)
        else:
            idx = int(atom[len("feature") :])
        temp_tensor = c[sample_pos].clone().detach().view(1, -1)
        temp_tensor = (
            perturb_inputs_rem(temp_tensor, idx)
            if remove
            else perturb_inputs_add(temp_tensor, idx)
        )
        c2 = copy.deepcopy(c)
        c2[sample_pos] = temp_tensor
        if edge_index is None:
            new_pred = model(c2)[sample_pos].view(1, -1)
        else:
            new_pred = model(c2, edge_index)[sample_pos].view(1, -1)

        if new_pred[:, target_class] >= base[:, target_class]:
            bad.append(term)
        else:
            good.append(term)
        del temp_tensor
    return good, bad
