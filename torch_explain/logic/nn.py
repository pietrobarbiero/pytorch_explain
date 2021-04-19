import copy
from typing import List
import collections

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sympy import simplify_logic

from torch_explain.logic.metrics import test_explanation
from torch_explain.logic.utils import replace_names
from torch_explain.nn.logic import LogicAttention


def explain_class(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                  target_class: int, max_minterm_complexity: int = None, topk_explanations: int = 3,
                  max_accuracy: bool = False, concept_names: List = None) -> str:
    """
    Generate a local explanation for a single sample.

    :param model: pytorch model
    :param x: input samples
    :param y: target labels
    :param target_class: class ID
    :param topk_explanations: number of local explanations to be combined
    :param concept_names: list containing the names of the input concepts
    :return: Local explanation
    """
    x_validation, y_validation = _get_validation_data(x, y, model, target_class)
    if x_validation is None:
        return None, None

    class_explanation = ''
    class_explanations = {}
    is_first = True
    for layer_id, module in enumerate(model.children()):
        # analyze only logic layers
        if isinstance(module, LogicAttention):  # or isinstance(module, XLogicConv2d):

            if is_first:
                prev_module = module
                is_first = False
                feature_names = [f'feature{j:010}' for j in range(x_validation.size(1))]
                c_validation = prev_module.conceptizator.concepts[0]

            elif module.top:
                explanations = []
                n_classes = module.conceptizator.concepts.size(0)
                for neuron in range(n_classes):
                    if module.top and neuron != target_class:
                        continue

                    local_explanations = []
                    local_explanations_raw = {}

                    if module.top:
                        y_target = module.conceptizator.concepts.argmax(dim=1).eq(target_class)
                    else:
                        y_target = module.conceptizator.concepts[:, neuron] > module.conceptizator.threshold

                    neuron_list = torch.nonzero(y_target)
                    for i in neuron_list:
                        simplify = True if module.top else False
                        local_explanation, local_explanation_raw = _local_explanation(prev_module, feature_names, i,
                                                                                      local_explanations_raw,
                                                                                      c_validation, y_target,
                                                                                      target_class, simplify,
                                                                                      max_accuracy,
                                                                                      max_minterm_complexity)

                        if local_explanation and local_explanation_raw:
                            local_explanations_raw[local_explanation_raw] = local_explanation_raw
                            local_explanations.append(local_explanation)

                    aggregated_explanation = _aggregate_explanations(local_explanations, topk_explanations)

                    explanations.append(f'{aggregated_explanation}')
                    class_explanations[f'layer_{layer_id}-neuron_{neuron}'] = str(aggregated_explanation)

                prev_module = module
                c_validation = prev_module.conceptizator.concepts
                feature_names = explanations
                if not aggregated_explanation:
                    aggregated_explanation = ''
                class_explanation = str(aggregated_explanation)

    class_explanation_raw = class_explanation
    class_explanation, class_explanations = _replace_names_dict(class_explanation, class_explanations, concept_names)

    return class_explanation[1:-1], class_explanations, class_explanation_raw


def _simplify_formula(explanation: str, x: torch.Tensor, y: torch.Tensor, target_class: int, max_accuracy: bool) -> str:
    """
    Simplify formula to a simpler one that is still coherent.

    :param explanation: local formula to be simplified.
    :param x: input data.
    :param y: target labels (1D, categorical NOT one-hot encoded).
    :param target_class: target class
    :param max_accuracy: drop  term only if it gets max accuracy
    :return: Simplified formula
    """

    base_accuracy, _ = test_explanation(explanation, target_class, x, y, metric=accuracy_score)
    for term in explanation.split(' & '):
        explanation_simplified = copy.deepcopy(explanation)

        if explanation_simplified.endswith(f'{term}'):
            explanation_simplified = explanation_simplified.replace(f' & {term}', '')
        else:
            explanation_simplified = explanation_simplified.replace(f'{term} & ', '')

        if explanation_simplified:
            accuracy, preds = test_explanation(explanation_simplified, target_class, x, y, metric=accuracy_score)
            if (max_accuracy and accuracy == 1.) or (not max_accuracy and accuracy >= base_accuracy):
                explanation = copy.deepcopy(explanation_simplified)

    return explanation


def _aggregate_explanations(local_explanations, topk_explanations):
    if len(local_explanations) == 0:
        return ''

    else:
        # get most frequent local explanations
        counter = collections.Counter(local_explanations)
        topk = topk_explanations
        if len(counter) < topk_explanations:
            topk = len(counter)
        most_common_explanations = []
        for explanation, _ in counter.most_common(topk):
            most_common_explanations.append(f'({explanation})')

        # aggregate example-level explanations
        if local_explanations:
            aggregated_explanation = ' | '.join(most_common_explanations)
            aggregated_explanation_simplified = simplify_logic(aggregated_explanation, 'dnf', force=True)
            aggregated_explanation_simplified = f'({aggregated_explanation_simplified})'
        else:
            aggregated_explanation_simplified = ''
    return aggregated_explanation_simplified


def _local_explanation(prev_module, feature_names, neuron_id, neuron_explanations_raw,
                       c_validation, y_target, target_class, simplify, max_accuracy, max_minterm_complexity):
    # explanation is the conjunction of non-pruned features
    explanation_raw = ''
    non_pruned_neurons = prev_module.alpha[target_class]
    if max_minterm_complexity:
        neurons_to_retain = torch.argsort(non_pruned_neurons, descending=True)[:max_minterm_complexity]
    else:
        neurons_to_retain_idx = prev_module.beta[target_class] > 0.5
        neurons_sorted = torch.argsort(prev_module.beta[target_class])
        neurons_to_retain = neurons_sorted[neurons_to_retain_idx[neurons_sorted]]
    for j in neurons_to_retain:
        if feature_names[j] not in ['()', '']:
            if explanation_raw:
                explanation_raw += ' & '
            if prev_module.conceptizator.concepts[0][neuron_id, j] > prev_module.conceptizator.threshold:
                # if non_pruned_neurons[j] > 0:
                explanation_raw += feature_names[j]
            else:
                explanation_raw += f'~{feature_names[j]}'

    explanation_raw = str(explanation_raw)
    if explanation_raw in ['', 'False', 'True', '(False)', '(True)']:
        return None, None

    if explanation_raw in neuron_explanations_raw:
        explanation = neuron_explanations_raw[explanation_raw]
    elif simplify:
        # accuracy, _ = test_explanation(explanation_raw, target_class, c_validation, y_target, metric=accuracy_score)
        explanation = _simplify_formula(explanation_raw, c_validation, y_target, target_class, max_accuracy)
        # accuracy2, _ = test_explanation(explanation, target_class, c_validation, y_target, metric=accuracy_score)
        # print(f'{accuracy:.2f} VS {accuracy2:.2f}')
    else:
        explanation = explanation_raw

    if explanation in ['', 'False', 'True', '(False)', '(True)']:
        return None, None

    return explanation, explanation_raw


def _replace_names_dict(class_explanation, class_explanations, concept_names):
    if concept_names is not None:
        class_explanation = replace_names(class_explanation, concept_names)
        for k, explanation in class_explanations.items():
            class_explanations[k] = replace_names(explanation, concept_names)
    return class_explanation, class_explanations


def _get_validation_data(x, y, model, target_class):
    _, idx = np.unique((x[y.argmax(dim=1) == target_class] >= 0.5).cpu().detach().numpy(), axis=0, return_index=True)
    if len(idx) == 1:
        idx = torch.tensor([idx, idx]).squeeze()
    x_target = x[y.argmax(dim=1) == target_class][idx]
    y_target = y[y.argmax(dim=1) == target_class][idx]

    # get model's predictions
    preds = model(x_target)

    # identify samples correctly classified of the target class
    correct_mask = y_target.argmax(dim=1).eq(preds.argmax(dim=1))
    if sum(correct_mask) < 2:
        return None, None

    x_target_correct = x_target[correct_mask]
    y_target_correct = y_target[correct_mask]

    # collapse samples having the same boolean values and class label different from the target class
    _, idx = np.unique((x[y.argmax(dim=1) != target_class] > 0.5).cpu().detach().numpy(), axis=0, return_index=True)
    x_reduced_opposite = x[y.argmax(dim=1) != target_class][idx]
    y_reduced_opposite = y[y.argmax(dim=1) != target_class][idx]
    preds_opposite = model(x_reduced_opposite)

    # identify samples correctly classified of the opposite class
    correct_mask = y_reduced_opposite.argmax(dim=1).eq(preds_opposite.argmax(dim=1))
    if sum(correct_mask) < 2:
        return None, None

    x_reduced_opposite_correct = x_reduced_opposite[correct_mask]
    y_reduced_opposite_correct = y_reduced_opposite[correct_mask]

    # select the subset of samples belonging to the target class
    x_validation = torch.cat([x_reduced_opposite_correct, x_target_correct], dim=0)
    y_validation = torch.cat([y_reduced_opposite_correct, y_target_correct], dim=0)

    model.eval()
    model(x_validation)
    return x_validation, y_validation
