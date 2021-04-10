import copy
from typing import List
import collections

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sympy import simplify_logic

from .base import replace_names, test_explanation
from ..nn import XLogic  # , XLogicConv2d
from ..utils.base import to_categorical


def explain_class(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, binary: bool,
                  target_class: int, simplify: bool = True, topk_explanations: int = 3,
                  concept_names: List = None) -> str:
    """
    Generate a local explanation for a single sample.

    :param model: pytorch model
    :param x: input samples
    :param y: target labels
    :param x_sample: input for which the explanation is required
    :param target_class: class ID
    :param method: local feature importance method
    :param simplify: simplify local explanation
    :param concept_names: list containing the names of the input concepts
    :param device: cpu or cuda device
    :param num_classes: override the number of classes
    :return: Local explanation
    """
    x_validation, y_validation, model = _get_validation_data(x, y, model, target_class, binary)

    class_explanation = ''
    class_explanations = {}
    is_first = True
    for layer_id, module in enumerate(model.children()):
        # analyze only logic layers
        if isinstance(module, XLogic):  # or isinstance(module, XLogicConv2d):

            if is_first:
                prev_module = module
                is_first = False
                feature_names = [f'feature{j:010}' for j in range(prev_module.conceptizator.concepts.size(1))]
                c_validation = prev_module.conceptizator.concepts

            else:
                explanations = []
                for neuron in range(module.conceptizator.concepts.size(1)):
                    if module.top and not binary and neuron != target_class:
                        continue

                    local_explanations = []
                    local_explanations_raw = {}

                    neuron_concepts = module.conceptizator.concepts[:, neuron] > module.conceptizator.threshold
                    if module.top and binary:
                        neuron_concepts = neuron_concepts.eq(target_class)
                    elif module.top and not binary:
                        neuron_concepts = module.conceptizator.concepts.argmax(dim=1).eq(target_class)
                    neuron_list = torch.nonzero(neuron_concepts)

                    for i in neuron_list:
                        if module.top:
                            simplify = True
                            y_target = y_validation.eq(target_class)
                        else:
                            simplify = False
                            y_target = neuron_concepts
                        local_explanation, local_explanation_raw = _local_explanation(prev_module, feature_names, i,
                                                                                      local_explanations_raw,
                                                                                      c_validation, y_target,
                                                                                      target_class, simplify)

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

    class_explanation, class_explanations = _replace_names_dict(class_explanation, class_explanations,
                                                                concept_names)

    return class_explanation[1:-1], class_explanations


def _simplify_formula(explanation: str, x: torch.Tensor, y: torch.Tensor, target_class: int) -> str:
    """
    Simplify formula to a simpler one that is still coherent.

    :param explanation: local formula to be simplified.
    :param x: input data.
    :param y: target labels (1D, categorical NOT one-hot encoded).
    :param target_class: target class
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
            if accuracy == 1.:
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
            aggregated_explanation_simplified = simplify_logic(aggregated_explanation, 'dnf', force=False)
            aggregated_explanation_simplified = f'({aggregated_explanation_simplified})'
        else:
            aggregated_explanation_simplified = ''
    return aggregated_explanation_simplified


def _local_explanation(prev_module, feature_names, neuron_id, neuron_explanations_raw,
                       c_validation, y_target, target_class, simplify):
    # explanation is the conjunction of non-pruned features
    explanation_raw = ''
    for j in torch.nonzero(prev_module.weight.sum(axis=0)):
        if feature_names[j[0]] not in ['()', '']:
            if explanation_raw:
                explanation_raw += ' & '
            if prev_module.conceptizator.concepts[neuron_id, j[0]] > prev_module.conceptizator.threshold:
                explanation_raw += feature_names[j[0]]
            else:
                explanation_raw += f'~{feature_names[j[0]]}'

    if explanation_raw:
        explanation_raw = simplify_logic(explanation_raw, 'dnf', force=True)
    explanation_raw = str(explanation_raw)
    if explanation_raw in ['', 'False', 'True', '(False)', '(True)']:
        return None, None

    if explanation_raw in neuron_explanations_raw:
        explanation = neuron_explanations_raw[explanation_raw]
    elif simplify:
        # accuracy, _ = test_explanation(explanation_raw, target_class, c_validation, y_target, metric=accuracy_score)
        explanation = _simplify_formula(explanation_raw, c_validation, y_target, target_class)
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


def _get_validation_data(x, y, model, target_class, binary=True):
    y = to_categorical(y)
    assert (y == target_class).any(), "Cannot get explanation if target class is not amongst target labels"

    # # collapse samples having the same boolean values and class label different from the target class
    # w, b = collect_parameters(model, device)
    # feature_weights = w[0]
    # feature_used_bool = np.sum(np.abs(feature_weights), axis=0) > 0
    # feature_used = np.sort(np.nonzero(feature_used_bool)[0])
    # _, idx = np.unique((x[:, feature_used][y == target_class] >= 0.5).cpu().detach().numpy(), axis=0, return_index=True)
    threshold = 0.5
    _, idx = np.unique((x[y == target_class] >= threshold).cpu().detach().numpy(), axis=0, return_index=True)
    x_target = x[y == target_class][idx]
    y_target = y[y == target_class][idx]
    # x_target = x[y == target_class]
    # y_target = y[y == target_class]
    # print(len(y_target))

    # get model's predictions
    preds = model(x_target)
    if binary:
        preds = preds.unsqueeze(-1)
    preds = to_categorical(preds)

    # identify samples correctly classified of the target class
    correct_mask = y_target.eq(preds)
    x_target_correct = x_target[correct_mask]
    y_target_correct = y_target[correct_mask]

    # collapse samples having the same boolean values and class label different from the target class
    _, idx = np.unique((x[y != target_class] > threshold).cpu().detach().numpy(), axis=0, return_index=True)
    x_reduced_opposite = x[y != target_class][idx]
    y_reduced_opposite = y[y != target_class][idx]
    preds_opposite = model(x_reduced_opposite)
    if len(preds_opposite.squeeze(-1).shape) > 1:
        preds_opposite = torch.argmax(preds_opposite, dim=1)
    else:
        preds_opposite = (preds_opposite > 0.5).squeeze()

    # identify samples correctly classified of the opposite class
    correct_mask = y_reduced_opposite.eq(preds_opposite)
    x_reduced_opposite_correct = x_reduced_opposite[correct_mask]
    y_reduced_opposite_correct = y_reduced_opposite[correct_mask]

    # select the subset of samples belonging to the target class
    x_validation = torch.cat([x_reduced_opposite_correct, x_target_correct], dim=0)
    y_validation = torch.cat([y_reduced_opposite_correct, y_target_correct], dim=0)

    model.train()
    model(x_validation)
    return x_validation, y_validation, model
