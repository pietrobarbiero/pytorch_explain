import copy
from typing import List, Tuple, Dict

import torch
import numpy as np
from sympy import simplify_logic
from torch.nn.functional import one_hot

from torch_explain.logic.metrics import test_explanation, complexity
from torch_explain.logic.utils import replace_names
from torch_explain.logic.utils import get_predictions
from torch_explain.logic.utils import get_the_good_and_bad_terms
from torch_explain.nn import Conceptizator
from torch_explain.nn.logic import EntropyLinear


def explain_classes(model: torch.nn.Module, c: torch.Tensor, y: torch.Tensor,
                    train_mask: torch.Tensor, test_mask: torch.Tensor, val_mask: torch.Tensor = None,
                    edge_index: torch.Tensor = None, max_minterm_complexity: int = 1000,
                    topk_explanations: int = 1000, try_all: bool = False,
                    c_threshold: float = 0.5, y_threshold: float = 0.,
                    concept_names: List[str] = None, class_names: List[str] = None,
                    verbose: bool = False) -> Dict:
    """
    Explain LENs predictions with concept-based logic explanations.

    :param model: pytorch model
    :param c: input concepts
    :param y: target labels
    :param train_mask: train mask
    :param test_mask: test mask
    :param val_mask: validation mask
    :param edge_index: edge index for graph data used in graph-based models
    :param max_minterm_complexity: maximum number of concepts per logic formula (per sample)
    :param topk_explanations: number of local explanations to be combined
    :param try_all: if True, then tries all possible conjunctions of the top k explanations
    :param c_threshold: threshold to get truth values for concept predictions (i.e. pred<threshold = false, pred>threshold = true)
    :param y_threshold: threshold to get truth values for class predictions (i.e. pred<threshold = false, pred>threshold = true)
    :param concept_names: list of concept names
    :param class_names: list of class names
    :param verbose: if True, then prints the explanations
    :return: Global explanations
    """
    if len(y.shape) == 1:
        y = one_hot(y)

    if val_mask is None:
        val_mask = train_mask

    explanations = {}
    for class_id in range(y.shape[1]):
        explanation, _ = explain_class(model, c, y, train_mask, val_mask, target_class=class_id,
                                       edge_index=edge_index, max_minterm_complexity=max_minterm_complexity,
                                       topk_explanations=topk_explanations, try_all=try_all,
                                       c_threshold=c_threshold, y_threshold=y_threshold)

        explanation_accuracy, _ = test_explanation(explanation, c, y, class_id, test_mask, c_threshold)
        explanation_complexity = complexity(explanation)

        explanations[str(class_id)] = {'explanation': explanation,
                                       'name': str(class_id),
                                       'explanation_accuracy': explanation_accuracy,
                                       'explanation_complexity': explanation_complexity}

        if concept_names is not None and class_names is not None:
            explanations[str(class_id)]['explanation'] = replace_names(explanations[str(class_id)]['explanation'],
                                                                 concept_names)
            explanations[str(class_id)]['name'] = class_names[class_id]

        if verbose:
            print(f'Explanation class {explanations[str(class_id)]["name"]}: '
                  f'{explanations[str(class_id)]["explanation"]} - '
                  f'acc. = {explanation_accuracy:.4f} - '
                  f'compl. = {explanation_complexity:.4f}')

    return explanations


def explain_class(model: torch.nn.Module, c: torch.Tensor, y: torch.Tensor,
                  train_mask: torch.Tensor, val_mask: torch.Tensor, target_class: int, edge_index: torch.Tensor = None,
                  max_minterm_complexity: int = None, topk_explanations: int = 3, max_accuracy: bool = False,
                  concept_names: List = None, try_all: bool = True, c_threshold: float = 0.5,
                  y_threshold: float = 0.) -> Tuple[str, str]:
    """
    Generate a local explanation for a single sample.
    :param model: pytorch model
    :param c: input concepts
    :param y: target labels (MUST be one-hot encoded)
    :param train_mask: train mask
    :param val_mask: validation mask
    :param target_class: target class
    :param edge_index: edge index for graph data used in graph-based models
    :param max_minterm_complexity: maximum number of concepts per logic formula (per sample)
    :param topk_explanations: number of local explanations to be combined
    :param max_accuracy: if True a formula is simplified only if the simplified formula gets 100% accuracy
    :param concept_names: list containing the names of the input concepts
    :param try_all: if True, then tries all possible conjunctions of the top k explanations
    :param c_threshold: threshold to get truth values for concept predictions (i.e. pred<threshold = false, pred>threshold = true)
    :param y_threshold: threshold to get truth values for class predictions (i.e. pred<threshold = false, pred>threshold = true)
    :return: Global explanation
    """
    c_correct, y_correct, correct_mask, active_mask = _get_correct_data(c, y, train_mask, model, target_class,
                                                                        edge_index, y_threshold)
    if c_correct is None:
        return '', ''

    feature_names = [f"feature{j:010}" for j in range(c_correct.size(1))]
    class_explanation = ""
    class_explanation_raw = ""
    for layer_id, module in enumerate(model.children()):
        if isinstance(module, EntropyLinear):
            local_explanations = []
            local_explanations_accuracies = {}
            local_explanations_raw = {}

            idx_and_exp = []

            # look at the "positive" rows of the truth table only
            positive_samples = torch.nonzero(y_correct[:, target_class]).numpy().ravel()
            for positive_sample in positive_samples:
                local_explanation, local_explanation_raw = _local_explanation(
                    module,
                    feature_names,
                    positive_sample,
                    local_explanations_raw,
                    c_correct,
                    y_correct,
                    target_class,
                    max_accuracy,
                    max_minterm_complexity,
                    c_threshold=c_threshold,
                    y_threshold=y_threshold,
                    simplify=False,
                )

                if local_explanation and local_explanation_raw and local_explanation_raw not in local_explanations_raw:
                    idx_and_exp.append((positive_sample, local_explanation_raw))
                    local_explanations_raw[
                        local_explanation_raw
                    ] = local_explanation_raw
                    local_explanations.append(local_explanation)

            for positive_sample, local_explanation_raw in idx_and_exp:
                sample_pos = train_mask[positive_sample]
                good, bad = get_the_good_and_bad_terms(
                    model=model,
                    c=c,
                    edge_index=edge_index,
                    sample_pos=sample_pos,
                    explanation=local_explanation_raw,
                    target_class=target_class,
                    concept_names=feature_names,
                    threshold=c_threshold
                )

                local_explanation_raw = " & ".join(good)

                # test explanation accuracy
                if local_explanation_raw not in local_explanations_accuracies:
                    accuracy, _ = test_explanation(
                        local_explanation_raw, c, y, target_class, val_mask, c_threshold
                    )
                    local_explanations_accuracies[local_explanation_raw] = accuracy

            # aggregate local explanations and replace concept names in the final formula
            if try_all:
                aggregated_explanation, best_acc = _aggregate_explanations_try_all(
                    local_explanations_accuracies,
                    topk_explanations,
                    target_class,
                    c,
                    y,
                    max_accuracy,
                    val_mask,
                    c_threshold
                )
            else:
                aggregated_explanation, best_acc = _aggregate_explanations(
                    local_explanations_accuracies,
                    topk_explanations,
                    target_class,
                    c,
                    y,
                    max_accuracy,
                    val_mask,
                    c_threshold
                )
            class_explanation_raw = str(aggregated_explanation)
            class_explanation = class_explanation_raw
            if concept_names is not None:
                class_explanation = replace_names(class_explanation, concept_names)

            break

    return class_explanation[1:-1], class_explanation_raw


def _simplify_formula(
        explanation: str,
        x: torch.Tensor,
        y: torch.Tensor,
        target_class: int,
        max_accuracy: bool,
        mask: torch.Tensor = None,
        c_threshold: float = 0.5,
        y_threshold: float = 0.
) -> str:
    """
    Simplify formula to a simpler one that is still coherent.
    :param explanation: local formula to be simplified.
    :param x: input data.
    :param y: target labels (1D, categorical NOT one-hot encoded).
    :param target_class: target class
    :param max_accuracy: drop  term only if it gets max accuracy
    :param c_threshold: threshold to get truth values for concept predictions (i.e. pred<threshold = false, pred>threshold = true)
    :param y_threshold: threshold to get truth values for class predictions (i.e. pred<threshold = false, pred>threshold = true)
    :return: Simplified formula
    """

    base_accuracy, _ = test_explanation(explanation, x, y, target_class, mask, c_threshold)
    for term in explanation.split(" & "):
        explanation_simplified = copy.deepcopy(explanation)

        if explanation_simplified.endswith(f"{term}"):
            explanation_simplified = explanation_simplified.replace(f" & {term}", "")
        else:
            explanation_simplified = explanation_simplified.replace(f"{term} & ", "")

        if explanation_simplified:
            accuracy, preds = test_explanation(
                explanation_simplified, x, y, target_class, mask, c_threshold
            )
            if (max_accuracy and accuracy == 1.0) or (
                    not max_accuracy and accuracy >= base_accuracy
            ):
                explanation = copy.deepcopy(explanation_simplified)
                base_accuracy = accuracy

    return explanation


def _aggregate_explanations(
        local_explanations_accuracy, topk_explanations, target_class, x, y, max_accuracy, val_mask, threshold
):
    """
    Sort explanations by accuracy and then aggregate explanations which increase the accuracy of the aggregated formula.
    :param local_explanations_accuracy: dictionary of explanations and related accuracies.
    :param topk_explanations: limits the number of explanations to be aggregated.
    :param target_class: target class.
    :param x: observations in validation set.
    :param y: labels in validation set.
    :param max_accuracy: if True a formula is simplified only if the simplified formula gets 100% accuracy.
    :return:
    """
    if len(local_explanations_accuracy) == 0:
        return ""

    else:
        # get the topk most accurate local explanations
        local_explanations_sorted = sorted(
            local_explanations_accuracy.items(), key=lambda x: -x[1]
        )[:topk_explanations]
        explanations = []
        best_accuracy = 0
        best_explanation = ""
        for explanation_raw, accuracy in local_explanations_sorted:
            explanation = _simplify_formula(explanation_raw, x, y, target_class, max_accuracy, val_mask, threshold)
            if not explanation:
                continue

            explanations.append(explanation)

            # aggregate example-level explanations
            aggregated_explanation = " | ".join(explanations)
            aggregated_explanation_simplified = simplify_logic(
                aggregated_explanation, "dnf", force=True
            )
            aggregated_explanation_simplified = f"({aggregated_explanation_simplified})"

            if aggregated_explanation_simplified in [
                "",
                "False",
                "True",
                "(False)",
                "(True)",
            ]:
                continue
            accuracy, _ = test_explanation(
                aggregated_explanation_simplified, x, y, target_class, val_mask, threshold
            )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_explanation = aggregated_explanation_simplified
                explanations = [best_explanation]

    return best_explanation, best_accuracy


def _aggregate_explanations_try_all(
        local_explanations_accuracy, topk_explanations, target_class, x, y, max_accuracy, val_mask, c_threshold
):
    """
    Sort explanations by accuracy and then aggregate explanations which increase the accuracy of the aggregated formula.
    :param local_explanations_accuracy: dictionary of explanations and related accuracies.
    :param topk_explanations: limits the number of explanations to be aggregated.
    :param target_class: target class.
    :param x: observations in validation set.
    :param y: labels in validation set.
    :param max_accuracy: if True a formula is simplified only if the simplified formula gets 100% accuracy.
    :return:
    """
    if len(local_explanations_accuracy) == 0:
        return ""

    else:
        # get the topk most accurate local explanations
        local_explanations_sorted = sorted(
            local_explanations_accuracy.items(), key=lambda x: -x[1]
        )[:topk_explanations]
        predictions = []
        explanations = []

        best_accuracy = 0.0
        best_explanation = ""

        for explanation_raw, accuracy in local_explanations_sorted:
            explanation = _simplify_formula(explanation_raw, x, y, target_class, max_accuracy, val_mask, c_threshold)
            if not explanation:
                continue

            predictions.append(get_predictions(explanation, x, target_class, c_threshold))
            explanations.append(explanation)

        predictions = np.array(predictions)
        explanations = np.array(explanations)

        y = y[:, target_class]
        y = y.detach().numpy()

        for i in range(1, 1 << len(predictions)):
            include = i & (1 << np.arange(len(predictions))) > 0
            pred = predictions[np.nonzero(include)]
            pred = np.sum(pred, axis=0)
            pred = pred > 0.5

            accuracy = np.sum(pred == y)
            if accuracy > best_accuracy:
                # aggregate example-level explanations
                explanation = explanations[np.nonzero(include)]
                aggregated_explanation = " | ".join(explanation)
                aggregated_explanation_simplified = simplify_logic(
                    aggregated_explanation, "dnf", force=True
                )
                aggregated_explanation_simplified = (
                    f"({aggregated_explanation_simplified})"
                )

                if aggregated_explanation_simplified in [
                    "",
                    "False",
                    "True",
                    "(False)",
                    "(True)",
                ]:
                    continue
                else:
                    best_accuracy = accuracy
                    best_explanation = aggregated_explanation_simplified

    return best_explanation, best_accuracy


def _local_explanation(
        module,
        feature_names,
        neuron_id,
        neuron_explanations_raw,
        c_validation,
        y_target,
        target_class,
        max_accuracy,
        max_minterm_complexity,
        c_threshold=0.5,
        y_threshold=0.,
        simplify=True,
):
    # explanation is the conjunction of non-pruned features
    explanation_raw = ""
    if max_minterm_complexity:
        concepts_to_retain = torch.argsort(module.alpha[target_class], descending=True)[
                             :max_minterm_complexity
                             ]
    else:
        non_pruned_concepts = module.concept_mask[target_class]
        concepts_sorted = torch.argsort(module.alpha[target_class])
        concepts_to_retain = concepts_sorted[non_pruned_concepts[concepts_sorted]]

    for j in concepts_to_retain:
        if feature_names[j] not in ["()", ""]:
            if explanation_raw:
                explanation_raw += " & "
            if c_validation[neuron_id, j] > c_threshold:
                # if non_pruned_neurons[j] > 0:
                explanation_raw += feature_names[j]
            else:
                explanation_raw += f"~{feature_names[j]}"

    explanation_raw = str(explanation_raw)
    if explanation_raw in ["", "False", "True", "(False)", "(True)"]:
        return None, None

    if explanation_raw in neuron_explanations_raw:
        explanation = neuron_explanations_raw[explanation_raw]
    elif simplify:
        explanation = _simplify_formula(
            explanation_raw, c_validation, y_target, target_class, max_accuracy, c_threshold, y_threshold
        )
    else:
        explanation = explanation_raw

    if explanation in ["", "False", "True", "(False)", "(True)"]:
        return None, None

    return explanation, explanation_raw


def _get_correct_data(c, y, train_mask, model, target_class, edge_index, threshold=0.):
    active_mask = y[train_mask, target_class] == 1

    # get model's predictions
    if edge_index is None:
        preds = model(c).squeeze(-1)
    else:
        preds = model(c, edge_index).squeeze(-1)

    # identify samples correctly classified of the target class
    correct_mask = y[train_mask, target_class].eq(preds[train_mask, target_class] > threshold)
    if (sum(correct_mask & ~active_mask) < 2) or (sum(correct_mask & active_mask) < 2):
        return None, None, None, None

    # select correct samples from both classes
    c_target_correct = c[train_mask][correct_mask & active_mask]
    y_target_correct = y[train_mask][correct_mask & active_mask]
    c_opposite_correct = c[train_mask][correct_mask & ~active_mask]
    y_opposite_correct = y[train_mask][correct_mask & ~active_mask]

    # merge correct samples in the same dataset
    c_validation = torch.cat([c_opposite_correct, c_target_correct], dim=0)
    y_validation = torch.cat([y_opposite_correct, y_target_correct], dim=0)

    return c_validation, y_validation, correct_mask, active_mask
