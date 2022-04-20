from typing import List, Tuple

import sympy

import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score, accuracy_score, homogeneity_score
from sklearn_extra.cluster import KMedoids
from sympy import to_dnf, lambdify


def auc_truth_table_score(c_vec: np.array, y_pred: np.array, y_test: np.array,
                          step: int = 100) -> [int, np.array, np.array]:
    """
    Computes the AUC of the truth table from concepts to tasks.

    :param c_vec: predicted concept representations (can be concept embeddings)
    :param y_pred: task predictions
    :param y_test: task ground truth labels
    :param step: integration step
    :return: AUC score, cluster sizes and corresponding accuracies
    """
    n_clusters = np.arange(2, len(c_vec), step)
    accuracy_list = []
    for nc in n_clusters:
        kmedoids = KMedoids(n_clusters=nc, random_state=0)
        c_cluster_labels = kmedoids.fit_predict(c_vec)
        y_center_labels = y_pred[kmedoids.medoid_indices_] > 0.5
        min_dist = cdist(c_vec, kmedoids.cluster_centers_).argmin(axis=1)
        y_cluster_labels = np.array([y_center_labels[md] for md in min_dist])
        accuracy_list.append(accuracy_score(y_test, y_cluster_labels))

    accuracies = np.array(accuracy_list)
    max_auc = np.trapz(np.ones(len(n_clusters)))
    auc_score = np.trapz(accuracies) / max_auc
    return auc_score, n_clusters, np.array(accuracy_list)


def embedding_homogeneity(c_vec: np.array, c_test: np.array, y_test: np.array, step: int) -> [float, float]:
    """
    Computes the alignment between learnt concepts and labels.

    :param c_vec: predicted concept representations (can be concept embeddings)
    :param c_test: concept ground truth labels
    :param y_test: task ground truth labels
    :param step: integration step
    :return: concept alignment AUC, task alignment AUC
    """
    # compute the maximum value for the AUC
    n_clusters = np.arange(2, len(c_vec), step)
    max_auc = np.trapz(np.ones(len(n_clusters)))

    # for each concept:
    #   1. find clusters
    #   2. compare cluster assignments with ground truth concept/task labels
    concept_auc, task_auc = [], []
    for concept_id in range(c_test.shape[1]):
        concept_homogeneity, task_homogeneity = [], []
        for nc in n_clusters:
            # train clustering algorithm
            kmedoids = KMedoids(n_clusters=nc, random_state=0)
            if c_vec.shape[1] != c_test.shape[1]:
                c_cluster_labels = kmedoids.fit_predict(c_vec)
            elif c_vec.shape[1] == c_test.shape[1] and len(c_vec.shape) == 2:
                c_cluster_labels = kmedoids.fit_predict(c_vec[:, concept_id].reshape(-1, 1))
            else:
                c_cluster_labels = kmedoids.fit_predict(c_vec[:, concept_id])

            # compute alignment with ground truth labels
            concept_homogeneity.append(homogeneity_score(c_test[:, concept_id], c_cluster_labels))
            task_homogeneity.append(homogeneity_score(y_test, c_cluster_labels))

        # compute the area under the curve
        concept_auc.append(np.trapz(np.array(concept_homogeneity)) / max_auc)
        task_auc.append(np.trapz(np.array(task_homogeneity)) / max_auc)

    # return the average alignment across all concepts
    concept_auc = np.mean(concept_auc)
    task_auc = np.mean(task_auc)
    return concept_auc, task_auc


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
