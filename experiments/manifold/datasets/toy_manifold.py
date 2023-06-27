from itertools import product
import sklearn as skl
import torch
from sklearn import datasets
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import TensorDataset


def manifold_toy_dataset(name, threshold = 0.5, n_samples = 100, perc_super = 0.2,
                         only_on_manifold = True, random_seed = 42, train = True):

    if name == "moon":
        X, y = datasets.make_moons(n_samples, random_state=random_seed)
    else:
        raise Exception("Dataset %s unknowns." % name)

    if train:
        supervised = np.random.RandomState(random_seed).choice(range(len(X)), int(len(X)*perc_super))
        X = X[supervised]
        y = y[supervised]
    groundings = []
    task_labels = []
    relation_labels = []
    for i1, i2 in product(range(len(X)), range(len(X))):
        groundings.append((i1,i2))
        x1 = X[i1]
        x2 = X[i2]
        dist = np.linalg.norm( x1 - x2)
        task_labels.append(y[i2])
        relation_labels.append(int(dist < threshold))
    groundings = np.array(groundings)
    task_labels = np.array(task_labels)
    relation_labels = np.array(relation_labels)
    if only_on_manifold:
        groundings = groundings[relation_labels == 1]
        task_labels = task_labels[relation_labels == 1]

    body_index = groundings[:, 0:1]
    head_index = groundings[:, 1:2]

    X = torch.tensor(X, dtype=torch.float)
    y, body_index, head_index, relation_labels, task_labels = (torch.tensor(i) for i in (y, body_index, head_index, relation_labels, task_labels))
    c_train = F.one_hot(y.long().ravel()).float()
    y_train = F.one_hot(task_labels.long().ravel()).float()
    X, c_train, body_index, head_index, relation_labels, y_train = (i.unsqueeze(0) for i in (X, c_train, body_index, head_index, relation_labels, y_train))
    return TensorDataset(X, c_train, body_index, head_index, relation_labels, y_train)
