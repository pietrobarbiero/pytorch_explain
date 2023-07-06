from itertools import product
import torch
from sklearn import datasets
import numpy as np
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

    # in the end we want something like this:
    # q_names = ['q(0)', 'q(1)', 'q(2)', 'r(0,1)', 'r(1,2)', 'r(2,0)']
    # q_labels = torch.tensor([0, 1, 0, 0, 1, 1], dtype=torch.long)

    q_names = {'concepts': [], 'tasks': []}
    q_labels = []

    # Add the query (task label) for each point
    for i in range(len(X)):
        q_names['tasks'].append(f'q({i})')
        q_labels.append(y[i])

    # Add the query (relation label) for each pair of points
    for i1, i2 in product(range(len(X)), range(len(X))):
        x1 = X[i1]
        x2 = X[i2]
        dist = np.linalg.norm( x1 - x2)
        is_close_x1_x2 = int(dist < threshold)

        if only_on_manifold:
            if is_close_x1_x2:
                q_labels.append(is_close_x1_x2)
                q_names['concepts'].append(f'r({i1},{i2})')
        else:
            q_labels.append(is_close_x1_x2)
            q_names['concepts'].append(f'r({i1},{i2})')

    X = torch.tensor(X, dtype=torch.float).unsqueeze(0)
    q_labels = torch.tensor(q_labels, dtype=torch.long).unsqueeze(1).unsqueeze(0)
    return X, q_labels, q_names
