import random
from pathlib import Path
from torchvision import transforms as transforms
from itertools import product
import sklearn as skl
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
data_root = Path(__file__).parent / ".." / "data"


def manifold_toy_dataset(name, threshold = 0.5, n_samples = 100, perc_super = 0.2, only_on_manifold = True):

    if name == "moon":
        X, y = datasets.make_moons(n_samples)
    else:
        raise Exception("Dataset %s unknowns." % name)


    supervised = random.sample(range(len(X)), int(len(X)*perc_super))
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

    return X,y, body_index, head_index, relation_labels, task_labels

if __name__ == '__main__':
    X,y, body_index, head_index, relation_labels, task_labels =  manifold_toy_dataset("moon")
    plt.scatter(X[:,0], X[:,1])
    plt.show()