import torch
import matplotlib.pyplot as plt
from datasets import manifold_toy_dataset
from model import ManifoldRelationalDCR
import numpy as np

# Only groundings where the relation is true are given
X, y, body_index, head_index, relation_labels, task_labels = manifold_toy_dataset("moon", only_on_manifold=True)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
m = ManifoldRelationalDCR(input_features=2, emb_size=3, manifold_arity=2, num_classes=2, predict_relation=False)

X = torch.tensor(X, dtype=torch.float)
y, body_index, head_index, relation_labels, task_labels = (torch.tensor(i) for i in (y, body_index, head_index, relation_labels, task_labels))
c, t = m(X,body_index, head_index)
# Losses should match the following vectors (now dummy comparison, just for shape)
print(c == torch.eye(2)[y])
print(t == torch.eye(2)[task_labels])



# All the groundings are given and we predict the relation as well (and we use it in reasoning)
X, y, body_index, head_index, relation_labels, task_labels = manifold_toy_dataset("moon", only_on_manifold=False)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
m = ManifoldRelationalDCR(input_features=2, emb_size=3, manifold_arity=2, num_classes=2, predict_relation=True)

X = torch.tensor(X, dtype=torch.float)
y, body_index, head_index, relation_labels, task_labels = (torch.tensor(i) for i in (y, body_index, head_index, relation_labels, task_labels))
c, r, t = m(X,body_index, head_index)
# Losses should match the following vectors (now dummy comparison, just for shape)
print(c == torch.eye(2)[y])
print(t == torch.eye(2)[task_labels])
print(r == np.reshape(relation_labels, [-1,1]))