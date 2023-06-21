from datasets import manifold_toy_dataset
from model import ManifoldRelationalDCR

X, y, body_index, head_index, task_labels = manifold_toy_dataset("moon")


m = ManifoldRelationalDCR(input_features=2, emb_size=3, manifold_arity=2, num_classes=2)

c, t = m(X,body_index, head_index)
print(c == y)
print(t == task_labels)