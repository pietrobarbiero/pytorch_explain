import os

import torch
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as p
from sklearn.manifold import TSNE

from experiments.data.load_datasets import load_vector_mnist
from experiments.vlens.mnist_sum import ResNetMNISTsum
from torch_explain.nn import semantics, to_boolean, context


def main():
    train_data, test_data, concept_names, label_names = load_vector_mnist('../data')

    result_dir = './results/mnist_sum/'
    model = ResNetMNISTsum.load_from_checkpoint(os.path.join(result_dir, 'model.pt'), map_location="cuda")
    model.freeze()

    c1 = train_data.tensors[2]
    c2 = train_data.tensors[3]
    y = train_data.tensors[4]
    c1_emb, c2_emb, c_emb, y_emb = model(train_data.tensors[0], train_data.tensors[1])
    c1_logits = semantics(c1_emb)
    c2_logits = semantics(c2_emb)
    y_logits = semantics(y_emb)

    # compute accuracy
    c_pred = torch.concat([c1_logits.reshape(-1), c2_logits.reshape(-1)]).cpu().detach() > 0.5
    y_pred = y_logits.reshape(-1).cpu().detach() > 0.5
    c_true = torch.concat([c1.reshape(-1), c2.reshape(-1)]).cpu().detach()
    y_true = y.reshape(-1).cpu().detach()
    c_accuracy = accuracy_score(c_true, c_pred)
    y_accuracy = accuracy_score(y_true, y_pred)
    c_f1 = f1_score(c_true > 0.5, c_pred)
    y_f1 = f1_score(y_true, y_pred)
    print(f'c_f1: {c_f1:.4f}, y_f1: {y_f1:.4f}, c_acc: {c_accuracy:.4f}, y_acc: {y_accuracy:.4f}, ')

    # emb_a = context(c_emb[i, na])
    c_emb_flat = torch.concat([c1_emb.reshape(-1, 10), c2_emb.reshape(-1, 10)]).cpu().detach()
    c_emb_flat = c_emb_flat[c_true.bool()]
    c_flat = torch.concat([c1.argmax(dim=-1), c2.argmax(dim=-1)]).cpu().detach()

    c_emb_flat = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(c_emb_flat)

    sns.set_style("whitegrid")
    plt.figure()
    plt.title("MNIST-Sum Concepts' Context")
    sns.scatterplot(c_emb_flat[:, 0], c_emb_flat[:, 1], hue=c_flat, palette='colorblind')
    plt.xlabel('TSNE dim #1')
    plt.ylabel('TSNE dim #2')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'mnist_sum_context.png'))
    plt.savefig(os.path.join(result_dir, 'mnist_sum_context.pdf'))
    plt.show()


if __name__ == '__main__':
    main()
