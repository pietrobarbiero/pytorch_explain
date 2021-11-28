import os

import torch
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as p

from experiments.data.load_datasets import load_vector_mnist
from experiments.vlens.mnist_sum import ResNetMNISTsum
from torch_explain.nn import semantics, to_boolean, context


def main():
    train_data, test_data, concept_names, label_names = load_vector_mnist('../data')

    result_dir = './results/mnist_sum/'
    model = ResNetMNISTsum.load_from_checkpoint(os.path.join(result_dir, 'model.pt'),
                                                map_location="cuda")
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
    print(f'c_f1: {c_f1:.4f}, y_f1: {y_f1:.4f}, '
          f'c_acc: {c_accuracy:.4f}, y_acc: {y_accuracy:.4f}, ')

    #
    c_emb_norm = to_boolean(c_emb, true_norm=1, false_norm=2)
    i = 0
    nab = torch.argmax((y_logits[i] > 0.5).float())
    na = torch.argmax(c_emb_norm[i, :10].norm(dim=1))
    nb = torch.argmax(c_emb_norm[i, 10:].norm(dim=1))
    emb_ab = context(y_emb[i, nab]) * 2
    emb_a = context(c_emb_norm[i, na]) * 2
    emb_b = context(c_emb_norm[i, nb]) * 2
    emb_not_ab = context(y_emb[i])

    sns.set_style("whitegrid")

    plt.figure(figsize=[5, 5])
    plt.title(f'is{nab}(img1,img2) <- is{na}(img1), is{nb}(img2)')
    sns.scatterplot(emb_not_ab[:, 0], emb_not_ab[:, 1], label=f'isX(img1,img2), X!={nab}')
    sns.scatterplot([emb_ab[0]], [emb_ab[1]], label=f'is{nab}(img1,img2)')
    sns.scatterplot([emb_a[0]], [emb_a[1]], label=f'is{na}(img1)')
    sns.scatterplot([emb_b[0]], [emb_b[1]], label=f'is{nb}(img2)')

    p.arrow(0, 0, emb_ab[0], emb_ab[1], fc="k", ec="k", head_width=0.05, head_length=0)
    p.arrow(0, 0, emb_a[0], emb_a[1], fc="k", ec="k", head_width=0.05, head_length=0)
    p.arrow(0, 0, emb_b[0], emb_b[1], fc="k", ec="k", head_width=0.05, head_length=0)

    plt.xlabel('Emb dim #1')
    plt.ylabel('Emb dim #2')
    plt.xlim([-2.5, 2.5])
    plt.ylim([-2.5, 2.5])
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'mnist_sum.png'))
    plt.savefig(os.path.join(result_dir, 'mnist_sum.pdf'))
    plt.show()


if __name__ == '__main__':
    main()
