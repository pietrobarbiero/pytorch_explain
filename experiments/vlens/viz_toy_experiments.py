import copy
import joblib
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from torch.nn import BCELoss, BCEWithLogitsLoss, Sequential, LeakyReLU, Linear
from torch.utils.data import DataLoader, TensorDataset
from torch_explain.nn import ConceptEmbeddings, context, semantics
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch_explain.nn.vector_logic import NeSyLayer, to_boolean


def main():
    # parameters for data, model, and training
    cv = 5
    result_dir = './results/toy_xor/'
    result_dir = './results/toy_trigonometry/'
    result_dir = './results/toy_vectors_hard/'
    result_dir = './results/toy_vectors_hard_2/'

    results = joblib.load(os.path.join(result_dir, f'results.joblib'))

    n_epochs = len(results['0']['train_acc_bool'])
    c_train_loss = []
    y_train_loss = []
    c_test_acc = []
    y_test_acc = []
    y_test_acc_full = []
    for split in range(cv):
        # train accuracy
        for epoch in range(n_epochs):
            # concept
            c_train_loss.append(['Embeddings', split, epoch, results[f'{split}']['train_acc_emb'][epoch][0]])
            c_train_loss.append(['Fuzzy', split, epoch, results[f'{split}']['train_acc_fuzzy'][epoch][0]])
            c_train_loss.append(['Boolean', split, epoch, results[f'{split}']['train_acc_bool'][epoch][0]])
            # tasks
            y_train_loss.append(['Embeddings', split, epoch, results[f'{split}']['train_acc_emb'][epoch][1]])
            y_train_loss.append(['Fuzzy', split, epoch, results[f'{split}']['train_acc_fuzzy'][epoch][1]])
            y_train_loss.append(['Boolean', split, epoch, results[f'{split}']['train_acc_bool'][epoch][1]])
        
        # true labels
        c_train = results[f'{split}']['c_train']
        y_train = results[f'{split}']['y_train']
        c_test = results[f'{split}']['c_test']
        y_test = results[f'{split}']['y_test']
        
        # test accuracy
        c_accuracy_bool, y_accuracy_bool = compute_accuracy(results[f'{split}']['c_bool_test'],
                                                            results[f'{split}']['y_bool_test'],
                                                            c_test, y_test)
        c_accuracy_fuzzy, y_accuracy_fuzzy = compute_accuracy(results[f'{split}']['c_fuzzy_test'],
                                                              results[f'{split}']['y_fuzzy_test'],
                                                              c_test, y_test)
        c_accuracy_emb, y_accuracy_emb = compute_accuracy(semantics(results[f'{split}']['c_emb_test']),
                                                          torch.exp(-torch.norm(results[f'{split}']['y_emb_test'], p=2, dim=-1)),
                                                          c_test, y_test)
        c_test_acc.append(['Embeddings', split, c_accuracy_emb])
        c_test_acc.append(['Fuzzy', split, c_accuracy_fuzzy])
        c_test_acc.append(['Boolean', split, c_accuracy_bool])
        y_test_acc.append(['Embeddings', split, y_accuracy_emb])
        y_test_acc.append(['Fuzzy', split, y_accuracy_fuzzy])
        y_test_acc.append(['Boolean', split, y_accuracy_bool])

        # check the accuracy of context and semantics separately
        clf = DecisionTreeClassifier(random_state=42)
        # fuzzy
        c_bool_train = results[f'{split}']['c_bool_train']
        c_bool_test = results[f'{split}']['c_bool_test']
        c_bool_sem_train = c_bool_train.cpu().detach() > 0.5
        c_bool_sem_test = c_bool_test.cpu().detach() > 0.5
        bool_context_accuracy = clf.fit(c_bool_train, y_train).score(c_bool_test, y_test)
        bool_semantics_accuracy = clf.fit(c_bool_sem_train, y_train).score(c_bool_sem_test, y_test)
        # fuzzy
        c_fuzzy_train = results[f'{split}']['c_fuzzy_train']
        c_fuzzy_test = results[f'{split}']['c_fuzzy_test']
        c_fuzzy_sem_train = c_fuzzy_train.cpu().detach() > 0.5
        c_fuzzy_sem_test = c_fuzzy_test.cpu().detach() > 0.5
        fuzzy_context_accuracy = clf.fit(c_fuzzy_train, y_train).score(c_fuzzy_test, y_test)
        fuzzy_semantics_accuracy = clf.fit(c_fuzzy_sem_train, y_train).score(c_fuzzy_sem_test, y_test)
        # embeddings
        c_emb_train = results[f'{split}']['c_emb_train']
        c_emb_test = results[f'{split}']['c_emb_test']
        c_emb_ctx_train = context(c_emb_train).reshape(c_train.shape[0], -1)
        c_emb_ctx_test = context(c_emb_test).reshape(c_test.shape[0], -1)
        c_emb_sem_train = semantics(c_emb_train).cpu().detach() > 0.5
        c_emb_sem_test = semantics(c_emb_test).cpu().detach() > 0.5
        emb_context_accuracy = clf.fit(c_emb_ctx_train, y_train).score(c_emb_ctx_test, y_test)
        emb_semantics_accuracy = clf.fit(c_emb_sem_train, y_train).score(c_emb_sem_test, y_test)
        emb_ctx_sem_accuracy = clf.fit(c_emb_train.reshape(c_train.shape[0], -1), y_train).score(c_emb_test.reshape(c_test.shape[0], -1), y_test)
        y_test_acc_full.append(['Embeddings', split, 'context', emb_context_accuracy])
        y_test_acc_full.append(['Embeddings', split, 'semantics', emb_semantics_accuracy])
        y_test_acc_full.append(['Embeddings', split, 'context+semantics', emb_ctx_sem_accuracy])
        y_test_acc_full.append(['Fuzzy', split, 'context', fuzzy_context_accuracy])
        y_test_acc_full.append(['Fuzzy', split, 'semantics', fuzzy_semantics_accuracy])
        y_test_acc_full.append(['Fuzzy', split, 'context+semantics', fuzzy_context_accuracy])
        y_test_acc_full.append(['Boolean', split, 'context', bool_context_accuracy])
        y_test_acc_full.append(['Boolean', split, 'semantics', bool_semantics_accuracy])
        y_test_acc_full.append(['Boolean', split, 'context+semantics', bool_context_accuracy])

    c_train_loss = pd.DataFrame(c_train_loss, columns=['Representation', 'cv_split', 'epoch', 'train accuracy'])
    y_train_loss = pd.DataFrame(y_train_loss, columns=['Representation', 'cv_split', 'epoch', 'train accuracy'])

    c_test_acc = pd.DataFrame(c_test_acc, columns=['Representation', 'cv_split', 'test accuracy'])
    y_test_acc = pd.DataFrame(y_test_acc, columns=['Representation', 'cv_split', 'test accuracy'])

    y_test_acc_full = pd.DataFrame(y_test_acc_full, columns=['Representation', 'cv_split', 'part', 'test accuracy'])

    sns.set_style('whitegrid')
    figsize = [6, 3]

    # plot test concept accuracy
    plt.figure(figsize=figsize)
    plt.title(f'Concept Accuracy')
    sns.boxenplot(data=c_test_acc, x='Representation', y='test accuracy')
    plt.ylim([0.7, 1.0])
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'c_accuracy_test.png'))
    plt.show()

    # plot test task accuracy
    plt.figure(figsize=figsize)
    plt.title(f'Task Accuracy')
    sns.boxenplot(data=y_test_acc, x='Representation', y='test accuracy')
    plt.ylim([0.7, 1.0])
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'y_accuracy_test.png'))
    plt.show()

    # plot test concept accuracy full
    plt.figure(figsize=figsize)
    plt.title(f'Isolated Task Accuracy (with Decision Tree)')
    g = sns.boxenplot(data=y_test_acc_full, x='part', y='test accuracy', hue='Representation')
    plt.ylim([0.7, 1.0])
    plt.xlabel('')
    g.legend_.set_title(None)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'y_accuracy_test_full.png'))
    plt.show()

    # plot train concept accuracy
    plt.figure(figsize=figsize)
    plt.title(f'Train Concept Accuracy')
    sns.lineplot(data=c_train_loss, x='epoch', y='train accuracy', hue='Representation')
    plt.ylim([0.7, 1.0])
    # plt.xlabel('epochs')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'c_accuracy.png'))
    plt.show()

    # plot train task accuracy
    plt.figure(figsize=figsize)
    plt.title(f'Train Task Accuracy')
    sns.lineplot(data=y_train_loss, x='epoch', y='train accuracy', hue='Representation')
    plt.ylim([0.7, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'y_accuracy.png'))
    plt.show()

    return


def compute_accuracy(c_pred, y_pred, c_true, y_true):
    c_pred = c_pred.ravel().cpu().detach() > 0.5
    y_pred = y_pred.ravel().cpu().detach() > 0.5
    c_true = c_true.ravel().cpu().detach()
    y_true = y_true.ravel().cpu().detach()
    c_accuracy = accuracy_score(c_true, c_pred)
    y_accuracy = accuracy_score(y_true, y_pred)
    return c_accuracy, y_accuracy


if __name__ == '__main__':
    main()
