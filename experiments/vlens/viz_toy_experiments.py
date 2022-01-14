import copy
import joblib
import os
import glob
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
    emb_sizes = [2, 10, 50]
    sns.set(font_scale=1.3)
    sns.set_style('whitegrid')
    figsize = [9, 4]

    experiments = {
        'Dot': './results/dot',
        'Trigonometric': './results/trigonometry',
        'XOR': './results/xor',
    }
    methods = {
        'embedding_plane': ['Plane', 4],
        'embedding_norm': ['Norm', 3],
        'fuzzy_extra': ['Fuzzy+EC', 2],
        'fuzzy': ['Fuzzy', 1],
        'bool': ['Boolean', 0],
    }

    train_loss_df = pd.DataFrame()
    val_loss_df = pd.DataFrame()
    train_acc_df = pd.DataFrame()
    val_acc_df = pd.DataFrame()
    test_acc_df = pd.DataFrame()
    for experiment_name, result_dir in experiments.items():
        results_files = glob.glob(os.path.join(result_dir, 'results_**.joblib'))

        for results_file in results_files:
            results = joblib.load(results_file)
            for split in range(cv):
                method_name, method_order = methods[results[f'{split}']['model_name']]
                # if method_name == 'Plane':
                emb_size = results[f'{split}']['emb_size']

                # test accuracy
                c_train = results[f'{split}']['c_train']
                c_test = results[f'{split}']['c_test']
                y_train = results[f'{split}']['y_train']
                y_test = results[f'{split}']['y_test']
                c_accuracy, y_accuracy = compute_accuracy(results[f'{split}']['c_test_pred'],
                                                          results[f'{split}']['y_test_pred'],
                                                          c_test, y_test)
                # check the test accuracy of context and semantics separately
                clf = DecisionTreeClassifier(random_state=42)
                # context accuracy
                c_train_pred = results[f'{split}']['y_train_pred_full'].detach()
                c_train_pred = c_train_pred.reshape(c_train_pred.shape[0], -1)
                c_test_pred = results[f'{split}']['y_test_pred_full'].detach()
                c_test_pred = c_test_pred.reshape(c_test_pred.shape[0], -1)
                context_accuracy = clf.fit(c_train_pred, y_train).score(c_test_pred, y_test)
                # Fuzzy semantics accuracy
                c_train_pred = results[f'{split}']['c_train_pred'].detach()
                c_train_pred = c_train_pred.reshape(c_train.shape[0], -1)
                c_test_pred = results[f'{split}']['c_test_pred'].detach()
                c_test_pred = c_test_pred.reshape(c_test.shape[0], -1)
                semantics_fuzzy_accuracy = clf.fit(c_train_pred, y_train).score(c_test_pred, y_test)
                # Fuzzy semantics accuracy
                semantics_bool_accuracy = clf.fit(c_train_pred>0.5, y_train).score(c_test_pred>0.5, y_test)
                # save test accuracy
                test_acc = pd.DataFrame([[c_accuracy, y_accuracy, context_accuracy,
                                          semantics_fuzzy_accuracy, semantics_bool_accuracy, split, method_name,
                                          emb_size, experiment_name, method_order]],
                                        columns=['Accuracy c', 'Accuracy y', 'Context',
                                                 'Semantics (fuzzy)', 'Semantics (Boolean)', 'split', 'method',
                                                 'Embedding size', 'Dataset', 'Order'])
                test_acc_df = test_acc_df.append(test_acc)

                # train loss
                train_loss = pd.DataFrame(results[f'{split}']['train_loss'], columns=['Loss c', 'Loss y', 'Loss'])
                train_loss['epochs'] = train_loss.index
                train_loss['split'] = split
                train_loss['method'] = method_name
                train_loss['Embedding size'] = emb_size
                train_loss['Dataset'] = experiment_name
                train_loss['Order'] = method_order
                train_loss_df = train_loss_df.append(train_loss)
                # val loss
                val_loss = pd.DataFrame(results[f'{split}']['val_loss'], columns=['Loss c', 'Loss y', 'Loss'])
                val_loss['epochs'] = val_loss.index
                val_loss['split'] = split
                val_loss['method'] = method_name
                val_loss['Embedding size'] = emb_size
                val_loss['Dataset'] = experiment_name
                val_loss['Order'] = method_order
                val_loss_df = val_loss_df.append(val_loss)
                # train accuracy
                train_acc = pd.DataFrame(results[f'{split}']['train_accuracy'], columns=['Accuracy c', 'Accuracy y'])
                train_acc['epochs'] = train_acc.index
                train_acc['split'] = split
                train_acc['method'] = method_name
                train_acc['Embedding size'] = emb_size
                train_acc['Dataset'] = experiment_name
                train_acc['Order'] = method_order
                train_acc_df = train_acc_df.append(train_acc)
                # val accuracy
                val_acc = pd.DataFrame(results[f'{split}']['val_accuracy'], columns=['Accuracy c', 'Accuracy y'])
                val_acc['epochs'] = val_acc.index
                val_acc['split'] = split
                val_acc['method'] = method_name
                val_acc['Embedding size'] = emb_size
                val_acc['Dataset'] = experiment_name
                val_acc['Order'] = method_order
                val_acc_df = val_acc_df.append(val_acc)

    test_acc_df = test_acc_df.sort_values(['Dataset'], ascending=False)
    test_acc_df = test_acc_df.sort_values(['Order', 'method', 'Embedding size'])
    test_acc_df2 = test_acc_df[(test_acc_df['Embedding size'] != 2) & (test_acc_df['Embedding size'] != 10)]

    # # plot test concept accuracy
    # plt.figure()
    # g = sns.catplot(data=test_acc_df, kind="box", x='method', y='Accuracy c',
    #                 hue='Embedding size', col="Dataset", legend=False, height=3.2, aspect=1.5)
    # plt.ylim([0.7, 1.0])
    # g.set_xlabels('')
    # g.set_ylabels('concept accuracy')
    # plt.tight_layout()
    # plt.savefig(os.path.join('./results', 'c_accuracy_test.png'))
    # plt.savefig(os.path.join('./results', 'c_accuracy_test.pdf'))
    # plt.show()
    #
    # # plot test task accuracy
    # plt.figure()
    # g = sns.catplot(data=test_acc_df, kind="bar", x='method', y='Accuracy y',
    #                 hue='Embedding size', col="Dataset", legend=False, height=3.2, aspect=1.5)
    # plt.ylim([0.7, 1.0])
    # g.set_xlabels('')
    # g.set_ylabels('task accuracy')
    # plt.tight_layout()
    # plt.savefig(os.path.join('./results', 'y_accuracy_test.png'))
    # plt.savefig(os.path.join('./results', 'y_accuracy_test.pdf'))
    # plt.show()

    # plot test concept accuracy
    plt.figure()
    g = sns.catplot(data=test_acc_df2, kind="bar", x='method', y='Accuracy c', col="Dataset", legend=False, height=3.2, aspect=1.5)
    plt.ylim([0.7, 1.0])
    g.set_xlabels('')
    g.set_ylabels('concept accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join('./results', 'c_accuracy_test_simple.png'))
    plt.savefig(os.path.join('./results', 'c_accuracy_test_simple.pdf'))
    plt.show()

    # plot test task accuracy
    plt.figure()
    g = sns.catplot(data=test_acc_df2, kind="bar", x='method', y='Accuracy y', col="Dataset", legend=False, height=3.2, aspect=1.5)
    plt.ylim([0.7, 1.0])
    g.set_xlabels('')
    g.set_ylabels('task accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join('./results', 'y_accuracy_test_simple.png'))
    plt.savefig(os.path.join('./results', 'y_accuracy_test_simple.pdf'))
    plt.show()

    context_df = test_acc_df[['Context', 'split', 'method', 'Embedding size', 'Dataset', 'Order']]
    fuzzy_df = test_acc_df[['Semantics (fuzzy)', 'split', 'method', 'Embedding size', 'Dataset', 'Order']]
    boolean_df = test_acc_df[['Semantics (Boolean)', 'split', 'method', 'Embedding size', 'Dataset', 'Order']]
    context_df.columns = ['Accuracy', 'split', 'method', 'Embedding size', 'Dataset', 'Order']
    fuzzy_df.columns = ['Accuracy', 'split', 'method', 'Embedding size', 'Dataset', 'Order']
    boolean_df.columns = ['Accuracy', 'split', 'method', 'Embedding size', 'Dataset', 'Order']
    context_df['Representation'] = 'Neural-Symbolic'
    fuzzy_df['Representation'] = 'Probabilistic'
    boolean_df['Representation'] = 'Boolean'
    context_df['Order2'] = 2
    fuzzy_df['Order2'] = 1
    boolean_df['Order2'] = 0
    context_df = context_df[(context_df['Embedding size'] != 2) & (context_df['Embedding size'] != 10)]
    fuzzy_df = fuzzy_df[(fuzzy_df['Embedding size'] != 2) & (fuzzy_df['Embedding size'] != 10)]
    boolean_df = boolean_df[(boolean_df['Embedding size'] != 2) & (boolean_df['Embedding size'] != 10)]
    repr_accuracy = pd.concat([boolean_df, fuzzy_df, context_df], ignore_index=True)
    repr_accuracy = repr_accuracy.sort_values(['Dataset'], ascending=False)
    repr_accuracy = repr_accuracy.sort_values(['Order', 'Order2', 'method'])
    # plot test task accuracy
    plt.figure()
    g = sns.catplot(data=repr_accuracy, kind="bar", x='Representation', y='Accuracy',
                    hue='method', col="Dataset", legend=False, height=3.2, aspect=1.5)
    plt.ylim([0.6, 1.0])
    g.set_xlabels('Inference')
    g.set_ylabels('task accuracy')
    # g.legend.set_legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join('./results', 'repr_accuracy_test.png'))
    plt.savefig(os.path.join('./results', 'repr_accuracy_test.pdf'))
    plt.show()

    # # draw legend (to be cropped)
    # plt.figure()
    # g = sns.barplot(data=test_acc_df, x='method', y='Accuracy y', hue='Embedding size')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4)
    # plt.tight_layout()
    # plt.savefig(os.path.join('./results', 'legend.png'), bbox_inches=[])
    # # plt.savefig(os.path.join('./results', 'legend.pdf'), bbox_inches=)
    # plt.show()


        # c_train_loss = pd.DataFrame(c_train_loss, columns=['Representation', 'cv_split', 'epoch', 'train accuracy'])
        # y_train_loss = pd.DataFrame(y_train_loss, columns=['Representation', 'cv_split', 'epoch', 'train accuracy'])
        #
        # c_test_acc = pd.DataFrame(c_test_acc, columns=['Representation', 'cv_split', 'test accuracy'])
        # y_test_acc = pd.DataFrame(y_test_acc, columns=['Representation', 'cv_split', 'test accuracy'])
        #
        # y_test_acc_full = pd.DataFrame(y_test_acc_full, columns=['Representation', 'cv_split', 'part', 'test accuracy'])
        #
        # # plot test concept accuracy
        # plt.figure(figsize=figsize)
        # plt.title(f'{experiment} Dataset - Concept Accuracy')
        # sns.boxenplot(data=c_test_acc, x='Representation', y='test accuracy')
        # plt.ylim([0.7, 1.0])
        # plt.xlabel('')
        # plt.tight_layout()
        # plt.savefig(os.path.join(result_dir, 'c_accuracy_test.png'))
        # plt.savefig(os.path.join(result_dir, 'c_accuracy_test.pdf'))
        # plt.show()
        #
        # # plot test task accuracy
        # plt.figure(figsize=figsize)
        # plt.title(f'{experiment} Dataset - Task Accuracy')
        # sns.boxenplot(data=y_test_acc, x='Representation', y='test accuracy')
        # plt.ylim([0.7, 1.0])
        # plt.xlabel('')
        # plt.tight_layout()
        # plt.savefig(os.path.join(result_dir, 'y_accuracy_test.png'))
        # plt.savefig(os.path.join(result_dir, 'y_accuracy_test.pdf'))
        # plt.show()
        #
        # # plot test concept accuracy full
        # plt.figure(figsize=figsize)
        # plt.title(f'{experiment} Dataset - Task Accuracy (Tree)')
        # g = sns.boxenplot(data=y_test_acc_full, x='part', y='test accuracy', hue='Representation')
        # plt.ylim([0.7, 1.0])
        # plt.xlabel('')
        # # g.legend_.set_title(None)
        # g.legend_.remove()
        # plt.tight_layout()
        # plt.savefig(os.path.join(result_dir, 'y_accuracy_test_full.png'))
        # plt.savefig(os.path.join(result_dir, 'y_accuracy_test_full.pdf'))
        # plt.show()
        #
        # # plot train concept accuracy
        # plt.figure(figsize=figsize)
        # plt.title(f'{experiment} Dataset - Train Concept Accuracy')
        # sns.lineplot(data=c_train_loss, x='epoch', y='train accuracy', hue='Representation')
        # plt.ylim([0.7, 1.0])
        # # plt.xlabel('epochs')
        # plt.tight_layout()
        # plt.savefig(os.path.join(result_dir, 'c_accuracy.png'))
        # plt.savefig(os.path.join(result_dir, 'c_accuracy.pdf'))
        # plt.show()
        #
        # # plot train task accuracy
        # plt.figure(figsize=figsize)
        # plt.title(f'{experiment} Dataset - Train Task Accuracy')
        # sns.lineplot(data=y_train_loss, x='epoch', y='train accuracy', hue='Representation')
        # plt.ylim([0.7, 1.0])
        # plt.tight_layout()
        # plt.savefig(os.path.join(result_dir, 'y_accuracy.pdf'))
        # plt.show()

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
