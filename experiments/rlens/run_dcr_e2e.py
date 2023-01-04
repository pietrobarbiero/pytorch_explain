import pandas as pd
import numpy as np
import os
import math

import torch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from torch.nn import BCELoss
from torch.nn.functional import one_hot
from collections import Counter

from xgboost import XGBClassifier

from experiments.rlens.model import DeepConceptReasoner
from torch_explain.logic.semantics import ProductTNorm, VectorLogic
import pandas as pd
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torchvision
import torch
import os

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from torch_explain.nn.logic import DCR

# torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def load_data(dataset, fold, train_epochs):
    if dataset in ['cub', 'celeba']:
        c = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_Adaptive_NoProbConcat_resnet34_fold_{fold}/test_embedding_semantics_on_epoch_{train_epochs}.npy')
        c_emb = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_Adaptive_NoProbConcat_resnet34_fold_{fold}/test_embedding_vectors_on_epoch_{train_epochs}.npy')
        y_cem = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_Adaptive_NoProbConcat_resnet34_fold_{fold}/test_model_output_on_epoch_{train_epochs}.npy')
        # c1 = np.load('./results/xor_activations_final_rerun/test_embedding_acts/c_test.npy')
        # c2 = np.load('./results/xor_activations_final_rerun/test_embedding_acts/c_val.npy')
        # y1 = np.load('./results/xor_activations_final_rerun/test_embedding_acts/y_test.npy')
        y = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/y_val.npy')
    else:
        c = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_AdaptiveDropout_NoProbConcat_lambda_fold_{fold}/test_embedding_semantics_on_epoch_{train_epochs}.npy')
        c_emb = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_AdaptiveDropout_NoProbConcat_lambda_fold_{fold}/test_embedding_vectors_on_epoch_{train_epochs}.npy')
        y_cem = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_AdaptiveDropout_NoProbConcat_lambda_fold_{fold}/test_model_output_on_epoch_{train_epochs}.npy')
        # c1 = np.load('./results/xor_activations_final_rerun/test_embedding_acts/c_test.npy')
        # c2 = np.load('./results/xor_activations_final_rerun/test_embedding_acts/c_val.npy')
        # y1 = np.load('./results/xor_activations_final_rerun/test_embedding_acts/y_test.npy')
        y = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/y_val.npy')

    c = torch.FloatTensor(c)
    c_emb = torch.FloatTensor(c_emb)
    c_emb = c_emb.reshape(c_emb.shape[0], c.shape[1], -1)
    y_cem = torch.FloatTensor(y_cem).squeeze()
    y = torch.LongTensor(y)
    y1h = one_hot(y).float()
    concept_names = [f'x{i}' for i in range(c.shape[1])]
    class_names = [f'y{i}' for i in range(y1h.shape[1])]
    in_concepts = c_emb.shape[1]
    out_concepts = y1h.shape[1]
    emb_size = c_emb.shape[2]
    np.random.seed(42)
    train_mask = set(np.random.choice(np.arange(c.shape[0]), int(c.shape[0] * 0.8), replace=False))
    test_mask = set(np.arange(c.shape[0])) - train_mask
    train_mask = torch.LongTensor(list(train_mask))
    test_mask = torch.LongTensor(list(test_mask))
    train_dataset = torch.utils.data.TensorDataset(c[train_mask], c_emb[train_mask], (c[train_mask]>0.5).float(), y1h[train_mask])
    test_dataset = torch.utils.data.TensorDataset(c[test_mask], c_emb[test_mask], (c[test_mask]>0.5).float(), y1h[test_mask])
    return train_dataset, test_dataset, in_concepts, out_concepts, emb_size, concept_names, class_names


def main():
    random_state = 42
    datasets = ['xor', 'trig', 'vec']
    train_epochs = [500, 500, 500]
    n_epochs = [3000, 3000, 3000]
    temperatures = [10000000, 10000000, 10000000]

    # datasets = ['cub', 'celeba']
    # train_epochs = [300, 200]
    # datasets = ['celeba', 'bla']
    # train_epochs = [200, 200]
    # n_epochs = [300, 300]
    # temperatures = [100, 0.1]
    max_classes = 10

    # datasets = ['xor']
    competitors = [
        DecisionTreeClassifier(random_state=random_state),
        LogisticRegression(random_state=random_state),
        # XGBClassifier(),
        # GradientBoostingClassifier(random_state=random_state)
        RandomForestClassifier(random_state=random_state)
    ]
    folds = [i+1 for i in range(5)]
    # train_epochs = 500
    # epochs = 500
    learning_rate = 0.008
    batch_size = 500
    limit_batches = 1.0
    n_hidden = 50
    results = []
    local_explanations_df = pd.DataFrame()
    global_explanations_df = pd.DataFrame()
    counterfactuals_df = pd.DataFrame()
    logic = ProductTNorm()
    cols = ['rules', 'accuracy', 'fold', 'model', 'dataset']
    for dataset, train_epoch, epochs, temperature in zip(datasets, train_epochs, n_epochs, temperatures):
        for fold in folds:
            results_dir = f"./results/dcr2/"
            os.makedirs(results_dir, exist_ok=True)
            model_path = os.path.join(results_dir, 'model.pt')

            train_data, test_data, in_concepts, out_concepts, emb_size, concept_names, class_names = load_data(dataset, fold, train_epoch)
            train_dl = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
            test_dl = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True)

            c_emb_train = train_data.tensors[1]
            c_scores_train = (train_data.tensors[0]>0.5).float()
            # c_emb = torch.concat((c_emb, torch.randn(c_emb.shape)), dim=1)
            # c_scores = torch.concat((c_scores, torch.zeros_like(c_scores)), dim=1)
            y_train = train_data.tensors[3]
            # n_classes = len(class_names)
            # n_classes = 3
            if dataset == 'cub':
                y_train = y_train[:, :max_classes]
            class_names = [f'y{i}' for i in range(y_train.shape[1])]
            n_classes = len(class_names)

            model = DCR(c_scores_train.shape[1], emb_size, n_classes, logic, temperature, 10.)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            loss_form = torch.nn.CrossEntropyLoss()
            model.train()
            for epoch in range(epochs):
                # train step
                optimizer.zero_grad()
                y_pred, sign_attn_mask, filter_attn_mask = model.forward(c_emb_train, c_scores_train, return_attn=True)
                loss = loss_form(y_pred, y_train) #+ epoch / epochs * 1 / (torch.linalg.norm(filter_attn_mask.ravel() - 0.5, -float('inf')) + 0.001)
                loss.backward()
                optimizer.step()

                # compute accuracy
                if epoch % 100 == 0:
                    accuracy = f1_score(y_train.argmax(dim=-1).detach(), y_pred.argmax(dim=-1).detach(), average='weighted')
                    # accuracy = accuracy_score((y_train>0.5).ravel(), (y_pred>0.5).ravel().detach())
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            c_emb_test = test_data.tensors[1]
            c_scores_test = test_data.tensors[0]
            y_test = test_data.tensors[3]
            if dataset == 'cub':
                y_test = y_test[:, :max_classes]
            y_pred = model(c_emb_test, c_scores_test)
            test_accuracy = f1_score(y_test.argmax(dim=-1).detach(), y_pred.argmax(dim=-1).detach(), average='weighted')
            # test_accuracy = accuracy_score((y > 0.5).ravel(), (y_pred > 0.5).ravel().detach())
            print(f'Test accuracy: {test_accuracy:.4f}')

            local_explanations = model.explain(c_emb_test, c_scores_test, mode='local')
            local_explanations = pd.DataFrame(local_explanations)
            local_explanations['fold'] = fold
            local_explanations['dataset'] = dataset
            local_explanations_df = pd.concat([local_explanations_df, local_explanations], axis=0)
            local_explanations_df.to_csv(os.path.join(results_dir, 'local_explanations.csv'))

            global_explanations = model.explain(c_emb_test, c_scores_test, mode='global')
            global_explanations = pd.DataFrame(global_explanations)
            global_explanations['fold'] = fold
            global_explanations['dataset'] = dataset
            global_explanations_df = pd.concat([global_explanations_df, global_explanations], axis=0)
            global_explanations_df.to_csv(os.path.join(results_dir, 'global_explanations.csv'))

            results.append(['', test_accuracy, fold, 'DCR (ours)', dataset])
            pd.DataFrame(results, columns=cols).to_csv(os.path.join(results_dir, 'accuracy.csv'))

            counterfactuals = model.counterfact(c_emb_test, c_scores_test)
            counterfactuals = pd.DataFrame(counterfactuals)
            counterfactuals['fold'] = fold
            counterfactuals['dataset'] = dataset
            counterfactuals_df = pd.concat([counterfactuals_df, counterfactuals], axis=0)
            counterfactuals_df.to_csv(os.path.join(results_dir, 'counterfactuals.csv'))

            # competitors!
            print('\nAnd now run competitors!\n')
            for classifier in competitors:
                classifier.fit(c_scores_train, y_train.argmax(dim=-1).detach())
                y_pred = classifier.predict(c_scores_test)
                test_accuracy = f1_score(y_test.argmax(dim=-1).detach(), y_pred, average='weighted')
                print(f'{classifier.__class__.__name__}: Test accuracy: {test_accuracy:.4f}')

                results.append(['', test_accuracy, fold, classifier.__class__.__name__, dataset])
                pd.DataFrame(results, columns=cols).to_csv(os.path.join(results_dir, 'accuracy.csv'))


if __name__ == '__main__':
    main()
