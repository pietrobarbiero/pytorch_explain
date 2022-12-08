import pandas as pd
import numpy as np
import os

import torch
from torch.nn import BCELoss
from torch.nn.functional import one_hot

from experiments.rlens.model import get_rule_learner, get_scorer, DeepConceptReasoner, get_reasoner
from torch_explain.logic.semantics import ProductTNorm, VectorLogic

torch.autograd.set_detect_anomaly(True)

def main():
    datasets = ['xor', 'trig', 'vec']
    # datasets = ['vec']
    folds = [i+1 for i in range(5)]
    train_epochs = 500
    epochs = 1000
    lr = 0.008
    results = pd.DataFrame()
    for dataset in datasets:
        for fold in folds:
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
            np.random.seed(42)
            train_mask = set(np.random.choice(np.arange(c.shape[0]), int(c.shape[0] * 0.8), replace=False))
            test_mask = set(np.arange(c.shape[0])) - train_mask
            train_mask = torch.LongTensor(list(train_mask))
            test_mask = torch.LongTensor(list(test_mask))

            rule_learner = get_rule_learner(c.shape[1], y1h.shape[1], temperature=10)
            scorer = get_scorer(c_emb.shape[2])
            # reasoner = get_reasoner(ProductTNorm(), scorer, BCELoss())
            reasoner = get_reasoner(VectorLogic(c_emb.shape[2]), scorer, BCELoss())
            model = DeepConceptReasoner(rule_learner, reasoner, scorer, concept_names, class_names, verbose=True)
            model.fit(c[train_mask], c_emb[train_mask], c[train_mask], y1h[train_mask], lr, epochs)
            y_test_pred_reasoner, y_test_pred_learner, c_pred_reasoner, y_pred_bool = model.predict(c[test_mask], c_emb[test_mask])

            test_accuracy_learner = (y_test_pred_learner > 0.).eq(y1h[test_mask]).sum().item() / (y1h[test_mask].size(0) * y1h[test_mask].size(1))
            print(f'Test accuracy learner: {test_accuracy_learner:.4f}')
            test_accuracy_reasoner = (y_test_pred_reasoner > 0.).eq(y1h[test_mask]).sum().item() / (y1h[test_mask].size(0) * y1h[test_mask].size(1))
            print(f'Test accuracy reasoner: {test_accuracy_reasoner:.4f}')
            test_accuracy_bool = (y_pred_bool.squeeze() > 0.5).eq(y1h[test_mask]).sum().item() / (y1h[test_mask].size(0) * y1h[test_mask].size(1))
            print(f'Test accuracy reasoner (bool): {test_accuracy_bool:.4f}')
            c_accuracy_reasoner = (c_pred_reasoner > 0.).eq(c[test_mask]>0.5).sum().item() / (c[test_mask].size(0) * c[test_mask].size(1))
            print(f'Test concept accuracy reasoner: {c_accuracy_reasoner:.4f}')
            test_accuracy_cem = (y_cem[test_mask] > 0.).eq(y[test_mask]).sum().item() / len(y[test_mask])
            print(f'Test accuracy CEM: {test_accuracy_cem:.4f}')

            res_dir = f'./results-2/'
            os.makedirs(res_dir, exist_ok=True)
            out_file = os.path.join(res_dir, 'reasoner_results.csv')

            res1 = pd.DataFrame([[model.learnt_rules_, test_accuracy_reasoner, fold, 'DCR', dataset]], columns=['rules', 'accuracy', 'fold', 'model', 'dataset'])
            res2 = pd.DataFrame([[None, test_accuracy_cem, fold, 'CEM', dataset]], columns=['rules', 'accuracy', 'fold', 'model', 'dataset'])
            res3 = pd.DataFrame([[None, test_accuracy_learner, fold, 'LENs', dataset]], columns=['rules', 'accuracy', 'fold', 'model', 'dataset'])
            res4 = pd.DataFrame([[model.learnt_rules_, test_accuracy_bool, fold, 'TNorm', dataset]], columns=['rules', 'accuracy', 'fold', 'model', 'dataset'])

            if len(results) == 0:
                results = res1
                results = pd.concat((results, res2, res3, res4))
            else:
                results = pd.concat((results, res1, res2, res3, res4))

            results.to_csv(out_file)



if __name__ == '__main__':
    main()
