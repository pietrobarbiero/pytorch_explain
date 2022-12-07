import pandas as pd
import numpy as np
import os

import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from torch.nn import BCELoss
from torch.nn.functional import one_hot

from experiments.rlens.model import get_rule_learner, get_scorer, DeepConceptReasoner, get_reasoner
from torch_explain.logic.semantics import ProductTNorm


def main():
    datasets = ['xor', 'trig', 'vec']
    # datasets = ['vec']
    folds = [i + 1 for i in range(5)]
    train_epochs = 500
    epochs = 10000
    few_shot_epochs = 10
    lr = 0.008
    results = pd.DataFrame()
    for dataset in datasets:
        for fold in folds:
            c = np.load(
                f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_AdaptiveDropout_NoProbConcat_lambda_fold_{fold}/test_embedding_semantics_on_epoch_{train_epochs}.npy')
            c_emb = np.load(
                f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_AdaptiveDropout_NoProbConcat_lambda_fold_{fold}/test_embedding_vectors_on_epoch_{train_epochs}.npy')
            y_cem = np.load(
                f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_AdaptiveDropout_NoProbConcat_lambda_fold_{fold}/test_model_output_on_epoch_{train_epochs}.npy')
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
            reasoner = get_reasoner(ProductTNorm(), scorer, BCELoss())
            model = DeepConceptReasoner(rule_learner, reasoner, scorer, concept_names, class_names, verbose=True)
            model.fit(c[train_mask], c_emb[train_mask], c[train_mask], y1h[train_mask], lr, epochs)

            # define OOD task
            # y_ood = ((c[:, 0] > 0.5) & (c[:, 1] > 0.5)).long().detach()
            # y_ood = (torch.logical_not(c[:, 0] > 0.5)).long().detach()
            y_ood = torch.logical_and(c[:, 0] > 0.5, c[:, 1] > 0.5).long().detach()
            # y_ood = torch.logical_or(y_ood, torch.sigmoid(torch.randn(len(c)) - 0.5) > 0.5).long()
            y1h_ood = one_hot(y_ood).float()
            # ood_rule = [{'explanation': 'x0', 'name': 'y_ood1'},
            #             {'explanation': '~x0', 'name': 'y_ood2'}]
            ood_rule = [{'explanation': 'x0 & x1', 'name': 'y_ood2'}]

            y1h_all = torch.hstack((y1h, y_ood.unsqueeze(-1)))
            model.learnt_rules_ += ood_rule
            model.fit(c[train_mask], c_emb[train_mask], c[train_mask], y1h_all[train_mask], lr, epochs, use_learnt_rules=False)

            # # x_wmc = torch.sigmoid(c_emb[test_mask])
            # # y_emb = self.reasoner(x_wmc, self.concept_names, self.learnt_rules_)
            # # y_pred = self.scorer(y_emb).squeeze(-1)
            # tsne = TSNE()
            # c2 = tsne.fit_transform(torch.sigmoid(c_emb[test_mask, 0].squeeze()))
            # mask1 = y_ood[test_mask] > 0.5
            # mask2 = c[test_mask, 0] > 0.5
            # plt.figure()
            # plt.scatter(c2[mask1, 0], c2[mask1, 1], s=50)
            # plt.scatter(c2[~mask1, 0], c2[~mask1, 1], s=50)
            # plt.scatter(c2[mask2, 0], c2[mask2, 1], marker='+', s=5)
            # plt.scatter(c2[~mask2, 0], c2[~mask2, 1], marker='+', s=5)
            # plt.show()

            # few shot learning
            # skf = StratifiedKFold(n_splits=10, shuffle=False)
            # test_mask_ood, train_mask_ood = next(iter(skf.split(c[test_mask], y_ood[test_mask])))
            # model.learnt_rules_ = ood_rule
            # model.fit(c[test_mask][train_mask_ood], c_emb[test_mask][train_mask_ood], y1h_ood[test_mask][train_mask_ood], lr, few_shot_epochs, use_learnt_rules=False)

            y_test_pred_reasoner, y_test_pred_learner, c_pred_reasoner, y_pred_bool = model.predict(c[test_mask],
                                                                                                    c_emb[test_mask])

            test_accuracy_learner = 0
            # test_accuracy_learner = np.mean([y_test_pred_learner.argmax(axis=-1).eq(y_ood[test_mask][test_mask_ood]).sum().item() / y[test_mask][test_mask_ood].size(0),
            #                                  (y_test_pred_learner[:, :2] > 0.).eq(y1h[test_mask][test_mask_ood]).sum().item() / (y1h[test_mask][test_mask_ood].size(0) * y1h[test_mask][test_mask_ood].size(1))])
            print(f'Test accuracy learner: {test_accuracy_learner:.4f}')
            # test_accuracy_reasoner = np.mean([(y_test_pred_reasoner[:, 2] > 0.).eq(y_ood[test_mask][test_mask_ood]).sum().item() / y1h_ood[test_mask][test_mask_ood].size(0),
            #                                  (y_test_pred_reasoner[:, :2] > 0.).eq(y1h[test_mask][test_mask_ood]).sum().item() / (y1h[test_mask][test_mask_ood].size(0) * y1h[test_mask][test_mask_ood].size(1))])
            test_accuracy_reasoner = (y_test_pred_reasoner[:, 2].squeeze() > 0.5).eq(y_ood[test_mask]).sum().item() / y1h_ood[
                test_mask].size(0)
            print(f'Test accuracy reasoner: {test_accuracy_reasoner:.4f}')
            test_accuracy_bool = (y_pred_bool[:, 2].squeeze() > 0.5).eq(y_ood[test_mask]).sum().item() / y1h_ood[
                test_mask].size(0)
            print(f'Test accuracy reasoner (bool): {test_accuracy_bool:.4f}')
            c_accuracy_reasoner = (c_pred_reasoner > 0.5).eq(c[test_mask] > 0.5).sum().item() / (
                        c[test_mask].size(0) * c[test_mask].size(1))
            print(f'Test concept accuracy reasoner: {c_accuracy_reasoner:.4f}')
            test_accuracy_cem = 0
            # test_accuracy_cem = np.mean([(y_cem[test_mask_ood] > 0.).eq(y_ood[test_mask_ood]).sum().item() / y[test_mask_ood].size(0),
            #                              (y_cem[test_mask_ood] > 0.).eq(y[test_mask_ood]).sum().item() / y[test_mask_ood].size(0)])
            print(f'Test accuracy CEM: {test_accuracy_cem:.4f}')

            res_dir = f'./results/ood-nt/'
            os.makedirs(res_dir, exist_ok=True)
            out_file = os.path.join(res_dir, 'reasoner_results.csv')

            res1 = pd.DataFrame([[model.learnt_rules_, test_accuracy_reasoner, fold, 'DCR', dataset]],
                                columns=['rules', 'accuracy', 'fold', 'model', 'dataset'])
            res2 = pd.DataFrame([[None, test_accuracy_cem, fold, 'CEM', dataset]],
                                columns=['rules', 'accuracy', 'fold', 'model', 'dataset'])
            res3 = pd.DataFrame([[model.learnt_rules_, test_accuracy_learner, fold, 'LENs', dataset]],
                                columns=['rules', 'accuracy', 'fold', 'model', 'dataset'])
            res4 = pd.DataFrame([[model.learnt_rules_, test_accuracy_bool, fold, 'TNorm', dataset]],
                                columns=['rules', 'accuracy', 'fold', 'model', 'dataset'])

            if len(results) == 0:
                results = res1
                results = pd.concat((results, res2, res3, res4))
            else:
                results = pd.concat((results, res1, res2, res3, res4))

            results.to_csv(out_file)


if __name__ == '__main__':
    main()
