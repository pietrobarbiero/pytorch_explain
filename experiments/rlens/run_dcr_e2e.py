import pandas as pd
import numpy as np
import os
import math

import torch
from sklearn.metrics import accuracy_score
from torch.nn import BCELoss
from torch.nn.functional import one_hot

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

# torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def load_data(dataset, fold, train_epochs):
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


class DCR(torch.nn.Module):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.w_key = torch.nn.Parameter(torch.empty((emb_size, emb_size)))
        self.w_query = torch.nn.Parameter(torch.empty((emb_size, n_classes)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.w_key, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w_query, a=math.sqrt(5))

    def forward(self, x, c):
        keys = x @ self.w_key
        attn_scores = keys @ self.w_query
        attn_scores_softmax = torch.sigmoid(attn_scores)  # or softmax

        neg_scores = 1 - attn_scores_softmax
        c2 = c.unsqueeze(-1).repeat(1, 1, self.n_classes)
        c_or = 1 - neg_scores * c2
        return torch.softmax(torch.prod(c_or, dim=1, keepdim=True).squeeze(), dim=1)


def main():
    datasets = ['xor', 'trig', 'vec']
    # datasets = ['vec']
    folds = [i+1 for i in range(5)]
    train_epochs = 500
    epochs = 500
    learning_rate = 0.008
    batch_size = 500
    limit_batches = 1.0
    results = []
    cols = ['rules', 'accuracy', 'fold', 'model', 'dataset']
    for dataset in datasets:
        for fold in folds:
            results_dir = f"./results/dcr/"
            os.makedirs(results_dir, exist_ok=True)
            model_path = os.path.join(results_dir, 'model.pt')

            train_data, test_data, in_concepts, out_concepts, emb_size, concept_names, class_names = load_data(dataset, fold, train_epochs)
            train_dl = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
            test_dl = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True)

            c_emb = train_data.tensors[1]
            c_scores = train_data.tensors[0]
            y = train_data.tensors[3]
            n_classes = len(class_names)
            # n_classes = 3
            logic = ProductTNorm()

            model = DCR(emb_size, n_classes)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            loss_form = torch.nn.BCEWithLogitsLoss()
            model.train()
            for epoch in range(1001):
                # train step
                optimizer.zero_grad()
                y_pred = model(c_emb, c_scores)
                loss = loss_form(y_pred, y)
                loss.backward()
                optimizer.step()

                # compute accuracy
                if epoch % 100 == 0:
                    accuracy = accuracy_score(y[:, 1], y_pred.argmax(dim=-1).detach())
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            c_emb = test_data.tensors[1]
            c_scores = test_data.tensors[0]
            y = test_data.tensors[3]
            y_pred = model(c_emb, c_scores)
            test_accuracy = accuracy_score(y[:, 1], y_pred.argmax(dim=-1).detach())
            print(f'Test accuracy: {test_accuracy:.4f}')

            results.append(['', test_accuracy, fold, 'DCR (ours)', dataset])
            pd.DataFrame(results, columns=cols).to_csv(os.path.join(results_dir, 'accuracy.csv'))



if __name__ == '__main__':
    main()
