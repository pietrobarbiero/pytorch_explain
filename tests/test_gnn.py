import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
import networkx as nx

import torch
from pytorch_lightning import seed_everything
from torch import nn
from torch.nn.functional import one_hot, leaky_relu
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.utils import from_networkx

import torch_explain as te
from torch_explain.logic.metrics import test_explanation, complexity, concept_consistency, formula_consistency
from torch_explain.logic.nn import entropy, psi
from torch_explain.nn.functional import prune_equal_fanin


class TestTemplateObject(unittest.TestCase):

    def test_entropy_gnn(self):
        x, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
        train_mask, test_mask = next(sss.split(x, y))
        x = torch.FloatTensor(x)
        y = one_hot(torch.LongTensor(y))

        C = np.corrcoef(x)
        C_abs = np.abs(C) > 0.9
        print(C_abs.sum() / 2)
        G = nx.from_numpy_array(C_abs)
        edge_index = from_networkx(G).edge_index

        class GCN(nn.Module):
            def __init__(self, num_in_features, num_hidden_features, num_classes):
                super(GCN, self).__init__()
                self.num_classes = num_classes

                self.conv0 = GCNConv(num_hidden_features, num_hidden_features)
                self.conv1 = GCNConv(num_hidden_features, 1)

                # linear layers
                self.lens = te.nn.EntropyLinear(num_in_features, num_hidden_features, n_classes=num_classes)

            def forward(self, x, edge_index):
                x = self.lens(x)

                preds = []
                for nc in range(self.num_classes):
                    xc = self.conv0(x[:, nc], edge_index)
                    xc = leaky_relu(xc)

                    xc = self.conv1(xc, edge_index)
                    xc = leaky_relu(xc)
                    preds.append(xc)

                preds = torch.hstack(preds)
                return preds

        model = GCN(x.shape[1], 10, y.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        loss_form = torch.nn.CrossEntropyLoss()
        model.train()

        for epoch in range(1001):
            # train step
            optimizer.zero_grad()
            y_pred = model(x, edge_index)
            loss = loss_form(y_pred[train_mask], y[train_mask].argmax(dim=1)) #+ 0.00001 * te.nn.functional.entropy_logic_loss(model)
            loss.backward()
            optimizer.step()

            # compute accuracy
            if epoch % 100 == 0:
                accuracy = (y_pred[test_mask]>0).eq(y[test_mask]).sum().item() / (y[test_mask].size(0) * y.size(1))
                print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

        # extract logic formulas
        explanations = entropy.explain_classes(model, x, y, train_mask, test_mask,
                                               edge_index=edge_index, c_threshold=0, topk_explanations=3)

        return


if __name__ == '__main__':
    unittest.main()
