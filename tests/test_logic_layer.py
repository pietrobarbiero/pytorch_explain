import unittest

import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
from torch.nn.functional import one_hot
from torch import nn
from torch.nn.functional import one_hot, leaky_relu
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.utils import from_networkx
import networkx as nx
import numpy as np

import torch_explain as te
from torch_explain.logic.metrics import test_explanation, complexity, concept_consistency, formula_consistency
from torch_explain.logic.nn import entropy, psi
from torch_explain.nn.functional import prune_equal_fanin


class TestTemplateObject(unittest.TestCase):
    def test_psi_explain_class_binary(self):
        for i in range(1):

            # Problem 1
            x = torch.tensor([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ], dtype=torch.float)
            y = torch.tensor([0, 1, 1, 0], dtype=torch.float).unsqueeze(1)

            layers = [
                torch.nn.Linear(x.shape[1], 10),
                torch.nn.Sigmoid(),
                torch.nn.Linear(10, 5),
                torch.nn.Sigmoid(),
                torch.nn.Linear(5, 1),
                torch.nn.Sigmoid(),
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss_form = torch.nn.BCELoss()
            model.train()
            for epoch in range(6001):
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_form(y_pred, y) + 0.000001 * te.nn.functional.l1_loss(model)
                loss.backward()
                optimizer.step()

                model = prune_equal_fanin(model, epoch, prune_epoch=1000, k=2)

                # compute accuracy
                if epoch % 100 == 0:
                    accuracy = y.eq(y_pred>0.5).sum().item() / y.size(0)
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            y1h = one_hot(y.squeeze().long())

            explanation = psi.explain_class(model, x)
            explanation_complexity = complexity(explanation)
            cc = concept_consistency([explanation])
            fc = formula_consistency([explanation])
            print(explanation)
            print(explanation_complexity)
            print(cc)
            print(fc)
            accuracy, preds = test_explanation(explanation, x, y1h, target_class=1)
            print(f'Accuracy: {100*accuracy:.2f}%')

        return

    def test_entropy_multi_target(self):

        # eye, nose, window, wheel, hand, radio
        x = torch.tensor([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ], dtype=torch.float)
        # human, car
        y = torch.tensor([  # 1, 0, 0, 1], dtype=torch.long)
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ], dtype=torch.float)
        y1h = y  # one_hot(y)

        layers = [
            te.nn.EntropyLinear(x.shape[1], 20, n_classes=y1h.shape[1], temperature=0.3),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(20, 10),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(10, 1),
        ]
        model = torch.nn.Sequential(*layers)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        loss_form = torch.nn.BCEWithLogitsLoss()
        model.train()

        concept_names = ['x1', 'x2', 'x3', 'x4']
        class_names = ['y', '¬y', 'z', '¬z']
        train_mask = torch.tensor([1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1], dtype=torch.bool)
        test_mask = ~train_mask
        train_mask = torch.where(train_mask)[0]
        test_mask = torch.where(test_mask)[0]
        for epoch in range(2001):
            # train step
            optimizer.zero_grad()
            y_pred = model(x).squeeze(-1)
            loss = loss_form(y_pred[train_mask], y[train_mask]) + 0.0001 * te.nn.functional.entropy_logic_loss(model)
            loss.backward()
            optimizer.step()

            # compute accuracy
            if epoch % 100 == 0:
                train_accuracy = (y_pred[train_mask]>0.5).eq(y[train_mask]).sum().item() / (y.size(0) * y.size(1))
                test_accuracy = (y_pred[test_mask]>0.5).eq(y[test_mask]).sum().item() / (y.size(0) * y.size(1))
                print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {train_accuracy:.4f} test accuracy: {test_accuracy:.4f}')

                # extract logic formulas
                # train_mask = test_mask = torch.arange(len(y))
                explanations = entropy.explain_classes(model, x, y, train_mask, test_mask,
                                                       c_threshold=0.5, y_threshold=0.5, verbose=True,
                                                       concept_names=concept_names, class_names=class_names)

        return


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
                                               edge_index=edge_index, c_threshold=0,
                                               topk_explanations=3, verbose=True)

        return


if __name__ == '__main__':
    unittest.main()
