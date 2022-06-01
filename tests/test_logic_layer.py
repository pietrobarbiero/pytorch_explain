import unittest

import torch
from pytorch_lightning import seed_everything
from torch.nn.functional import one_hot

import torch_explain as te
from torch_explain.logic.metrics import test_explanation, complexity, concept_consistency, formula_consistency
from torch_explain.logic.nn import entropy, psi
from torch_explain.nn.functional import prune_equal_fanin


class TestTemplateObject(unittest.TestCase):
    def test_psi_explain_class_binary(self):
        for i in range(1):
            seed_everything(i)

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
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ], dtype=torch.float)
        # human, car
        y = torch.tensor([  # 1, 0, 0, 1], dtype=torch.long)
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
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
        target_class_names = ['y', '¬y', 'z', '¬z']

        for epoch in range(2001):
            # train step
            optimizer.zero_grad()
            y_pred = model(x).squeeze(-1)
            loss = loss_form(y_pred, y) + 0.0001 * te.nn.functional.entropy_logic_loss(model)
            loss.backward()
            optimizer.step()

            # compute accuracy
            if epoch % 100 == 0:
                accuracy = (y_pred>0.5).eq(y).sum().item() / (y.size(0) * y.size(1))
                print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

                # extract logic formulas
                train_mask = test_mask = torch.arange(len(y))
                explanations = entropy.explain_classes(model, x, y, train_mask, test_mask,
                                                       c_threshold=0.5, y_threshold=0.5)

        return


if __name__ == '__main__':
    unittest.main()
