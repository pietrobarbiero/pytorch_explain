import unittest

import torch
from pytorch_lightning import seed_everything
from torch.nn.functional import one_hot

import torch_explain as te
from torch_explain.logic import explain_class


class TestTemplateObject(unittest.TestCase):
    def test_explain_class_binary(self):
        for i in range(20):
            seed_everything(i)

            # Problem 1
            x = torch.tensor([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ], dtype=torch.float)
            y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

            layers = [
                te.nn.LogicAttention(2, 10, n_classes=2, n_heads=1),
                torch.nn.LeakyReLU(),
                te.nn.LogicAttention(10, 10, n_classes=2),
                torch.nn.LeakyReLU(),
                te.nn.LogicAttention(10, 1, n_classes=2, top=True),
                torch.nn.LogSoftmax(dim=1)
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss_form = torch.nn.NLLLoss()
            model.train()
            for epoch in range(61):
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_form(y_pred, y) #+ 0.0001 * te.nn.functional.l1_loss(model)

                loss.backward()
                optimizer.step()

                # compute accuracy
                if epoch % 10 == 0:
                    accuracy = y_pred.argmax(dim=1).eq(y).sum().item() / y.size(0)
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            y1h = one_hot(y)
            class_explanation, class_explanations = explain_class(model, x, y1h, target_class=0)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & feature0000000001) | (~feature0000000000 & ~feature0000000001)'

            class_explanation, class_explanations = explain_class(model, x, y1h, target_class=1)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & ~feature0000000001) | (feature0000000001 & ~feature0000000000)'

        return

    def test_explain_class_binary_pruning(self):
        for i in range(20):
            seed_everything(i)

            # Problem 1
            x = torch.tensor([
                [0, 0, 0, 1],
                [0, 1, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 0, 1],
            ], dtype=torch.float)
            y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

            layers = [
                te.nn.LogicAttention(4, 10, n_classes=2, n_heads=1),
                torch.nn.LeakyReLU(),
                te.nn.LogicAttention(10, 10, n_classes=2),
                torch.nn.LeakyReLU(),
                te.nn.LogicAttention(10, 1, n_classes=2, top=True),
                torch.nn.LogSoftmax(dim=1)
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss_form = torch.nn.NLLLoss()
            model.train()
            for epoch in range(1001):
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_form(y_pred, y) + 0.0001 * te.nn.functional.l1_loss(model)

                loss.backward()
                optimizer.step()

                # compute accuracy
                if epoch % 100 == 0:
                    accuracy = y_pred.argmax(dim=1).eq(y).sum().item() / y.size(0)
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            y1h = one_hot(y)
            class_explanation, class_explanations = explain_class(model, x, y1h, target_class=0)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & feature0000000001) | (~feature0000000000 & ~feature0000000001)'

            class_explanation, class_explanations = explain_class(model, x, y1h, target_class=1)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & ~feature0000000001) | (feature0000000001 & ~feature0000000000)'

        return

    def test_explain_multi_class(self):
        for i in range(20):
            seed_everything(i)

            # Problem 1
            x = torch.tensor([
                [0, 0, 0, 1],
                [0, 1, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 0, 1],
            ], dtype=torch.float)
            y = torch.tensor([0, 1, 1, 2], dtype=torch.long)

            layers = [
                te.nn.LogicAttention(4, 10, n_classes=3, n_heads=3),
                torch.nn.LeakyReLU(),
                te.nn.LogicAttention(10, 10, n_classes=3),
                torch.nn.LeakyReLU(),
                te.nn.LogicAttention(10, 1, n_classes=3, top=True),
                torch.nn.LogSoftmax(dim=1)
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            loss_form = torch.nn.NLLLoss()
            model.train()
            for epoch in range(501):
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_form(y_pred, y) + 0.0001 * te.nn.functional.l1_loss(model)

                loss.backward()
                optimizer.step()

                # compute accuracy
                if epoch % 100 == 0:
                    accuracy = y_pred.argmax(dim=1).eq(y).sum().item() / y.size(0)
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            y1h = one_hot(y)
            class_explanation, class_explanations = explain_class(model, x, y1h, target_class=2)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == 'feature0000000000 & feature0000000001'

            class_explanation, class_explanations = explain_class(model, x, y1h, target_class=1)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & ~feature0000000001) | (feature0000000001 & ~feature0000000000)'

            class_explanation, class_explanations = explain_class(model, x, y1h, target_class=0)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '~feature0000000000 & ~feature0000000001'

        return



if __name__ == '__main__':
    unittest.main()
