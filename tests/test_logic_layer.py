import unittest

import torch
from pytorch_lightning import seed_everything
from torch.nn.functional import one_hot

import torch_explain as te
from torch_explain.logic import test_explanation
from torch_explain.logic.nn import explain_class


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
                te.nn.EntropyLinear(2, 10, n_classes=2, awareness='entropy'),
                torch.nn.LeakyReLU(),
                te.nn.LinearIndependent(10, 10, n_classes=2),
                torch.nn.LeakyReLU(),
                te.nn.LinearIndependent(10, 1, n_classes=2, top=True),
                torch.nn.LogSoftmax(dim=1)
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss_form = torch.nn.NLLLoss()
            model.train()
            for epoch in range(61):
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_form(y_pred, y) + 0.0001 * te.nn.functional.entropy_logic_loss(model)

                loss.backward()
                optimizer.step()

                # compute accuracy
                if epoch % 10 == 0:
                    accuracy = y_pred.argmax(dim=1).eq(y).sum().item() / y.size(0)
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            print(model[0].gamma)

            y1h = one_hot(y)
            class_explanation, class_explanations, _ = explain_class(model, x, y1h, target_class=0)
            print(class_explanation)
            print(class_explanations)
            print(test_explanation(class_explanation, 0, x, y1h[:, 0]))
            assert class_explanation == '(feature0000000000 & feature0000000001) | (~feature0000000000 & ~feature0000000001)'

            class_explanation, class_explanations, _ = explain_class(model, x, y1h, target_class=1)
            print(class_explanation)
            print(class_explanations)
            print(test_explanation(class_explanation, 1, x, y1h[:, 1]))
            assert class_explanation == '(feature0000000000 & ~feature0000000001) | (feature0000000001 & ~feature0000000000)'

        return

    def test_explain_class_binary_l1(self):
        for i in range(20):
            seed_everything(i)

            # Problem 1
            x0 = torch.zeros((4, 100))
            x = torch.tensor([
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
            ], dtype=torch.float)
            x = torch.cat([x, x0], dim=1)
            y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

            layers = [
                te.nn.EntropyLinear(x.shape[1], 10, n_classes=2, awareness='l1'),
                torch.nn.LeakyReLU(),
                te.nn.LinearIndependent(10, 10, n_classes=2),
                torch.nn.LeakyReLU(),
                te.nn.LinearIndependent(10, 1, n_classes=2, top=True),
                # torch.nn.LogSoftmax(dim=1)
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss_form = torch.nn.CrossEntropyLoss()
            model.train()
            for epoch in range(1001):
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_form(y_pred, y) + 0.00001 * te.nn.functional.entropy_logic_loss(model)

                loss.backward()
                optimizer.step()

                # compute accuracy
                if epoch % 100 == 0:
                    accuracy = y_pred.argmax(dim=1).eq(y).sum().item() / y.size(0)
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            print(model[0].gamma)

            y1h = one_hot(y)
            class_explanation, class_explanations, _ = explain_class(model, x, y1h, x, y1h, target_class=0)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & feature0000000001) | (~feature0000000000 & ~feature0000000001)'

            class_explanation, class_explanations, _ = explain_class(model, x, y1h, x, y1h, target_class=1)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & ~feature0000000001) | (feature0000000001 & ~feature0000000000)'

        return

    def test_explain_class_binary_entropy(self):
        for i in range(20):
            seed_everything(i)

            # Problem 1
            x0 = torch.zeros((4, 100))
            x = torch.tensor([
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
            ], dtype=torch.float)
            # x = torch.cat([x, x0], dim=1)
            y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

            layers = [
                te.nn.EntropyLinear(x.shape[1], 10, n_classes=2, awareness='entropy'),
                torch.nn.LeakyReLU(),
                te.nn.LinearIndependent(10, 10, n_classes=2),
                torch.nn.LeakyReLU(),
                te.nn.LinearIndependent(10, 1, n_classes=2, top=True),
                torch.nn.LogSoftmax(dim=1)
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss_form = torch.nn.NLLLoss()
            model.train()
            for epoch in range(1001):
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_form(y_pred, y) + 0.00001 * te.nn.functional.entropy_logic_loss(model)

                loss.backward()
                optimizer.step()

                # compute accuracy
                if epoch % 100 == 0:
                    accuracy = y_pred.argmax(dim=1).eq(y).sum().item() / y.size(0)
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            print(model[0].gamma)

            y1h = one_hot(y)
            class_explanation, class_explanations, _ = explain_class(model, x, y1h, target_class=0)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & feature0000000001) | (~feature0000000000 & ~feature0000000001)'

            class_explanation, class_explanations, _ = explain_class(model, x, y1h, target_class=1)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & ~feature0000000001) | (feature0000000001 & ~feature0000000000)'

        return

    def test_explain_class_binary_pruning_bce(self):
        for i in range(3):
            seed_everything(i)

            # Problem 1
            x = torch.tensor([
                [0, 0, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ], dtype=torch.float)
            y = torch.tensor([0, 1, 1, 2], dtype=torch.long)
            y1h = one_hot(y).to(torch.float)

            layers = [
                te.nn.EntropyLinear(4, 10, n_classes=3, awareness='l1'),
                torch.nn.LeakyReLU(),
                te.nn.LinearIndependent(10, 10, n_classes=3),
                torch.nn.LeakyReLU(),
                te.nn.LinearIndependent(10, 1, n_classes=3, top=True),
                torch.nn.Sigmoid()
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss_form = torch.nn.BCELoss()
            model.train()
            for epoch in range(1001):
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_form(y_pred, y1h) + 0.0001 * te.nn.functional.entropy_logic_loss(model)

                loss.backward()
                optimizer.step()

                # compute accuracy
                if epoch % 100 == 0:
                    accuracy = y_pred.argmax(dim=1).eq(y).sum().item() / y.size(0)
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            print(model[0].gamma)

            class_explanation, class_explanations, _ = explain_class(model, x, y1h, target_class=0)
            print(class_explanation)
            print(class_explanations)

            class_explanation, class_explanations, _ = explain_class(model, x, y1h, target_class=1)
            print(class_explanation)
            print(class_explanations)

            class_explanation, class_explanations, _ = explain_class(model, x, y1h, target_class=2)
            print(class_explanation)
            print(class_explanations)

        return


if __name__ == '__main__':
    unittest.main()
