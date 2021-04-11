import unittest

import torch
from pytorch_lightning import seed_everything

import torch_explain as te
from torch_explain.logic import explain_class
from torch_explain.utils.pruning import prune_logic_layers, l1_loss


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
            y = torch.tensor([0, 1, 1, 0], dtype=torch.float)

            layers = [
                te.nn.Logic(2, 5, activation='identity_bool'),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 5),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 1),
                te.nn.Logic(1, 1, top=True, activation='sigmoid'),
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss_form = torch.nn.BCELoss()
            model.train()
            for epoch in range(2000):
                optimizer.zero_grad()
                y_pred = model(x).squeeze()
                loss = loss_form(y_pred, y)

                loss.backward()
                optimizer.step()

                # compute accuracy
                if epoch % 100 == 0:
                    y_pred_d = y_pred > 0.5
                    accuracy = y_pred_d.eq(y).sum().item() / y.size(0)
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            class_explanation, class_explanations = explain_class(model, x, y, binary=True,
                                                                  target_class=0,
                                                                  topk_explanations=10)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & feature0000000001) | (~feature0000000000 & ~feature0000000001)'

            class_explanation, class_explanations = explain_class(model, x, y, binary=True,
                                                                  target_class=1,
                                                                  topk_explanations=10)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & ~feature0000000001) | (feature0000000001 & ~feature0000000000)'

        return

    def test_explain_class_leaky_relu(self):
        for i in range(20):
            seed_everything(i)

            # Problem 1
            x = torch.tensor([
                [-1, -1],
                [-1, 1],
                [1, -1],
                [1, 1],
            ], dtype=torch.float)
            y = torch.tensor([0, 1, 1, 0], dtype=torch.float)

            layers = [
                te.nn.Logic(2, 5, activation='leaky_relu'),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 5),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 1),
                te.nn.Logic(1, 1, top=True, activation='sigmoid'),
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss_form = torch.nn.BCELoss()
            model.train()
            for epoch in range(2000):
                optimizer.zero_grad()
                y_pred = model(x).squeeze()
                loss = loss_form(y_pred, y)

                loss.backward()
                optimizer.step()

                # compute accuracy
                if epoch % 100 == 0:
                    y_pred_d = y_pred > 0.5
                    accuracy = y_pred_d.eq(y).sum().item() / y.size(0)
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            class_explanation, class_explanations = explain_class(model, x, y, binary=True,
                                                                  target_class=0,
                                                                  topk_explanations=10)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & feature0000000001) | (~feature0000000000 & ~feature0000000001)'

            class_explanation, class_explanations = explain_class(model, x, y, binary=True,
                                                                  target_class=1,
                                                                  topk_explanations=10)
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
            y = torch.tensor([0, 1, 1, 0], dtype=torch.float)

            layers = [
                te.nn.Logic(4, 5, activation='identity_bool'),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 5),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 1),
                te.nn.Logic(1, 1, top=True, activation='sigmoid'),
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss_form = torch.nn.BCELoss()
            model.train()
            for epoch in range(2000):
                optimizer.zero_grad()
                y_pred = model(x).squeeze()
                loss = loss_form(y_pred, y) + 0.0001 * l1_loss(model)

                loss.backward()
                optimizer.step()

                prune_logic_layers(model, current_epoch=epoch, prune_epoch=1000, fan_in=2)

                # compute accuracy
                if epoch % 100 == 0:
                    y_pred_d = y_pred > 0.5
                    accuracy = y_pred_d.eq(y).sum().item() / y.size(0)
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            class_explanation, class_explanations = explain_class(model, x, y, binary=True,
                                                                  target_class=0,
                                                                  topk_explanations=10)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & feature0000000001) | (~feature0000000000 & ~feature0000000001)'

            class_explanation, class_explanations = explain_class(model, x, y, binary=True,
                                                                  target_class=1,
                                                                  topk_explanations=10)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & ~feature0000000001) | (feature0000000001 & ~feature0000000000)'

        return

    def test_explain_multi_class(self):
        for i in range(20):
            seed_everything(i)

            # Problem 1
            x = torch.tensor([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ], dtype=torch.float)
            y = torch.tensor([0, 1, 1, 2], dtype=torch.long)

            layers = [
                te.nn.Logic(2, 5, activation='identity_bool'),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 5),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 3),
                te.nn.Logic(3, 3, activation='identity', top=True),
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss_form = torch.nn.CrossEntropyLoss()
            model.train()
            for epoch in range(2000):
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_form(y_pred, y)

                loss.backward()
                optimizer.step()

                # compute accuracy
                if epoch % 100 == 0:
                    y_pred_d = y_pred.argmax(dim=1)
                    accuracy = y_pred_d.eq(y).sum().item() / y.size(0)
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            class_explanation, class_explanations = explain_class(model, x, y, binary=False,
                                                                  target_class=2,
                                                                  topk_explanations=10)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == 'feature0000000000 & feature0000000001'

            class_explanation, class_explanations = explain_class(model, x, y, binary=False,
                                                                  target_class=1,
                                                                  topk_explanations=10)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & ~feature0000000001) | (feature0000000001 & ~feature0000000000)'

            class_explanation, class_explanations = explain_class(model, x, y, binary=False,
                                                                  target_class=0,
                                                                  topk_explanations=10)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '~feature0000000000 & ~feature0000000001'

        return


if __name__ == '__main__':
    unittest.main()
