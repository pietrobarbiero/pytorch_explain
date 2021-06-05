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
                te.nn.EntropyLinear(x.shape[1], 10, n_classes=2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(10, 1),
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss_form = torch.nn.CrossEntropyLoss()
            model.train()
            for epoch in range(1001):
                optimizer.zero_grad()
                y_pred = model(x).squeeze(-1)
                loss = loss_form(y_pred, y) + 0.00001 * te.nn.functional.entropy_logic_loss(model)

                loss.backward()
                optimizer.step()

                # compute accuracy
                if epoch % 100 == 0:
                    accuracy = y_pred.argmax(dim=1).eq(y).sum().item() / y.size(0)
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            y1h = one_hot(y)
            target_class = 0
            explanation, explanation_raw = explain_class(model, x, y1h, x, y1h, target_class)
            print(explanation)
            assert explanation == '(feature0000000000 & feature0000000001) | (~feature0000000000 & ~feature0000000001)'
            accuracy, preds = test_explanation(explanation_raw, x, y1h, target_class)
            print(f'Accuracy: {100*accuracy:.2f}%')
            assert accuracy == 1

            target_class = 1
            explanation, explanation_raw = explain_class(model, x, y1h, x, y1h, target_class)
            print(explanation)
            assert explanation == '(feature0000000000 & ~feature0000000001) | (feature0000000001 & ~feature0000000000)'
            accuracy, preds = test_explanation(explanation_raw, x, y1h, target_class)
            print(f'Accuracy: {100*accuracy:.2f}%')
            assert accuracy == 1

        return


if __name__ == '__main__':
    unittest.main()
