import unittest

import torch
from pytorch_lightning import seed_everything
from torch.nn.functional import one_hot

import torch_explain as te
from torch_explain.logic.metrics import test_explanation, complexity, concept_consistency, formula_consistency
from torch_explain.logic.nn import entropy, psi
from torch_explain.nn.functional import prune_equal_fanin, entropy_logic_loss


class TestTemplateObject(unittest.TestCase):
    def test_glens_modules(self):

        embed = torch.randn((10, 2))
        adj = torch.randint(0, 2, (10, 10)).float()
        pool = torch.randint(-5, 5, (10, 3)).float()

        ldp = te.nn.logic.LogDiffPool()
        model = torch.nn.Sequential(ldp)
        out, out_adj = ldp(embed, adj, pool)
        entropy_logic_loss(model)

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

        for epoch in range(7001):
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
                for target_class in range(y.shape[1]):
                    explanation_class_i, exp_raw = entropy.explain_class(model, x, y1h, x, y1h, target_class,
                                                                         concept_names=concept_names)
                    accuracy_i, preds = test_explanation(exp_raw, x, y1h, target_class)
                    if explanation_class_i: explanation_class_i = explanation_class_i.replace('&', '∧').replace('|', '∨').replace('~', '¬')
                    explanation_class_i = f'∀x: {explanation_class_i} ↔ {target_class_names[target_class]}'

                    print(f'\tExplanation class {target_class} (acc. {accuracy_i*100:.2f}): {explanation_class_i}')
                    print()

        return


if __name__ == '__main__':
    unittest.main()
