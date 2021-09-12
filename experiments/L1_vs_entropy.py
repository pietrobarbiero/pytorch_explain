import torch
from pytorch_lightning import seed_everything
from torch.nn.functional import one_hot
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

from matplotlib import rc
rc('text', usetex=True)

import torch_explain as te
from torch_explain.logic.metrics import test_explanation, complexity
from torch_explain.logic.nn import entropy, psi
from torch_explain.nn.functional import prune_equal_fanin

RESULTS_DIR = './results/L1_vs_entropy'
# RESULTS_DIR = './experiments/results/L1_vs_entropy'
seed_everything(51)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

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
    concept_names = ['x1', 'x2', 'x3', 'x4']  # , 'hand', 'radio']
    target_class_names = ['y', '¬y', 'z', '¬z']

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    loss_form = torch.nn.BCEWithLogitsLoss()
    model.train()

    max_epochs = 18001

    for epoch in range(max_epochs):
        # train step
        optimizer.zero_grad()
        y_pred = model(x).squeeze(-1)
        loss = loss_form(y_pred, y) + 0.0001 * te.nn.functional.entropy_logic_loss(model)
        # loss = loss_form(y_pred, y.argmax(dim=1)) + 0.00001 * te.nn.functional.entropy_logic_loss(model)
        # loss = loss_form(y_pred, y) + 0.001 * te.nn.functional.entropy_logic_loss(model)
        loss.backward()
        optimizer.step()

        # print()
        # print(layers[0].weight.grad[0].norm(dim=1))
        # print(layers[0].weight.grad[1].norm(dim=1))
        # print()

        # compute accuracy
        if epoch % 100 == 0:
            accuracy = (y_pred > 0.5).eq(y).sum().item() / (y.size(0) * y.size(1))

            target_class = 0
            explanation_class_1, _ = entropy.explain_class(model, x, y1h, x, y1h, target_class, concept_names=concept_names)
            if explanation_class_1: explanation_class_1 = explanation_class_1.replace('&', '∧').replace('|', '∨').replace('~', '¬')
            explanation_class_1 = f'∀x: {explanation_class_1} ↔ {target_class_names[target_class]}'
            target_class = 1
            explanation_class_2, _ = entropy.explain_class(model, x, y1h, x, y1h, target_class, concept_names=concept_names)
            if explanation_class_2: explanation_class_2 = explanation_class_2.replace('&', '∧').replace('|', '∨').replace('~', '¬')
            explanation_class_2 = f'∀x: {explanation_class_2} ↔ {target_class_names[target_class]}'
            target_class = 2
            explanation_class_3, _ = entropy.explain_class(model, x, y1h, x, y1h, target_class, concept_names=concept_names)
            if explanation_class_3: explanation_class_3 = explanation_class_3.replace('&', '∧').replace('|', '∨').replace('~', '¬')
            explanation_class_3 = f'∀x: {explanation_class_3} ↔ {target_class_names[target_class]}'
            target_class = 3
            explanation_class_4, _ = entropy.explain_class(model, x, y1h, x, y1h, target_class, concept_names=concept_names)
            if explanation_class_4: explanation_class_4 = explanation_class_4.replace('&', '∧').replace('|', '∨').replace('~', '¬')
            explanation_class_4 = f'∀x: {explanation_class_4} ↔ {target_class_names[target_class]}'

            # update loss and accuracy
            print(f'Epoch {epoch}: loss {loss.item():.4f} train accuracy: {accuracy * 100:.2f}')
            print(f'\tAlphas class 1: {layers[0].alpha_norm}')
            print()

    df = pd.DataFrame(layers[0].alpha_norm.detach().numpy(),
                      index=['y', '¬y', 'z', '¬z'],
                      columns=['x1', 'x2', 'x3', 'x4'])

    plt.figure(figsize=[4, 3])
    plt.title(r"Entropy concept scores $\tilde{\alpha}$")
    sns.heatmap(df, annot=True, fmt=".4f", vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'entropy_heatmap.png'))
    plt.savefig(os.path.join(RESULTS_DIR, 'entropy_heatmap.pdf'))
    plt.show()


    layers = [
        te.nn.EntropyLinear(x.shape[1], 20, n_classes=y1h.shape[1], temperature=0.3),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(20, 10),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(10, 1),
    ]
    model = torch.nn.Sequential(*layers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    loss_form = torch.nn.BCEWithLogitsLoss()
    model.train()

    for epoch in range(max_epochs):
        # train step
        optimizer.zero_grad()
        y_pred = model(x).squeeze(-1)
        loss = loss_form(y_pred, y) + 0.0001 * te.nn.functional.l1_loss(model)
        loss.backward()
        optimizer.step()

        # compute accuracy
        if epoch % 100 == 0:
            accuracy = (y_pred > 0.5).eq(y).sum().item() / (y.size(0) * y.size(1))

            # update loss and accuracy
            print(f'Epoch {epoch}: loss {loss.item():.4f} train accuracy: {accuracy * 100:.2f}')
            print(f'\tAlphas class 1: {layers[0].alpha_norm}')
            print()

    df = pd.DataFrame(layers[0].alpha_norm.detach().numpy(),
                      index=['y', '¬y', 'z', '¬z'],
                      columns=['x1', 'x2', 'x3', 'x4'])

    plt.figure(figsize=[4, 3])
    plt.title(r"L1$^*$ concept scores $\tilde{\alpha}$")
    sns.heatmap(df, annot=True, fmt=".4f", vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'L1_heatmap.png'))
    plt.savefig(os.path.join(RESULTS_DIR, 'L1_heatmap.pdf'))
    plt.show()



    layers = [
        torch.nn.Linear(x.shape[1], 20),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(20, 10),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(10, y.shape[1]),
    ]
    model = torch.nn.Sequential(*layers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    loss_form = torch.nn.BCEWithLogitsLoss()
    model.train()

    for epoch in range(max_epochs):
        # train step
        optimizer.zero_grad()
        y_pred = model(x).squeeze(-1)
        loss = loss_form(y_pred, y) + 0.0001 * te.nn.functional.l1_loss(model)
        loss.backward()
        optimizer.step()

        # compute accuracy
        if epoch % 100 == 0:
            accuracy = (y_pred > 0.5).eq(y).sum().item() / (y.size(0) * y.size(1))

            # update loss and accuracy
            print(f'Epoch {epoch}: loss {loss.item():.4f} train accuracy: {accuracy * 100:.2f}')
            print(f'\tAlphas class 1: {layers[0].weight}')
            print()

    df = pd.DataFrame(layers[0].weight.detach().numpy(),
                      index=[f'h{i}' for i in range(layers[0].weight.shape[0])],
                      columns=['x1', 'x2', 'x3', 'x4'])


    plt.figure(figsize=[4, 6])
    plt.title(r"L1 weights $W$")
    sns.heatmap(df, annot=True, fmt=".4f")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'L1_linear_heatmap.png'))
    plt.savefig(os.path.join(RESULTS_DIR, 'L1_linear_heatmap.pdf'))
    plt.show()

    # print(layers[0].alpha_norm)
    # print(layers[0].alpha_norm[0])
    # print(layers[0].alpha_norm[1])
    # print(layers[0].alpha_norm[2])
    # print(layers[0].alpha_norm[3])


if __name__ == '__main__':
    main()
