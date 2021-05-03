# %%
import glob
import sys
from shutil import rmtree

sys.path.append('..')
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from torch_explain.models.explainer import MuExplainer
from experiments.data.load_datasets import load_mimic

# %%

x, y, concept_names = load_mimic()

dataset = TensorDataset(x, y)

train_size = int(len(dataset) * 0.5)
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_data, batch_size=train_size)
val_loader = DataLoader(val_data, batch_size=val_size)
test_loader = DataLoader(test_data, batch_size=test_size)

n_concepts = next(iter(train_loader))[0].shape[1]
n_classes = 2

print(concept_names)
print(n_concepts)
print(n_classes)

# %%
rmtree('./results/mimic-ii', ignore_errors=True)
seed_everything(42)

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
results_list = []
for split, (trainval_index, test_index) in enumerate(
        skf.split(x.cpu().detach().numpy(), y.argmax(dim=1).cpu().detach().numpy())):
    print(f'Split [{split + 1}/{n_splits}]')
    x_trainval, x_test = torch.FloatTensor(x[trainval_index]), torch.FloatTensor(x[test_index])
    y_trainval, y_test = torch.FloatTensor(y[trainval_index]), torch.FloatTensor(y[test_index])
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.5, random_state=42)
    print(f'{len(y_train)}/{len(y_val)}/{len(y_test)}')

    base_dir = f'./results/mimic-ii/explainer-{split}'
    os.makedirs(base_dir, exist_ok=True)

    train_data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_val, y_val)
    test_data = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_data, batch_size=train_size)
    val_loader = DataLoader(val_data, batch_size=val_size)
    test_loader = DataLoader(test_data, batch_size=test_size)

    checkpoint_callback = ModelCheckpoint(dirpath=base_dir, monitor='val_loss', save_top_k=1)
    trainer = Trainer(min_epochs=400, max_epochs=400, gpus=1, auto_lr_find=True, deterministic=True,
                      check_val_every_n_epoch=1, default_root_dir=base_dir,
                      weights_save_path=base_dir, callbacks=[checkpoint_callback])
    model = MuExplainer(n_concepts=n_concepts, n_classes=n_classes,
                        l1=0.01,
                        explainer_hidden=[100, 100],
                        lr=0.01)

    path = glob.glob(f'{base_dir}/*.ckpt')
    if path:
        model = MuExplainer.load_from_checkpoint(path[0])
    trainer.fit(model, train_loader, val_loader)
    model.freeze()
    model_results = trainer.test(model, test_dataloaders=test_loader)
    results, results_full = model.explain_class(val_loader, test_loader,
                                                topk_explanations=1,
                                                max_minterm_complexity=1,
                                                concept_names=concept_names)
    results['model_accuracy'] = model_results[0]['test_acc']
    print(results_full)
    print(f"Explanation: {results_full[0]['explanation']}")
    print(f"Explanation: {results_full[0]['explanation_accuracy']}")
    print(f"Explanation: {results_full[1]['explanation']}")
    print(f"Explanation: {results_full[1]['explanation_accuracy']}")

    plt.figure(figsize=[8, 4])
    plt.subplot(1, 2, 1)
    plt.title(f'Alpha - {results["model_accuracy"]:.4f}')
    # sns.heatmap(model.model[0].weight[0].abs())
    sns.distplot(model.model[0].beta[0])
    plt.subplot(1, 2, 2)
    plt.title(f'Alpha - {results["model_accuracy"]:.4f}')
    # sns.heatmap(model.model[0].weight[1].abs())
    sns.distplot(model.model[0].alpha[1])
    plt.tight_layout()
    plt.savefig(f'./results/mimic-ii/l1_alpha_{split}.png')
    plt.savefig(f'./results/mimic-ii/l1_alpha_{split}.pdf')
    plt.show()
    #
    # sys.exit()

    results_list.append(results)

# %%

results_df = pd.DataFrame(results_list)
print(results_df)
# print(list(results_df['explanation'].values))
print(results_df.mean())

# %%

dt_scores, rf_scores = [], []
for split, (trainval_index, test_index) in enumerate(
        skf.split(x.cpu().detach().numpy(), y.argmax(dim=1).cpu().detach().numpy())):
    print(f'Split [{split + 1}/{n_splits}]')
    x_trainval, x_test = x[trainval_index], x[test_index]
    y_trainval, y_test = y[trainval_index].argmax(dim=1), y[test_index].argmax(dim=1)

    dt_model = DecisionTreeClassifier(max_depth=3, random_state=split)
    dt_model.fit(x_trainval, y_trainval)
    dt_scores.append(dt_model.score(x_test, y_test))

    rf_model = RandomForestClassifier(random_state=split)
    rf_model.fit(x_trainval, y_trainval)
    rf_scores.append(rf_model.score(x_test, y_test))

print(f'Random forest scores: {np.mean(rf_scores)} (+/- {np.std(rf_scores)})')
print(f'Decision tree scores: {np.mean(dt_scores)} (+/- {np.std(dt_scores)})')
print(f'Mu net scores (model): {results_df["model_accuracy"].mean()} (+/- {results_df["model_accuracy"].std()})')
print(f'Mu net scores (exp): {results_df["explanation_accuracy"].mean()} (+/- {results_df["explanation_accuracy"].std()})')
