import os
import pandas as pd
import numpy as np
import time
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import seaborn as sns

from torch_explain.models.explainer import Explainer
from torch_explain.logic.metrics import formula_consistency
from experiments.data.load_datasets import load_mimic

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

# %% md

## 5-fold cross-validation with explainer network

# %%

seed_everything(42)

base_dir = f'./results/mimic-ii/explainer'
os.makedirs(base_dir, exist_ok=True)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
results_list = []
feature_selection = []
explanations = {i: [] for i in range(n_classes)}
for split, (trainval_index, test_index) in enumerate(skf.split(x.cpu().detach().numpy(),
                                                               y.argmax(dim=1).cpu().detach().numpy())):
    print(f'Split [{split + 1}/{n_splits}]')
    x_trainval, x_test = torch.FloatTensor(x[trainval_index]), torch.FloatTensor(x[test_index])
    y_trainval, y_test = torch.FloatTensor(y[trainval_index]), torch.FloatTensor(y[test_index])
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.2, random_state=42)
    print(f'{len(y_train)}/{len(y_val)}/{len(y_test)}')

    train_data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_val, y_val)
    test_data = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_data, batch_size=train_size)
    val_loader = DataLoader(val_data, batch_size=val_size)
    test_loader = DataLoader(test_data, batch_size=test_size)

    checkpoint_callback = ModelCheckpoint(dirpath=base_dir, monitor='val_loss', save_top_k=1)
    trainer = Trainer(max_epochs=200, gpus=1, auto_lr_find=True, deterministic=True,
                      check_val_every_n_epoch=1, default_root_dir=base_dir,
                      weights_save_path=base_dir, callbacks=[checkpoint_callback])
    model = Explainer(n_concepts=n_concepts, n_classes=n_classes, l1=1e-3, lr=0.01,
                      explainer_hidden=[20], temperature=0.7)

    start = time.time()
    trainer.fit(model, train_loader, val_loader)
    print(f"Gamma: {model.model[0].concept_mask}")
    model.freeze()
    model_results = trainer.test(model, test_dataloaders=test_loader)
    for j in range(n_classes):
        n_used_concepts = sum(model.model[0].concept_mask[j] > 0.5)
        print(f"Extracted concepts: {n_used_concepts}")
    results, f = model.explain_class(val_loader, train_loader, test_loader,
                                     topk_explanations=10,
                                     concept_names=concept_names)
    end = time.time() - start
    results['model_accuracy'] = model_results[0]['test_acc']
    results['extraction_time'] = end

    results_list.append(results)
    extracted_concepts = []
    all_concepts = model.model[0].concept_mask[0] > 0.5
    common_concepts = model.model[0].concept_mask[0] > 0.5
    for j in range(n_classes):
        n_used_concepts = sum(model.model[0].concept_mask[j] > 0.5)
        print(f"Extracted concepts: {n_used_concepts}")
        print(f"Explanation: {f[j]['explanation']}")
        print(f"Explanation accuracy: {f[j]['explanation_accuracy']}")
        explanations[j].append(f[j]['explanation'])
        extracted_concepts.append(n_used_concepts)
        all_concepts += model.model[0].concept_mask[j] > 0.5
        common_concepts *= model.model[0].concept_mask[j] > 0.5

    results['extracted_concepts'] = np.mean(extracted_concepts)
    results['common_concepts_ratio'] = sum(common_concepts) / sum(all_concepts)

    # compare against standard feature selection
    i_mutual_info = mutual_info_classif(x_trainval, y_trainval[:, 1])
    i_chi2 = chi2(x_trainval, y_trainval[:, 1])[0]
    i_chi2[np.isnan(i_chi2)] = 0
    lasso = LassoCV(cv=5, random_state=0).fit(x_trainval, y_trainval[:, 1])
    i_lasso = np.abs(lasso.coef_)
    i_mu = model.model[0].concept_mask[1]
    df = pd.DataFrame(np.hstack([
        i_mu.numpy(),
        i_mutual_info / np.max(i_mutual_info),
        i_chi2 / np.max(i_chi2),
        i_lasso / np.max(i_lasso),
    ]).T, columns=['feature importance'])
    df['method'] = 'explainer'
    df.iloc[90:, 1] = 'MI'
    df.iloc[180:, 1] = 'CHI2'
    df.iloc[270:, 1] = 'Lasso'
    df['feature'] = np.hstack([np.arange(0, 90)] * 4)
    feature_selection.append(df)

consistencies = []
for j in range(n_classes):
    consistencies.append(formula_consistency(explanations[j]))
explanation_consistency = np.mean(consistencies)

feature_selection = pd.concat(feature_selection, axis=0)

# %% md

## Print results

# %%

f1 = feature_selection[feature_selection['feature'] <= 30]
f2 = feature_selection[(feature_selection['feature'] > 30) & (feature_selection['feature'] <= 60)]
f3 = feature_selection[feature_selection['feature'] > 60]

# %%

plt.figure(figsize=[10, 10])
plt.subplot(1, 3, 1)
ax = sns.barplot(y=f1['feature'], x=f1.iloc[:, 0],
                 hue=f1['method'], orient='h', errwidth=0.5, errcolor='k')
ax.get_legend().remove()
plt.subplot(1, 3, 2)
ax = sns.barplot(y=f2['feature'], x=f2.iloc[:, 0],
                 hue=f2['method'], orient='h', errwidth=0.5, errcolor='k')
plt.xlabel('')
ax.get_legend().remove()
plt.subplot(1, 3, 3)
sns.barplot(y=f3['feature'], x=f3.iloc[:, 0],
            hue=f3['method'], orient='h', errwidth=0.5, errcolor='k')
plt.xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'barplot_mimic.png'))
plt.savefig(os.path.join(base_dir, 'barplot_mimic.pdf'))
plt.show()

# %%

plt.figure(figsize=[6, 4])
sns.boxplot(x=feature_selection.iloc[:, 1], y=feature_selection.iloc[:, 0])
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'boxplot_mimic.png'))
plt.savefig(os.path.join(base_dir, 'boxplot_mimic.pdf'))
plt.show()

# %%

results_df = pd.DataFrame(results_list)
results_df['explanation_consistency'] = explanation_consistency
results_df.to_csv(os.path.join(base_dir, 'results_aware_mimic.csv'))
results_df

# %%

results_df.mean()

# %%

results_df.sem()

# %% md

## Compare with out-of-the-box models

# %%

dt_scores, rf_scores = [], []
for split, (trainval_index, test_index) in enumerate(
        skf.split(x.cpu().detach().numpy(), y.argmax(dim=1).cpu().detach().numpy())):
    print(f'Split [{split + 1}/{n_splits}]')
    x_trainval, x_test = x[trainval_index], x[test_index]
    y_trainval, y_test = y[trainval_index].argmax(dim=1), y[test_index].argmax(dim=1)

    dt_model = DecisionTreeClassifier(max_depth=5, random_state=split)
    dt_model.fit(x_trainval, y_trainval)
    dt_scores.append(dt_model.score(x_test, y_test))

    rf_model = RandomForestClassifier(random_state=split)
    rf_model.fit(x_trainval, y_trainval)
    rf_scores.append(rf_model.score(x_test, y_test))

print(f'Random forest scores: {np.mean(rf_scores)} (+/- {np.std(rf_scores)})')
print(f'Decision tree scores: {np.mean(dt_scores)} (+/- {np.std(dt_scores)})')
print(f'Mu net scores (model): {results_df["model_accuracy"].mean()} (+/- {results_df["model_accuracy"].std()})')
print(
    f'Mu net scores (exp): {results_df["explanation_accuracy"].mean()} (+/- {results_df["explanation_accuracy"].std()})')
