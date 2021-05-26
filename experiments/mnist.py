# %% md

# The Multiparameter Intelligent Monitoring in Intensive Care II (MIMIC-II)

# %% md

## Import libraries

# %%
import glob
import sys
sys.path.append('..')
import os
import pandas as pd
import numpy as np
import time
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import seaborn as sns

from torch_explain.models.explainer import MuExplainer
from torch_explain.logic.metrics import formula_consistency
from experiments.data.load_datasets import load_mnist

# %% md

## Import MIMIC-II dataset

# %%
x, y, concept_names = load_mnist()


dataset = TensorDataset(x, y)

train_size = int(len(dataset) * 0.9)
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=len(train_data))
val_loader = DataLoader(val_data, batch_size=len(val_data))
test_loader = DataLoader(test_data, batch_size=len(test_data))

n_concepts = next(iter(train_loader))[0].shape[1]
n_classes = 2

print(concept_names)
print(n_concepts)
print(n_classes)

# %% md

## 5-fold cross-validation with explainer network

base_dir = f'./results/MNIST/explainer'
os.makedirs(base_dir, exist_ok=True)

n_seeds = 5
results_list = []
explanations = {i: [] for i in range(n_classes)}
for seed in range(n_seeds):
    seed_everything(seed)
    print(f'Seed [{seed + 1}/{n_seeds}]')
    train_loader = DataLoader(train_data, batch_size=len(train_data))
    val_loader = DataLoader(val_data, batch_size=len(val_data))
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    checkpoint_callback = ModelCheckpoint(dirpath=base_dir, monitor='val_loss', save_top_k=1)
    trainer = Trainer(max_epochs=100, gpus=1, auto_lr_find=True, deterministic=True,
                      check_val_every_n_epoch=1, default_root_dir=base_dir,
                      weights_save_path=base_dir, callbacks=[checkpoint_callback])
    model = MuExplainer(n_concepts=n_concepts, n_classes=n_classes, l1=0.0000001, temperature=5, lr=0.01,
                        explainer_hidden=[10], conceptizator='identity')

    start = time.time()
    trainer.fit(model, train_loader, val_loader)
    print(f"Concept mask: {model.model[0].concept_mask}")
    model.freeze()
    model_results = trainer.test(model, test_dataloaders=test_loader)
    for j in range(n_classes):
        n_used_concepts = sum(model.model[0].concept_mask[j] > 0.5)
        print(f"Extracted concepts: {n_used_concepts}")
    results, f = model.explain_class(val_loader, val_loader, test_loader, topk_explanations=5,
                                     x_to_bool=None, max_accuracy=True, concept_names=concept_names)
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

consistencies = []
for j in range(n_classes):
    consistencies.append(formula_consistency(explanations[j]))
explanation_consistency = np.mean(consistencies)

results_df = pd.DataFrame(results_list)
results_df['explanation_consistency'] = explanation_consistency
results_df.to_csv(os.path.join(base_dir, 'results_aware_mnist.csv'))
results_df

# %%

results_df.mean()

# %%

results_df.sem()
