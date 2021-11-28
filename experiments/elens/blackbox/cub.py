import os
import pandas as pd
import numpy as np
import time
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything

from torch_explain.models.explainer import Explainer
from torch_explain.logic.metrics import formula_consistency
from experiments.data.load_datasets import load_cub

train_data, val_data, test_data, concept_names = load_cub('../data')

train_loader = DataLoader(train_data, batch_size=len(train_data))
val_loader = DataLoader(val_data, batch_size=len(val_data))
test_loader = DataLoader(test_data, batch_size=len(test_data))
n_concepts = next(iter(train_loader))[0].shape[1]
n_classes = next(iter(train_loader))[1].shape[1]
print(concept_names)
print(n_concepts)
print(n_classes)

# %% md

## 5-fold cross-validation with explainer network

base_dir = f'./results/CUB/blackbox'
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
    trainer = Trainer(max_epochs=500, gpus=1, auto_lr_find=True, deterministic=True,
                      check_val_every_n_epoch=1, default_root_dir=base_dir,
                      weights_save_path=base_dir, callbacks=[checkpoint_callback])
    model = Explainer(n_concepts=n_concepts, n_classes=n_classes, l1=0, lr=0.01, explainer_hidden=[10])

    trainer.fit(model, train_loader, val_loader)
    print(f"Concept mask: {model.model[0].concept_mask}")
    model.freeze()
    model_results = trainer.test(model, test_dataloaders=test_loader)
    for j in range(n_classes):
        n_used_concepts = sum(model.model[0].concept_mask[j] > 0.5)
        print(f"Extracted concepts: {n_used_concepts}")
    results = {}
    results['model_accuracy'] = model_results[0]['test_acc']

    results_list.append(results)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(base_dir, 'results_aware_cub.csv'))

results_df = pd.DataFrame(results_list)
results_df.to_csv(os.path.join(base_dir, 'results_aware_cub.csv'))
results_df

# %%

results_df.mean()

# %%

results_df.sem()


print(f'Mu net scores (model): {results_df["model_accuracy"].mean()} (+/- {results_df["model_accuracy"].std()})')