import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from torch.nn.functional import one_hot
from torch_explain.logic.parser import serialize_rules


from  torch.utils.data import TensorDataset, DataLoader
from experiments.distant_supervision.model import DeepConceptReasoner, AdditionAdHocReasoner
from experiments.distant_supervision.datasets import create_single_digit_addition, addition_dataset
from torch_explain.logic.semantics import ProductTNorm, VectorLogic
import pandas as pd
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torchvision
import torch
import os

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# torch.autograd.set_detect_anomaly(True)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def load_data(number_digits):


    concept_names, explanations = create_single_digit_addition(number_digits)

    X, y = addition_dataset(True, number_digits)
    X_test, y_test = addition_dataset(False, number_digits)

    train_dataset = TensorDataset(*X, y)
    test_dataset = TensorDataset(*X_test, y_test)



    return train_dataset, test_dataset, concept_names, explanations


def main():

    epochs = 500
    learning_rate = 0.001
    batch_size = 32
    limit_batches = 1.0
    emb_size = 100
    number_digits = 2


    results_dir = f"./results/"
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, 'model.pt')

    train_data, test_data, concept_names, explanations = load_data(number_digits)



    train_dl = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True)




    model = DeepConceptReasoner(emb_size=emb_size, num_digits = number_digits,
                                concept_names=concept_names, explanations=explanations,
                                learning_rate=learning_rate, verbose=True)


    # if not os.path.exists(model_path):
    print(f'Running epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}')
    logger = TensorBoardLogger(save_dir=results_dir, name="lightning_logs")
    trainer = pl.Trainer(max_epochs=epochs,
                         #devices=1, accelerator="gpu",
                         limit_train_batches=limit_batches,
                         limit_val_batches=limit_batches,
                         logger=logger)
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=test_dl)
    torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load(model_path))

    # trainer.test(model, dataloaders=test_dl)

    accuracy_learner = accuracy_score(model.task_labels[0].cpu().detach() > 0.5, model.learner_preds[0].cpu().detach() > 0.5)
    accuracy_logic = accuracy_score(model.task_labels[0].cpu().detach() > 0.5, model.logic_preds[0].cpu().detach() > 0.5)
    accuracy_neural = accuracy_score(model.task_labels[0].cpu().detach() > 0.5, model.neural_preds[0].cpu().detach() > 0.5)
    accuracy_concept = accuracy_score(model.concept_labels[0].cpu().detach(), model.concept_preds[0].cpu().detach() > 0.5)

    results = pd.DataFrame([[accuracy_learner, accuracy_logic, accuracy_neural, accuracy_concept]], columns=['learner', 'logic', 'neural', 'concept'])
    results.to_csv(os.path.join(results_dir, 'accuracy.csv'))


if __name__ == '__main__':
    main()
