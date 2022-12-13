import pandas as pd
import numpy as np
import os

import torch
from sklearn.metrics import accuracy_score
from torch.nn import BCELoss
from torch.nn.functional import one_hot

from experiments.rlens.model import DeepConceptReasoner
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

def load_data(dataset, fold, train_epochs):
    c = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_AdaptiveDropout_NoProbConcat_lambda_fold_{fold}/test_embedding_semantics_on_epoch_{train_epochs}.npy')
    c_emb = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_AdaptiveDropout_NoProbConcat_lambda_fold_{fold}/test_embedding_vectors_on_epoch_{train_epochs}.npy')
    y_cem = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/MixtureEmbModelSharedProb_AdaptiveDropout_NoProbConcat_lambda_fold_{fold}/test_model_output_on_epoch_{train_epochs}.npy')
    # c1 = np.load('./results/xor_activations_final_rerun/test_embedding_acts/c_test.npy')
    # c2 = np.load('./results/xor_activations_final_rerun/test_embedding_acts/c_val.npy')
    # y1 = np.load('./results/xor_activations_final_rerun/test_embedding_acts/y_test.npy')
    y = np.load(f'./results/{dataset}_activations_final_rerun/test_embedding_acts/y_val.npy')

    c = torch.FloatTensor(c)
    c_emb = torch.FloatTensor(c_emb)
    c_emb = c_emb.reshape(c_emb.shape[0], c.shape[1], -1)
    y_cem = torch.FloatTensor(y_cem).squeeze()
    y = torch.LongTensor(y)
    y1h = one_hot(y).float()
    concept_names = [f'x{i}' for i in range(c.shape[1])]
    class_names = [f'y{i}' for i in range(y1h.shape[1])]
    in_concepts = c_emb.shape[1]
    out_concepts = y1h.shape[1]
    emb_size = c_emb.shape[2]
    np.random.seed(42)
    train_mask = set(np.random.choice(np.arange(c.shape[0]), int(c.shape[0] * 0.8), replace=False))
    test_mask = set(np.arange(c.shape[0])) - train_mask
    train_mask = torch.LongTensor(list(train_mask))
    test_mask = torch.LongTensor(list(test_mask))
    train_dataset = torch.utils.data.TensorDataset(c[train_mask], c_emb[train_mask], (c[train_mask]>0.5).float(), y1h[train_mask])
    test_dataset = torch.utils.data.TensorDataset(c[test_mask], c_emb[test_mask], (c[test_mask]>0.5).float(), y1h[test_mask])
    return train_dataset, test_dataset, in_concepts, out_concepts, emb_size, concept_names, class_names


def main():
    datasets = ['xor', 'trig', 'vec']
    # datasets = ['vec']
    folds = [i+1 for i in range(5)]
    train_epochs = 500
    epochs = 500
    learning_rate = 0.008
    batch_size = 500
    limit_batches = 1.0
    for dataset in datasets:
        for fold in folds:
            results_dir = f"./results/{dataset}/{fold}/"
            os.makedirs(results_dir, exist_ok=True)
            model_path = os.path.join(results_dir, 'model.pt')

            train_data, test_data, in_concepts, out_concepts, emb_size, concept_names, class_names = load_data(dataset, fold, train_epochs)
            train_dl = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
            test_dl = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True)

            model = DeepConceptReasoner(in_concepts=in_concepts, out_concepts=out_concepts, emb_size=emb_size,
                                        concept_names=concept_names, class_names=class_names,
                                        learning_rate=learning_rate, loss_form=BCELoss(),
                                        learner_epochs=epochs//2, verbose=True)
            if not os.path.exists(model_path):
                # # saves top-K checkpoints based on "val_loss" metric
                # checkpoint_callback = ModelCheckpoint(
                #     save_top_k=1,
                #     monitor="val_loss",
                #     mode="min",
                #     dirpath=results_dir,
                #     filename="hat-{epoch:02d}-{val_loss:.2f}",
                # )
                print(f'Running epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}')
                logger = TensorBoardLogger(save_dir=results_dir, name="lightning_logs")
                trainer = pl.Trainer(max_epochs=epochs,
                                     # precision=16,
                                     # check_val_every_n_epoch=5,
                                     # accumulate_grad_batches=4,
                                     devices=1, accelerator="gpu",
                                     limit_train_batches=limit_batches,
                                     limit_val_batches=limit_batches,
                                     # callbacks=[checkpoint_callback],
                                     logger=logger)
                trainer.fit(model=model, train_dataloaders=train_dl)
                torch.save(model.state_dict(), model_path)

            model.load_state_dict(torch.load(model_path))
            trainer = pl.Trainer(devices=1, accelerator="gpu",
                                 # precision=16,
                                 limit_test_batches=limit_batches)
            trainer.test(model, test_dataloaders=test_dl)

            accuracy_learner = accuracy_score(model.task_labels[0].cpu().detach() > 0.5, model.learner_preds[0].cpu().detach() > 0.5)
            accuracy_logic = accuracy_score(model.task_labels[0].cpu().detach() > 0.5, model.logic_preds[0].cpu().detach() > 0.5)
            accuracy_neural = accuracy_score(model.task_labels[0].cpu().detach() > 0.5, model.neural_preds[0].cpu().detach() > 0.5)
            accuracy_concept = accuracy_score(model.concept_labels[0].cpu().detach(), model.concept_preds[0].cpu().detach() > 0.5)

            results = pd.DataFrame([[accuracy_learner, accuracy_logic, accuracy_neural, accuracy_concept]], columns=['learner', 'logic', 'neural', 'concept'])
            results.to_csv(os.path.join(results_dir, 'accuracy.csv'))


if __name__ == '__main__':
    main()
