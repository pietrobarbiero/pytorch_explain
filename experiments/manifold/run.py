from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset

from datasets.toy_manifold import manifold_toy_dataset
from model import ManifoldRelationalDCR
import torch
import os
import torch.nn.functional as F
import pytorch_lightning as pl


def main():

    epochs = 500
    learning_rate = 0.01
    batch_size = 32
    limit_batches = 1.0
    emb_size = 20
    number_digits = 2
    gpu = False
    crisp = True
    set_level_rules = False
    dataset_name = "moon"

    results_dir = f"./results/"
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, 'model.pt')

    train_data = manifold_toy_dataset(dataset_name, only_on_manifold=True, random_seed=42, train=True)
    test_data = manifold_toy_dataset(dataset_name, only_on_manifold=True, random_seed=84, train=False)

    train_dl = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True)

    model = ManifoldRelationalDCR(input_features=2, emb_size=8, manifold_arity=2,
                                  num_classes=2, predict_relation=False, crisp=crisp,
                                  set_level_rules=set_level_rules, learning_rate=learning_rate)

    # if not os.path.exists(model_path):
    print(f'Running epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}')
    logger = TensorBoardLogger(save_dir=results_dir, name="lightning_logs")
    trainer = pl.Trainer(max_epochs=epochs,
                         #devices=1, accelerator="gpu",
                         limit_train_batches=limit_batches,
                         limit_val_batches=limit_batches,
                         logger=logger)
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=train_dl)
    trainer.test(model=model, dataloaders=test_dl)
    torch.save(model.state_dict(), model_path)



    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    # loss_form_c = torch.nn.BCELoss()
    # loss_form_y = torch.nn.BCELoss()
    # model.train()
    # for epoch in range(501):
    #     optimizer.zero_grad()
    #     c_pred, y_pred = model(X, body_index, head_index)
    #
    #     concept_loss = loss_form_c(c_pred, c_train)
    #     task_loss = loss_form_y(y_pred, y_train)
    #     loss = concept_loss + 0.5*task_loss
    #     loss.backward()
    #     optimizer.step()
    #
    #     # compute accuracy
    #     if epoch % 100 == 0:
    #         task_accuracy = accuracy_score(y_train, y_pred > 0.5)
    #         concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
    #         print(f'Epoch {epoch}: loss {loss:.4f} task accuracy: {task_accuracy:.4f} concept accuracy: {concept_accuracy:.4f}')


if __name__ == '__main__':
    main()
