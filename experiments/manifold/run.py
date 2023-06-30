from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import TensorDataset

from datasets.toy_manifold import manifold_toy_dataset
from model import ManifoldRelationalDCR
import torch
import os
import pytorch_lightning as pl

from torch_explain.logic.commons import Rule, Domain
from torch_explain.logic.grounding import DomainGrounder
from torch_explain.logic.indexing import Indexer


def main():

    random_seed = 42
    epochs = 500
    learning_rate = 0.01
    batch_size = 32
    limit_batches = 1.0
    input_features = 2
    emb_size = 5
    manifold_arity = 2
    num_classes = 2
    gpu = False
    crisp = True
    set_level_rules = False
    predict_relation = True
    dataset_name = "moon"

    results_dir = f"./results/"
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, 'model.pt')

    # data
    X, q_labels, q_names = manifold_toy_dataset(dataset_name, only_on_manifold=True, random_seed=random_seed, train=True)

    # logic
    points = Domain("points", [f'{i}' for i in torch.arange(X.shape[1]).tolist()])
    rule = Rule("phi", body=["r(X,Y)", "q(X)"], head=["q(Y)"], var2domain={"X": "points", "Y": "points"})
    grounder = DomainGrounder({"points": points.constants}, [rule])
    groundings = grounder.ground()
    indexer = Indexer(groundings, q_names)
    indexer.index_all()

    # loaders
    train_data = TensorDataset(X, q_labels)
    train_dl = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)

    # model
    model = ManifoldRelationalDCR(indexer=indexer, input_features=input_features, emb_size=emb_size, manifold_arity=manifold_arity,
                                  num_classes=num_classes, predict_relation=predict_relation, crisp=crisp,
                                  set_level_rules=set_level_rules, learning_rate=learning_rate)

    # training
    print(f'Running epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}')
    logger = TensorBoardLogger(save_dir=results_dir, name="lightning_logs")
    trainer = pl.Trainer(max_epochs=epochs,
                         #devices=1, accelerator="gpu",
                         limit_train_batches=limit_batches,
                         limit_val_batches=limit_batches,
                         logger=logger)
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=train_dl)
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
