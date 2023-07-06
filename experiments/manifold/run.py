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
    emb_size = 10
    manifold_arity = 2
    num_classes = 1
    num_relations = 2
    gpu = False
    crisp = True
    set_level_rules = False
    predict_relation = True
    temperature = 100
    dataset_name = "moon"

    results_dir = f"./results/"
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, 'model.pt')

    # data
    X, q_labels, q_names = manifold_toy_dataset(dataset_name, only_on_manifold=False, random_seed=random_seed,
                                                train=True, perc_super=1, n_samples=30)

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
    model = ManifoldRelationalDCR(indexer=indexer, input_features=input_features, emb_size=emb_size,
                                  manifold_arity=manifold_arity, num_relations=num_relations, concept_names=rule.body,
                                  task_names=rule.head, temperature=temperature,
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
    trainer.test(model=model, dataloaders=train_dl)


if __name__ == '__main__':
    main()
