import joblib
import os
import sys
sys.path.append('../../')
import copy

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.nn import BCELoss, CrossEntropyLoss, Sequential, LeakyReLU, Linear, Sigmoid, Softmax
from torch.nn.functional import one_hot
from torchvision.models import resnet18, resnet34, resnet50, densenet121
from torch.utils.data import DataLoader, TensorDataset
from torch_explain.nn import ConceptEmbeddings, semantics
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch_explain.nn.vector_logic import NeSyLayer, to_boolean
from experiments.data.CUB200.cub_loader import load_data, find_class_imbalance
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import WandbLogger
import wandb
from prettytable import PrettyTable

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
BASE_DIR = '../data/CUB200/class_attr_data_10/'
sweep_wandb_logger = WandbLogger(project="cub_concept_sweep")
# Set up your default hyperparameters
GPU = 1

hyperparameter_defaults = dict(
    max_epochs=150,
    patience=7,
    batch_size=128,
    num_workers=12,
    emb_size=16,
    extra_dims=0,
    concept_loss_weight=1,
    normalize_loss=False,
    learning_rate=0.03,
    weight_decay=4e-05,
    scheduler_step=20,
    weight_loss=True,
    pretrain_model=True,
    c_extractor_arch="resnet34",
    optimizer="sgd",
    bool=False,
    early_stopping_monitor="val_loss",
    early_stopping_mode="min",
    early_stopping_delta=0.0,
    masked=False,
    architecture="SplitEmbModel",
)

def hyperparameter_sweep():
    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults)
    # Access all hyperparameter values through wandb.config
    config = wandb.config

    # Load all the data
    train_data_path = os.path.join(BASE_DIR, 'train.pkl')
    if config["weight_loss"]:
        imbalance = find_class_imbalance(train_data_path, True)
    else:
        imbalance = None
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')

    train_dl = load_data(
        pkl_paths=[train_data_path],
        use_attr=True,
        no_img=False,
        batch_size=config["batch_size"],
        uncertain_label=False,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir='../data/CUB200/',
        num_workers=config["num_workers"],
    )
    val_dl = load_data(
        pkl_paths=[val_data_path],
        use_attr=True,
        no_img=False,
        batch_size=config["batch_size"],
        uncertain_label=False,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir='../data/CUB200/',
        num_workers=config["num_workers"],
    )

    sample = next(iter(train_dl))
    n_concepts, n_tasks = sample[2].shape[-1], 200

    train_model(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        config=config,
        train_dl=train_dl,
        val_dl=val_dl,
        imbalance=imbalance,
        result_dir=None,
        logger=sweep_wandb_logger,
    )


def train_model(
    n_concepts,
    n_tasks,
    config,
    train_dl,
    val_dl,
    result_dir=None,
    test_dl=None,
    split=None,
    imbalance=None,
    rerun=False,
    logger=None,
    project_name='cub_concept_training',
    seed=None,
):
    if seed is not None:
        seed_everything(split)

    if split is not None:
        full_run_name = f"{config['architecture']}{config.get('extra_name', '')}_{config['c_extractor_arch']}_fold_{split + 1}"
    else:
        full_run_name = f"{config['architecture']}{config.get('extra_name', '')}_{config['c_extractor_arch']}"
    print(f"[Training {full_run_name}]")
    print("config:")
    for key, val in config.items():
        print(f"\t{key} -> {val}")

    if config["architecture"] == "NormEmbModel":
        model_cls = NormEmbModel
        extra_params = {
            "emb_size": config["emb_size"],
        }
    elif config["architecture"] == "SplitEmbModel":
        model_cls = SplitEmbModel
        extra_params = {
            "emb_size": config["emb_size"],
            "masked": config["masked"],
        }
    elif config["architecture"] == "ConceptBottleneckModel":
        model_cls = ConceptBottleneckModel
        extra_params = {
            "bool": config["bool"],
            "extra_dims": config["extra_dims"],
        }
    elif config["architecture"] == "MaskedSplitEmbModel":
        model_cls = MaskedSplitEmbModel
        extra_params = {
            "bounded_norm": config["bounded_norm"],
            "emb_size": config["emb_size"],
        }
    else:
        raise ValueError(f'Invalid architecture "{config["architecture"]}"')

    if config["c_extractor_arch"] == "resnet18":
        c_extractor_arch = resnet18
    elif config["c_extractor_arch"] == "resnet34":
        c_extractor_arch = resnet34
    elif config["c_extractor_arch"] == "resnet50":
        c_extractor_arch = resnet50
    elif config["c_extractor_arch"] == "densenet121":
        c_extractor_arch = densenet121
    else:
        raise ValueError(f'Invalid model_to_use "{config["model_to_use"]}"')

    # Create model
    model = model_cls(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        weight_loss=torch.FloatTensor(imbalance) if config['weight_loss'] else None,
        concept_loss_weight=config['concept_loss_weight'],
        normalize_loss=config['normalize_loss'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        scheduler_step=config['scheduler_step'],
        pretrain_model=config['pretrain_model'],
        c_extractor_arch=c_extractor_arch,
        optimizer=config['optimizer'],
        **extra_params,
    )
    if result_dir:
        with wandb.init(
            project=project_name,
            name=full_run_name,
            config=config,
            reinit=True
        ) as run:
            model_saved_path = os.path.join(
                result_dir,
                f'{full_run_name}.pt'
            )
            trainer = pl.Trainer(
                gpus=GPU,
                max_epochs=config['max_epochs'],
                check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
                callbacks=[
                    EarlyStopping(
                        monitor=config["early_stopping_monitor"],
                        min_delta=config.get("early_stopping_delta", 0.00),
                        patience=config['patience'],
                        verbose=config.get("verbose", False),
                        mode=config["early_stopping_mode"],
                    ),
                ],
                # Only use the wandb logger when it is a fresh run
                logger=(
                    logger or
                    (WandbLogger(
                        name=full_run_name,
                        project=project_name,
                        save_dir=os.path.join(result_dir, "logs"),
                    ) if rerun or (not os.path.exists(model_saved_path)) else True)
                ),
            )
            if (not rerun) and os.path.exists(model_saved_path):
                # Then we simply load the model and proceed
                print("\tFound cached model... loading it")
                model.load_state_dict(torch.load(model_saved_path))
            else:
                # Else it is time to train it
                trainer.fit(model, train_dl, val_dl)
                torch.save(
                    model.state_dict(),
                    model_saved_path,
                )
            # freeze model and compute test accuracy
            if test_dl is not None:
                model.freeze()
                [test_results] = trainer.test(model, test_dl)
                c_accuracy, y_accuracy = test_results["test_c_accuracy"], test_results["test_y_accuracy"]
                c_auc, y_auc = test_results["test_c_auc"], test_results["test_y_auc"]
                c_f1, y_f1 = test_results["test_c_f1"], test_results["test_y_f1"]
                print(
                    f'{full_run_name} c_acc: {c_accuracy:.4f}, '
                    f'{full_run_name} c_auc: {c_auc:.4f}, '
                    f'{full_run_name} c_f1: {c_f1:.4f}, '
                    f'{full_run_name} y_acc: {y_accuracy:.4f}, '
                    f'{full_run_name} y_auc: {y_auc:.4f}, '
                    f'{full_run_name} y_f1: {y_f1:.4f}'
                )
            else:
                test_results = None
    else:
        trainer = pl.Trainer(
            gpus=GPU,
            max_epochs=config['max_epochs'],
            check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
            callbacks=[
                EarlyStopping(
                    monitor=config["early_stopping_monitor"],
                    min_delta=config.get("early_stopping_delta", 0.00),
                    patience=config['patience'],
                    verbose=config.get("verbose", False),
                    mode=config["early_stopping_mode"],
                ),
            ],
            logger=logger or True,
        )
        # Else it is time to train it
        trainer.fit(model, train_dl, val_dl)
        if test_dl is not None:
            model.freeze()
            [test_results] = trainer.test(model, test_dl)
            c_accuracy, y_accuracy = test_results["test_c_accuracy"], test_results["test_y_accuracy"]
            c_auc, y_auc = test_results["test_c_auc"], test_results["test_y_auc"]
            c_f1, y_f1 = test_results["test_c_f1"], test_results["test_y_f1"]
            print(
                f'{full_run_name} c_acc: {c_accuracy:.4f}, '
                f'{full_run_name} c_auc: {c_auc:.4f}, '
                f'{full_run_name} c_f1: {c_f1:.4f}, '
                f'{full_run_name} y_acc: {y_accuracy:.4f}, '
                f'{full_run_name} y_auc: {y_auc:.4f}, '
                f'{full_run_name} y_f1: {y_f1:.4f}'
            )
        else:
            test_results = None
    return model, test_results

def update_statistics(results, config, model, test_results):
    full_run_name = f"{config['architecture']}{config.get('extra_name', '')}"
    results.update({
        f'train_acc_{full_run_name}': model.loss_list,
        f'test_acc_y_{full_run_name}': test_results['test_y_accuracy'],
        f'test_auc_y_{full_run_name}': test_results['test_y_auc'],
        f'test_f1_y_{full_run_name}': test_results['test_y_f1'],
        f'test_acc_c_{full_run_name}': test_results['test_c_accuracy'],
        f'test_auc_c_{full_run_name}': test_results['test_c_auc'],
        f'test_f1_c_{full_run_name}': test_results['test_c_f1'],
        f'trainable_params_{config["architecture"]}': sum(p.numel() for p in model.parameters()),
    })


def main(
    rerun=False,
    result_dir='results/cub/',
    project_name='cub_concept_training',
):
    seed_everything(42)
    # parameters for data, model, and training
    og_config = dict(
        cv=5,
        max_epochs=300,
        patience=15,
        batch_size=128,
        num_workers=12,
        emb_size=16,
        extra_dims=0,
        concept_loss_weight=3.11, #1, #1.5,
        normalize_loss=False,
        learning_rate=0.015, #0.03,
        weight_decay=4e-05,
        scheduler_step=20,
        weight_loss=True,
        pretrain_model=True,
        c_extractor_arch="resnet34",
        optimizer="sgd",
        bool=False,
        early_stopping_monitor="val_loss", #"val_y_accuracy",
        early_stopping_mode="min",
        early_stopping_delta=0.0,
        masked=False,
    )

    train_data_path = os.path.join(BASE_DIR, 'train.pkl')
    if og_config['weight_loss']:
        imbalance = find_class_imbalance(train_data_path, True)
    else:
        imbalance = None

    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    test_data_path = train_data_path.replace('train.pkl', 'test.pkl')

    train_dl = load_data(
        pkl_paths=[train_data_path],
        use_attr=True,
        no_img=False,
        batch_size=og_config['batch_size'],
        uncertain_label=False,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir='../data/CUB200/',
        num_workers=og_config['num_workers'],
    )
    val_dl = load_data(
        pkl_paths=[val_data_path],
        use_attr=True,
        no_img=False,
        batch_size=og_config['batch_size'],
        uncertain_label=False,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir='../data/CUB200/',
        num_workers=og_config['num_workers'],
    )
    test_dl = load_data(
        pkl_paths=[test_data_path],
        use_attr=True,
        no_img=False,
        batch_size=og_config['batch_size'],
        uncertain_label=False,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir='../data/CUB200/',
        num_workers=og_config['num_workers'],
    )
    sample = next(iter(train_dl))
    n_concepts, n_tasks = sample[2].shape[-1], 200

    print("Training sample shape is:", sample[0].shape)
    print("Training label shape is:", sample[1].shape)
    print("Training concept shape is:", sample[2].shape)


    os.makedirs(result_dir, exist_ok=True)
    joblib.dump(og_config, os.path.join(result_dir, f'experiment_config.joblib'))

    results = {}
    for split in range(og_config["cv"]):
        print(f'Experiment {split+1}/{og_config["cv"]}')
        results[f'{split}'] = {}

        # train model *with* normalized masked split concept embeddings
        config = copy.deepcopy(og_config)
        config["architecture"] = "MaskedSplitEmbModel"
        config["extra_name"] = "Bounded"
        config["bounded_norm"] = True
        norm_bounded_split_emb_model, norm_bounded_split_emb_test_results = train_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            split=split,
            imbalance=imbalance,
            result_dir=result_dir,
            rerun=rerun,
            project_name=project_name,
            seed=split,
        )
        update_statistics(
            results[f'{split}'],
            config,
            norm_bounded_split_emb_model,
            norm_bounded_split_emb_test_results,
        )

        # train model *with* unbounded masked split concept embeddings
        config = copy.deepcopy(og_config)
        config["architecture"] = "MaskedSplitEmbModel"
        config["extra_name"] = "Unbounded"
        config["bounded_norm"] = False
        norm_unbounded_split_emb_model, norm_unbounded_split_emb_test_results = train_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            split=split,
            imbalance=imbalance,
            result_dir=result_dir,
            rerun=rerun,
            project_name=project_name,
            seed=split,
        )
        update_statistics(
            results[f'{split}'],
            config,
            norm_unbounded_split_emb_model,
            norm_unbounded_split_emb_test_results,
        )

        # train model *with* unormalized split concept embeddings
        config = copy.deepcopy(og_config)
        config["architecture"] = "SplitEmbModel"
        config["extra_name"] = ""
        split_emb_model, split_emb_test_results = train_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            split=split,
            imbalance=imbalance,
            result_dir=result_dir,
            rerun=rerun,
            project_name=project_name,
            seed=split,
        )
        update_statistics(results[f'{split}'], config, split_emb_model, split_emb_test_results)

        # train model *with* normalized concept embeddings
        config = copy.deepcopy(og_config)
        config["architecture"] = "NormEmbModel"
        config["extra_name"] = ""
        norm_emb_model, norm_emb_test_results = train_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            split=split,
            imbalance=imbalance,
            result_dir=result_dir,
            rerun=rerun,
            project_name=project_name,
            seed=split,
        )
        update_statistics(results[f'{split}'], config, norm_emb_model, norm_emb_test_results)

        # train model *without* embeddings (concepts are just *fuzzy* scalars)
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["bool"] = False
        config["extra_name"] = "Fuzzy"
        fuzzy_model, fuzzy_test_results = train_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            split=split,
            imbalance=imbalance,
            result_dir=result_dir,
            rerun=rerun,
            project_name=project_name,
            seed=split,
        )
        update_statistics(results[f'{split}'], config, fuzzy_model, fuzzy_test_results)

        # train model *without* embeddings but with extra capacity (concepts are just *fuzzy* scalars
        # and the model also has some extra capacity)
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["bool"] = False
        config["extra_dims"] = config['emb_size'] * n_concepts
        config["extra_name"] = "FuzzyExtraCapacity"
        extra_fuzzy_model, extra_fuzzy_test_results = train_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            split=split,
            imbalance=imbalance,
            result_dir=result_dir,
            rerun=rerun,
            project_name=project_name,
            seed=split,
        )
        update_statistics(results[f'{split}'], config, extra_fuzzy_model, extra_fuzzy_test_results)

        # train model *without* embeddings (concepts are just *Boolean* scalars)
        config = copy.deepcopy(og_config)
        config["architecture"] = "ConceptBottleneckModel"
        config["extra_name"] = "Bool"
        config["bool"] = True
        bool_model, bool_test_results = train_model(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            config=config,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            split=split,
            imbalance=imbalance,
            result_dir=result_dir,
            rerun=rerun,
            project_name=project_name,
            seed=split,
        )
        update_statistics(results[f'{split}'], config, bool_model, bool_test_results)

        # save results
        joblib.dump(results, os.path.join(result_dir, f'results.joblib'))

#     # And let's put all of the results into a nice table
#     bool_y_accs = []
#     bool_c_accs = []
#     fuzzy_y_accs = []
#     fuzzy_c_accs = []
#     emb_y_accs = []
#     emb_c_accs = []

#     bool_y_aucs = []
#     bool_c_aucs = []
#     fuzzy_y_aucs = []
#     fuzzy_c_aucs = []
#     emb_y_aucs = []
#     emb_c_aucs = []
#     for split, vals in results.items():
#         bool_y_accs.append(vals['test_acc_y_bool'])
#         bool_c_accs.append(vals['test_acc_c_bool'])
#         fuzzy_y_accs.append(vals['test_acc_y_fuzzy'])
#         fuzzy_c_accs.append(vals['test_acc_c_fuzzy'])
#         emb_y_accs.append(vals['test_acc_y_emb'])
#         emb_c_accs.append(vals['test_acc_c_emb'])

#         bool_y_aucs.append(vals['test_auc_y_bool'])
#         bool_c_aucs.append(vals['test_auc_c_bool'])
#         fuzzy_y_aucs.append(vals['test_auc_y_fuzzy'])
#         fuzzy_c_aucs.append(vals['test_auc_c_fuzzy'])
#         emb_y_aucs.append(vals['test_auc_y_emb'])
#         emb_c_aucs.append(vals['test_auc_c_emb'])
#     t = PrettyTable()
#     t.field_names = ["Method", "Task Accuracy", "Task AUC", "Concept Accuracy", "Concept AUC"]
#     t.add_row([
#         "Bool",
#         f'{np.mean(bool_y_accs)*100:.2f}% ± {2 * np.std(bool_y_accs)*100:.2f}%',
#         f'{np.mean(bool_y_aucs)*100:.2f}% ± {2 * np.std(bool_y_aucs)*100:.2f}%',
#         f'{np.mean(bool_c_accs)*100:.2f}% ± {2 * np.std(bool_c_accs)*100:.2f}%',
#         f'{np.mean(bool_c_aucs)*100:.2f}% ± {2 * np.std(bool_c_aucs)*100:.2f}%',
#     ])
#     t.add_row([
#         "Fuzzy",
#         f'{np.mean(fuzzy_y_accs)*100:.2f}% ± {2 * np.std(fuzzy_y_accs)*100:.2f}%',
#         f'{np.mean(fuzzy_y_aucs)*100:.2f}% ± {2 * np.std(fuzzy_y_aucs)*100:.2f}%',
#         f'{np.mean(fuzzy_c_accs)*100:.2f}% ± {2 * np.std(fuzzy_c_accs)*100:.2f}%',
#         f'{np.mean(fuzzy_c_aucs)*100:.2f}% ± {2 * np.std(fuzzy_c_aucs)*100:.2f}%',
#     ])
#     t.add_row([
#         "Embedding",
#         f'{np.mean(emb_y_accs)*100:.2f}% ± {2 * np.std(emb_y_accs)*100:.2f}%',
#         f'{np.mean(emb_y_aucs)*100:.2f}% ± {2 * np.std(emb_y_aucs)*100:.2f}%',
#         f'{np.mean(emb_c_accs)*100:.2f}% ± {2 * np.std(emb_c_accs)*100:.2f}%',
#         f'{np.mean(emb_c_aucs)*100:.2f}% ± {2 * np.std(emb_c_aucs)*100:.2f}%',
#     ])
#     print(t)
    return results


class NormEmbModel(pl.LightningModule):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=5,
        concept_loss_weight=0.01,
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        normalize_loss=True,
        scheduler_step=20,
        pretrain_model=True,
        c_extractor_arch=resnet50,
        optimizer="adam",
    ):
        super().__init__()
        model = c_extractor_arch(pretrained=pretrain_model)
        if c_extractor_arch == densenet121:
            model.classifier = Linear(1024, n_concepts)
        else:
            model.fc = Linear(512, n_concepts)
        self.x2c_model = Sequential(*[
            model,
            LeakyReLU(),
            ConceptEmbeddings(
                in_features=n_concepts,
                out_features=n_concepts,
                emb_size=emb_size,
                bias=True,
            ),
        ])
        self.c2y_model = Sequential(*[
            Linear(n_concepts * emb_size, n_tasks),
        ])
        self.loss_concept = BCELoss(weight=weight_loss)
        self.loss_task = CrossEntropyLoss()
        self.loss_list = []
        self.concept_loss_weight = concept_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.normalize_loss = normalize_loss
        self.scheduler_step = scheduler_step
        self.optimizer_name = optimizer

    def forward(self, x):
        c = self.x2c_model(x)
        y = self.c2y_model(torch.flatten(c, start_dim=1))
        c_sem = torch.norm(c, p=2, dim=-1)
        return c_sem, y, c

    def training_step(self, batch, batch_no):
        x, y, c = batch
        c_sem, y_pred, _ = self(x)
        concept_loss = self.loss_concept(c_sem, c)
        task_loss = self.loss_task(y_pred, y)
        loss = self.concept_loss_weight * concept_loss + task_loss
        if self.normalize_loss:
            loss = loss / (1 + self.concept_loss_weight * c.shape[-1])
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_sem,
            y_pred,
            c,
            y,
        )
        self.log("c_accuracy", c_accuracy, prog_bar=True)
        self.log("c_auc", c_auc)
        self.log("c_f1", c_f1)
        self.log("y_accuracy", y_accuracy, prog_bar=True)
        self.log("y_auc", y_auc)
        self.log("y_f1", y_f1)
        self.log("loss", loss)
        self.log("concept_loss", concept_loss)
        self.log("task_loss", task_loss)
        self.loss_list.append([c_accuracy, y_accuracy])
        return {
            "loss": loss,
            "log": {
                "c_accuracy": c_accuracy,
                "c_auc": c_auc,
                "c_f1": c_f1,
                "y_accuracy": y_accuracy,
                "y_auc": y_auc,
                "y_f1": y_f1,
                "concept_loss": concept_loss.detach(),
                "task_loss": task_loss.detach(),
                "loss": loss.detach(),
            },
        }

    def validation_step(self, batch, batch_no):
        x, y, c = batch
        c_sem, y_pred, _ = self(x)
        concept_loss = self.loss_concept(c_sem, c)
        task_loss = self.loss_task(y_pred, y)
        loss = self.concept_loss_weight * concept_loss + task_loss
        if self.normalize_loss:
            loss = loss / (1 + self.concept_loss_weight * c.shape[-1])
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_sem,
            y_pred,
            c,
            y,
        )
        self.log("val_c_accuracy", c_accuracy, prog_bar=True)
        self.log("val_c_auc", c_auc)
        self.log("val_c_f1", c_f1)
        self.log("val_y_accuracy", y_accuracy, prog_bar=True)
        self.log("val_y_auc", y_auc)
        self.log("val_y_f1", y_f1)
        self.log("val_loss", loss)
        self.log("val_concept_loss", concept_loss)
        self.log("val_task_loss", task_loss)
        self.log("val_avg_c_y_acc", (c_accuracy + y_accuracy) / 2)
        return {
            "val_loss": loss,
            "val_c_accuracy": c_accuracy,
            "val_c_auc": c_auc,
            "val_c_f1": c_f1,
            "val_y_accuracy": y_accuracy,
            "val_y_auc": y_auc,
            "val_y_f1": y_f1,
            "val_concept_loss": concept_loss.detach(),
            "val_task_loss": task_loss.detach(),
            "loss": loss.detach(),
            "val_avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
        }

    def test_step(self, batch, batch_no):
        x, y, c = batch
        c_sem, y_pred, _ = self(x)
        concept_loss = self.loss_concept(c_sem, c)
        task_loss = self.loss_task(y_pred, y)
        loss = self.concept_loss_weight * concept_loss + task_loss
        if self.normalize_loss:
            loss = loss / (1 + self.concept_loss_weight * c.shape[-1])
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_sem,
            y_pred,
            c,
            y,
        )
        self.log("test_c_accuracy", c_accuracy, prog_bar=True)
        self.log("test_c_auc", c_auc, prog_bar=True)
        self.log("test_c_f1", c_f1, prog_bar=True)
        self.log("test_y_accuracy", y_accuracy, prog_bar=True)
        self.log("test_y_auc", y_auc, prog_bar=True)
        self.log("test_y_f1", y_f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }

class SplitEmbModel(NormEmbModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=5,
        concept_loss_weight=0.01,
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        normalize_loss=True,
        scheduler_step=20,
        pretrain_model=True,
        c_extractor_arch=resnet50,
        optimizer="adam",
        masked=False,
    ):
        pl.LightningModule.__init__(self)
        self.pre_concept_model = c_extractor_arch(
            pretrained=pretrain_model
        )
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        for i in range(n_concepts):
            self.concept_context_generators.append(
                Linear(
                    list(self.pre_concept_model.modules())[-1].out_features,
                    emb_size,
                )
            )
            self.concept_prob_generators.append(
                Linear(
                    emb_size,
                    1,
                )
            )
        self.c2y_model = Sequential(*[
            Linear(n_concepts * (emb_size + 1), n_tasks),
        ])
        self.sig = torch.nn.Sigmoid()
        self.loss_concept = BCELoss(weight=weight_loss)
        self.loss_task = CrossEntropyLoss()
        self.loss_list = []
        self.concept_loss_weight = concept_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.normalize_loss = normalize_loss
        self.scheduler_step = scheduler_step
        self.optimizer_name = optimizer

    def forward(self, x):
        pre_c = self.pre_concept_model(x)
        probs = []
        contexts = []
        full_vectors = []
        for context_gen, prob_gen in zip(
            self.concept_context_generators,
            self.concept_prob_generators,
        ):
            context = context_gen(pre_c)
            contexts.append(context)
            probs.append(prob_gen(context))
            full_vectors.append(torch.cat(
                [contexts[-1], probs[-1]],
                axis=-1,
            ))
        c_sem = self.sig(torch.cat(probs, axis=-1))
        c = torch.cat(full_vectors, axis=-1)
        y = self.c2y_model(c)
        return c_sem, y, c

class MaskedSplitEmbModel(SplitEmbModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=5,
        concept_loss_weight=0.01,
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        normalize_loss=True,
        scheduler_step=20,
        pretrain_model=True,
        c_extractor_arch=resnet50,
        optimizer="adam",
        masked=False,
        bounded_norm=True,
        eps=1e-5,
    ):
        pl.LightningModule.__init__(self)
        self.pre_concept_model = c_extractor_arch(
            pretrained=pretrain_model
        )
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        for i in range(n_concepts):
            self.concept_context_generators.append(
                Linear(
                    list(self.pre_concept_model.modules())[-1].out_features,
                    emb_size,
                )
            )
            self.concept_prob_generators.append(
                Linear(
                    emb_size,
                    1,
                )
            )
        self.c2y_model = Sequential(*[
            Linear(n_concepts * (emb_size), n_tasks),
        ])
        self.sig = torch.nn.Sigmoid()
        self.loss_concept = BCELoss(weight=weight_loss)
        self.loss_task = CrossEntropyLoss()
        self.loss_list = []
        self.concept_loss_weight = concept_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.normalize_loss = normalize_loss
        self.scheduler_step = scheduler_step
        self.optimizer_name = optimizer
        self.bounded_norm = bounded_norm
        self.eps = eps

    def forward(self, x):
        pre_c = self.pre_concept_model(x)
        probs = []
        full_vectors = []
        for context_gen, prob_gen in zip(
            self.concept_context_generators,
            self.concept_prob_generators,
        ):
            context = context_gen(pre_c)
            probs.append(prob_gen(context))
            # Mask the context by multiplying it by the
            # probability that it is activated
            if self.bounded_norm:
                context = context / (torch.norm(context, p=2, dim=-1, keepdim=True) + self.eps)
            full_vectors.append(torch.cat(
                [context * self.sig(probs[-1])],
                axis=-1,
            ))
        c_sem = self.sig(torch.cat(probs, axis=-1))
        c = torch.cat(full_vectors, axis=-1)
        y = self.c2y_model(c)
        return c_sem, y, c


class ConceptBottleneckModel(pl.LightningModule):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        bool,
        concept_loss_weight=0.01,
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        normalize_loss=True,
        scheduler_step=20,
        pretrain_model=True,
        c_extractor_arch=resnet50,
        optimizer="adam",
        extra_dims=0,
    ):
        super().__init__()
        self.n_concepts = n_concepts
        model = c_extractor_arch(pretrained=pretrain_model)
        if c_extractor_arch == densenet121:
            model.classifier = Linear(1024, n_concepts + extra_dims)
        else:
            model.fc = Linear(512, n_concepts + extra_dims)
        self.x2c_model = Sequential(*[
            model,
            Sigmoid()
        ])
        self.c2y_model = Sequential(*[
            Linear(n_concepts + extra_dims, n_tasks),
        ])
        self.loss_concept = BCELoss(weight=weight_loss)
        self.loss_task = CrossEntropyLoss()
        self.bool = bool
        self.loss_list = []
        self.concept_loss_weight = concept_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.normalize_loss = normalize_loss
        self.scheduler_step = scheduler_step
        self.optimizer_name = optimizer
        self.extra_dims = extra_dims

    def forward(self, x):
        c = self.x2c_model(x)
        if self.bool:
            y = self.c2y_model((c > 0.5).float())
        else:
            y = self.c2y_model(c)
        return c, y

    def training_step(self, batch, batch_no):
        x, y, c = batch
        c_logits, y_logits = self(x)
        if self.extra_dims:
            c_logits = c_logits[:, :-self.extra_dims]
        concept_loss = self.loss_concept(c_logits, c)
        task_loss = self.loss_task(y_logits, y)
        loss = self.concept_loss_weight * concept_loss + task_loss
        if self.normalize_loss:
            loss = loss / (1 + self.concept_loss_weight * c.shape[-1])
        # compute accuracy
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_logits,
            y_logits,
            c,
            y,
        )
        self.log("c_accuracy", c_accuracy, prog_bar=True)
        self.log("c_auc", c_auc)
        self.log("c_f1", c_f1)
        self.log("y_accuracy", y_accuracy, prog_bar=True)
        self.log("y_auc", y_auc)
        self.log("y_f1", y_f1)
        self.log("loss", loss)
        self.log("concept_loss", concept_loss)
        self.log("task_loss", task_loss)
        self.loss_list.append([c_accuracy, y_accuracy])
        return {
            "loss": loss,
            "log": {
                "c_accuracy": c_accuracy,
                "c_auc": c_auc,
                "c_f1": c_f1,
                "y_accuracy": y_accuracy,
                "y_auc": y_auc,
                "y_f1": y_f1,
                "concept_loss": concept_loss.detach(),
                "task_loss": task_loss.detach(),
                "loss": loss.detach(),
            },
        }


    def validation_step(self, batch, batch_no):
        x, y, c = batch
        c_logits, y_logits = self(x)
        if self.extra_dims:
            c_logits = c_logits[:, :-self.extra_dims]
        concept_loss = self.loss_concept(c_logits, c)
        task_loss = self.loss_task(y_logits, y)
        loss = self.concept_loss_weight * concept_loss + task_loss
        if self.normalize_loss:
            loss = loss / (1 + self.concept_loss_weight * c.shape[-1])
        # compute accuracy
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_logits,
            y_logits,
            c,
            y,
        )
        self.log("val_c_accuracy", c_accuracy, prog_bar=True)
        self.log("val_c_auc", c_auc)
        self.log("val_c_f1", c_f1)
        self.log("val_y_accuracy", y_accuracy, prog_bar=True)
        self.log("val_y_auc", y_auc)
        self.log("val_y_f1", y_f1)
        self.log("val_loss", loss)
        self.log("val_concept_loss", concept_loss)
        self.log("val_task_loss", task_loss)
        return {
            "val_loss": loss,
            "val_c_accuracy": c_accuracy,
            "val_c_auc": c_auc,
            "val_c_f1": c_f1,
            "val_y_accuracy": y_accuracy,
            "val_y_auc": y_auc,
            "val_y_f1": y_f1,
            "val_concept_loss": concept_loss.detach(),
            "val_task_loss": task_loss.detach(),
            "loss": loss.detach(),
        }


    def test_step(self, batch, batch_no):
        x, y, c = batch
        c_logits, y_logits = self(x)
        if self.extra_dims:
            c_logits = c_logits[:, :-self.extra_dims]
        loss = self.concept_loss_weight * self.loss_concept(c_logits, c) + (
            self.loss_task(y_logits, y)
        )
        if self.normalize_loss:
            loss = loss / (1 + self.concept_loss_weight * c.shape[-1])
        # compute accuracy
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_logits,
            y_logits,
            c,
            y,
        )
        self.log("test_c_accuracy", c_accuracy, prog_bar=True)
        self.log("test_c_auc", c_auc, prog_bar=True)
        self.log("test_c_f1", c_f1, prog_bar=True)
        self.log("test_y_accuracy", y_accuracy, prog_bar=True)
        self.log("test_y_auc", y_auc, prog_bar=True)
        self.log("test_y_f1", y_f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }


def compute_accuracy(c_pred, y_pred, c_true, y_true):
    c_pred = c_pred.reshape(-1).cpu().detach() > 0.5
    y_probs = torch.nn.Softmax(dim=-1)(y_pred).cpu().detach()
    used_classes = np.unique(y_true.reshape(-1).cpu().detach())
    y_probs = y_probs[:, sorted(list(used_classes))]
    y_pred = y_pred.argmax(dim=-1).cpu().detach()
    c_true = c_true.reshape(-1).cpu().detach()
    y_true = y_true.reshape(-1).cpu().detach()
    c_accuracy = accuracy_score(c_true, c_pred)
    c_auc = roc_auc_score(c_true, c_pred, multi_class='ovo')
    c_f1 = f1_score(c_true, c_pred, average='macro')
    y_accuracy = accuracy_score(y_true, y_pred)
    y_auc = 0.0 # roc_auc_score(y_true, y_probs, multi_class='ovo')
    y_f1 = f1_score(y_true, y_pred, average='macro')
    return (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1)


if __name__ == '__main__':
    main(
        rerun=False,
        result_dir='results/cub_rerun/',
        project_name='cub_rerun_concept_training',
    )
#     hyperparameter_sweep()
