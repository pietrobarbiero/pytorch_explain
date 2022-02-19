import joblib
import os
import shutil
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from experiments.data.load_datasets import generate_dot, generate_trigonometry, generate_xor
from experiments.vlens.networks_toy import compute_accuracy, ToyNetEmbNorm, ToyNetEmbPlane, ToyNetFuzzyExtra, \
    ToyNetFuzzy, ToyNetBool


def main():
    # shutil.rmtree('./results/', ignore_errors=True)

    # parameters for data, model, and training
    batch_size = 3000
    batch_size_test = 1000
    emb_sizes = [2, 10, 50]
    max_epochs = 3000
    check_val_every_n_epoch = 100
    gpu = 1
    cv = 5
    models = {
        'embedding_plane': emb_sizes,
        'embedding_norm': emb_sizes,
        'fuzzy_extra': emb_sizes,
        'fuzzy': [1],
        'bool': [1],
    }
    datasets = {
        'dot': [generate_dot(batch_size), generate_dot(batch_size_test), generate_dot(batch_size_test)],
        'trigonometry': [generate_trigonometry(batch_size), generate_trigonometry(batch_size_test), generate_trigonometry(batch_size_test)],
        'xor': [generate_xor(batch_size), generate_xor(batch_size_test), generate_xor(batch_size_test)],
    }

    for dataset_name, dataset in datasets.items():
        result_dir = f'./results/{dataset_name}/'
        os.makedirs(result_dir, exist_ok=True)

        x_train, c_train, y_train = dataset[0]
        x_val, c_val, y_val = dataset[1]
        x_test, c_test, y_test = dataset[2]
        train_data = TensorDataset(x_train, c_train, y_train)
        val_data = TensorDataset(x_val, c_val, y_val)
        train_dl = DataLoader(train_data, batch_size=batch_size)
        val_dl = DataLoader(val_data, batch_size=batch_size_test)
        n_features, n_concepts, n_tasks = x_train.shape[1], c_train.shape[1], 1

        for model_name, emb_sizes_i in models.items():
            for emb_size in emb_sizes_i:
                results = {}
                for split in range(cv):
                    print(f'Experiment {dataset_name} {split + 1}/{cv} ({model_name} {emb_size})')

                    seed_everything(split)

                    # instantiate model
                    if model_name == 'embedding_plane':
                        model = ToyNetEmbPlane(n_features, n_concepts, n_tasks, emb_size)
                    elif model_name == 'embedding_norm':
                        model = ToyNetEmbNorm(n_features, n_concepts, n_tasks, emb_size)
                    elif model_name == 'fuzzy_extra':
                        model = ToyNetFuzzyExtra(n_features, n_concepts, n_tasks, emb_size)
                    elif model_name == 'fuzzy':
                        model = ToyNetFuzzy(n_features, n_concepts, n_tasks)
                    elif model_name == 'bool':
                        model = ToyNetBool(n_features, n_concepts, n_tasks)
                    else:
                        continue

                    # train model
                    checkpoint_callback = ModelCheckpoint(
                        monitor="val_loss",
                        dirpath=result_dir,
                        filename=f"{model_name}-{emb_size}-{split}",
                        save_top_k=1,
                        mode="min",
                    )
                    trainer = pl.Trainer(gpus=gpu, max_epochs=max_epochs, callbacks=[checkpoint_callback],
                                         check_val_every_n_epoch=check_val_every_n_epoch)
                    trainer.fit(model, train_dl, val_dl)

                    # freeze model and compute test accuracy
                    model.freeze()
                    model_loaded = model.__class__.load_from_checkpoint(os.path.join(result_dir, f"{model_name}-{emb_size}-{split}.ckpt"))
                    c_sem, y_sem, _, _, _, _ = model_loaded.forward(x_test)
                    c_accuracy, y_accuracy = compute_accuracy(c_sem, y_sem, c_test, y_test)
                    print(f'c_acc: {c_accuracy:.4f}, y_acc: {y_accuracy:.4f}')

                    # model embeddings
                    c_train_pred, y_train_pred, c_train_pred_full, y_train_pred_full, c_train_logits, y_train_logits = model_loaded.forward(x_train)
                    c_test_pred, y_test_pred, c_test_pred_full, y_test_pred_full, c_test_logits, y_test_logits = model_loaded.forward(x_test)

                    results[f'{split}'] = {
                        'dataset_name': dataset_name,
                        'model_name': model_name,
                        'emb_size': emb_size,
                        'x_train': x_train,
                        'c_train': c_train,
                        'y_train': y_train,
                        'x_test': x_test,
                        'c_test': c_test,
                        'y_test': y_test,
                        f'c_train_pred': c_train_pred,
                        f'c_test_pred': c_test_pred,
                        f'y_train_pred': y_train_pred,
                        f'y_test_pred': y_test_pred,
                        f'c_train_pred_full': c_train_pred_full,
                        f'c_test_pred_full': c_test_pred_full,
                        f'y_train_pred_full': y_train_pred_full,
                        f'y_test_pred_full': y_test_pred_full,
                        f'c_train_logits': c_train_logits,
                        f'y_train_logits': y_train_logits,
                        f'c_test_logits': c_test_logits,
                        f'y_test_logits': y_test_logits,
                        f'train_loss': model.train_loss,
                        f'val_loss': model.val_loss,
                        f'train_accuracy': model.train_accuracy,
                        f'val_accuracy': model.val_accuracy,
                        f'trainable_params': sum(p.numel() for p in model.parameters()),
                    }

                    # save results
                    joblib.dump(results, os.path.join(result_dir, f'results_{model_name}_{emb_size}.joblib'))

    return


if __name__ == '__main__':
    main()
