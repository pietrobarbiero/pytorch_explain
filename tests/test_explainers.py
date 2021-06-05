import os
import unittest

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer, seed_everything

from torch_explain.models.explainer import Explainer


class TestTemplateObject(unittest.TestCase):
    def test_mu_classifier(self):
        for i in range(1):
            seed_everything(i)

            # XOR problem
            x0 = torch.zeros((4, 100))
            x = torch.tensor([
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
            ], dtype=torch.float)
            x = torch.cat([x, x0], dim=1)
            y = torch.tensor([0, 1, 1, 0], dtype=torch.long)
            y1h = one_hot(y)
            data = TensorDataset(x, y1h)
            train_loader = DataLoader(data, batch_size=10)
            val_loader = DataLoader(data, batch_size=10)
            test_loader = DataLoader(data, batch_size=10)

            # make dirs to save results
            base_dir = f'../experiments/results/test/explainer'
            os.makedirs(base_dir, exist_ok=True)

            # train
            checkpoint_callback = ModelCheckpoint(dirpath=base_dir, monitor='val_loss', save_top_k=1)
            trainer = Trainer(max_epochs=100, gpus=1, auto_lr_find=True, deterministic=True,
                              check_val_every_n_epoch=1, default_root_dir=base_dir,
                              weights_save_path=base_dir, callbacks=[checkpoint_callback])
            model = Explainer(n_concepts=x.shape[1], n_classes=2, l1=0.001, temperature=0.6, lr=0.01,
                              explainer_hidden=[10, 10], conceptizator='identity_bool')
            trainer.fit(model, train_loader, val_loader)

            # test
            model.freeze()
            trainer.test(model, test_dataloaders=test_loader)

            # explain
            results_avg, results_details = model.explain_class(train_loader, val_loader, test_loader)
            print(results_avg)
            print(results_details)
            assert results_avg == {'explanation_accuracy': 1.0, 'explanation_fidelity': 1.0, 'explanation_complexity': 4.0}
            assert results_details == [
                {'target_class': 0,
                 'explanation': '(feature0000000000 & feature0000000001) | (~feature0000000000 & ~feature0000000001)',
                 'explanation_accuracy': 1.0,
                 'explanation_fidelity': 1.0,
                 'explanation_complexity': 4},
                {'target_class': 1,
                 'explanation': '(feature0000000000 & ~feature0000000001) | (feature0000000001 & ~feature0000000000)',
                 'explanation_accuracy': 1.0,
                 'explanation_fidelity': 1.0,
                 'explanation_complexity': 4}
            ]


if __name__ == '__main__':
    unittest.main()
