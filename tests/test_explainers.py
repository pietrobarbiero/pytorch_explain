import os
import unittest

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything

from torch_explain.logic import test_explanation, complexity
from torch_explain.models.explainer import MuExplainer


class TestTemplateObject(unittest.TestCase):
    def test_mu_classifier(self):
        seed_everything(42)
        # data
        # train_data = torch.load('../experiments/data/MNIST_X_to_C/c2y_training.pt')
        val_data = torch.load('../experiments/data/MNIST_X_to_C/c2y_validation.pt')
        test_data = torch.load('../experiments/data/MNIST_X_to_C/c2y_test.pt')
        # val_data.tensors = ((val_data.tensors[1]>0.5).to(torch.float),
        #                     (val_data.tensors[1].argmax(dim=1) % 2 == 1).to(torch.long))
        # test_data.tensors = ((test_data.tensors[1]>0.5).to(torch.float),
        #                      (test_data.tensors[1].argmax(dim=1) % 2 == 1).to(torch.long))
        val_data.tensors = (torch.cat((val_data.tensors[1], torch.zeros((val_data.tensors[1].shape[0], 2))), 1),
                            (one_hot((val_data.tensors[1].argmax(dim=1) % 2 == 1).to(torch.long)).to(torch.float)))
        test_data.tensors = (torch.cat((test_data.tensors[1], torch.zeros((test_data.tensors[1].shape[0], 2))), 1),
                             (one_hot((test_data.tensors[1].argmax(dim=1) % 2 == 1).to(torch.long)).to(torch.float)))
        # train_loader = DataLoader(train_data, batch_size=180)
        val_loader = DataLoader(val_data, batch_size=180)
        test_loader = DataLoader(test_data, batch_size=180)

        # model
        base_dir = f'../experiments/results/MNIST/explainer'
        os.makedirs(base_dir, exist_ok=True)

        # training
        checkpoint_callback = ModelCheckpoint(dirpath=base_dir, monitor='val_loss', save_top_k=1)
        trainer = Trainer(max_epochs=20, gpus=1, auto_lr_find=True, deterministic=True,
                          check_val_every_n_epoch=1, default_root_dir=base_dir,
                          weights_save_path=base_dir, profiler="simple",
                          callbacks=[checkpoint_callback])

        model = MuExplainer(n_concepts=12, n_classes=2, l1=0.00001, lr=0.01, explainer_hidden=[10, 10])
        trainer.fit(model, val_loader, val_loader)

        model.freeze()
        trainer.test(model, test_dataloaders=test_loader)
        results, results_full = model.explain_class(val_loader, test_loader, topk_explanations=20)
        print(results)
        print(results_full[0]['explanation'])
        print(model.model[0].gamma)
        assert results_full[0]['explanation'] == '(feature0000000000 & ~feature0000000001 & ~feature0000000002 & ~feature0000000003 & ~feature0000000004 & ~feature0000000005 & ~feature0000000006 & ~feature0000000007 & ~feature0000000008 & ~feature0000000009) | (feature0000000002 & ~feature0000000000 & ~feature0000000001 & ~feature0000000003 & ~feature0000000004 & ~feature0000000005 & ~feature0000000006 & ~feature0000000007 & ~feature0000000008 & ~feature0000000009) | (feature0000000004 & ~feature0000000000 & ~feature0000000001 & ~feature0000000002 & ~feature0000000003 & ~feature0000000005 & ~feature0000000006 & ~feature0000000007 & ~feature0000000008 & ~feature0000000009) | (feature0000000006 & ~feature0000000000 & ~feature0000000001 & ~feature0000000002 & ~feature0000000003 & ~feature0000000004 & ~feature0000000005 & ~feature0000000007 & ~feature0000000008 & ~feature0000000009) | (feature0000000008 & ~feature0000000000 & ~feature0000000001 & ~feature0000000002 & ~feature0000000003 & ~feature0000000004 & ~feature0000000005 & ~feature0000000006 & ~feature0000000007 & ~feature0000000009)'


if __name__ == '__main__':
    unittest.main()
