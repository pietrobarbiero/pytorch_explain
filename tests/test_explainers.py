import glob
import os
import unittest

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pytorch_lightning import Trainer, seed_everything

from torch_explain.models.explainer import MNIST_C_to_Y, MuExplainer


class TestTemplateObject(unittest.TestCase):
    def test_mu_classifier(self):
        seed_everything(42)
        # data
        size = 224
        resize = int(size * 0.9)
        mnist_transforms = transforms.Compose([
            transforms.Resize(size=resize),
            transforms.CenterCrop(size=size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        dataset = MNIST_C_to_Y('../experiments/data', train=True, download=True, transform=mnist_transforms)
        train_data, val_data, test_data = random_split(dataset, [50000, 5000, 5000])
        train_loader = DataLoader(train_data, batch_size=180)
        val_loader = DataLoader(val_data, batch_size=180)
        test_loader = DataLoader(test_data, batch_size=180)

        # model
        base_dir = f'../experiments/results/MNIST/explainer'
        os.makedirs(base_dir, exist_ok=True)

        # training
        checkpoint_callback = ModelCheckpoint(dirpath=base_dir, monitor='val_loss', save_top_k=1)
        trainer = Trainer(min_epochs=20, max_epochs=20, gpus=1, auto_lr_find=True, deterministic=False,
                          check_val_every_n_epoch=1, default_root_dir=base_dir,
                          weights_save_path=base_dir, profiler="simple",
                          callbacks=[checkpoint_callback])

        path = glob.glob(f'{base_dir}/*.ckpt')
        model = MuExplainer(n_concepts=10, n_classes=2, concept_activation='identity', l1=0.01, prune_epoch=10,
                            fan_in=5)
        trainer.fit(model, val_loader, val_loader)

        model.freeze()
        trainer.test(model, test_dataloaders=test_loader)

        x, y = next(iter(test_loader))

        (x.cpu() > 0.5).to(torch.float)
        y2 = torch.zeros((y.shape[0], 2))
        y2[:, 0] = y.cpu() % 2 == 0
        y2[:, 1] = y.cpu() % 2 == 1
        class_explanation, _ = model.explain_class(x, y, target_class=1, topk_explanations=10)

        print(class_explanation)


if __name__ == '__main__':
    unittest.main()
