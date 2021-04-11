import glob
import os
import unittest

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pytorch_lightning import Trainer, seed_everything

from torch_explain.models.blackbox import MNIST_X_to_C, BlackBoxSimple, BlackBoxResNet18, MnistResNet


class TestTemplateObject(unittest.TestCase):
    def test_black_box_simple(self):
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
        dataset = MNIST_X_to_C('../experiments/data', train=True, download=True, transform=mnist_transforms)
        train_data, val_data, test_data = random_split(dataset, [50000, 5000, 5000])
        train_loader = DataLoader(train_data, batch_size=180)
        val_loader = DataLoader(val_data, batch_size=180)
        test_loader = DataLoader(test_data, batch_size=180)

        # model
        base_dir = f'../experiments/results/MNIST/blackbox'
        os.makedirs(base_dir, exist_ok=True)

        # training
        checkpoint_callback = ModelCheckpoint(dirpath=base_dir, monitor='val_loss', save_top_k=1)
        trainer = Trainer(max_epochs=5, gpus=1, auto_lr_find=True, deterministic=False,
                          check_val_every_n_epoch=1, default_root_dir=base_dir,
                          weights_save_path=base_dir, profiler="simple",
                          callbacks=[EarlyStopping(monitor='val_loss'), checkpoint_callback])

        path = glob.glob(f'{base_dir}/*.ckpt')
        if path:
            model = BlackBoxSimple.load_from_checkpoint(path[0])
        else:
            model = BlackBoxSimple(n_concepts=10)
            trainer.fit(model, val_loader, val_loader)

        model.freeze()
        trainer.test(model, test_dataloaders=test_loader)

        data_dir = '../experiments/data/MNIST_X_to_C'
        # dataset = model.transform(train_loader, base_dir=data_dir, extension='training')
        dataset = model.transform(val_loader, base_dir=data_dir, extension='validation')
        dataset = model.transform(test_loader, base_dir=data_dir, extension='test')


    def test_resnet18(self):
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
        dataset = MNIST_X_to_C('../experiments/data', train=True, download=True, transform=mnist_transforms)
        train_data, val_data, test_data = random_split(dataset, [50000, 5000, 5000])
        train_loader = DataLoader(train_data, batch_size=180)
        val_loader = DataLoader(val_data, batch_size=180)
        test_loader = DataLoader(test_data, batch_size=180)

        # model
        base_dir = f'../experiments/results/MNIST/resnet18'
        os.makedirs(base_dir, exist_ok=True)

        # training
        checkpoint_callback = ModelCheckpoint(dirpath=base_dir, monitor='val_loss', save_top_k=1)
        trainer = Trainer(max_epochs=5, gpus=1, auto_lr_find=True, deterministic=False,
                          check_val_every_n_epoch=1, default_root_dir=base_dir,
                          weights_save_path=base_dir, profiler="simple",
                          callbacks=[EarlyStopping(monitor='val_loss'), checkpoint_callback])

        path = glob.glob(f'{base_dir}/*.ckpt')
        if path:
            model = BlackBoxResNet18.load_from_checkpoint(path[0])
        else:
            model = BlackBoxResNet18(n_concepts=10, model=MnistResNet())
            trainer.fit(model, val_loader, val_loader)

        model.freeze()
        trainer.test(model, test_dataloaders=test_loader)


if __name__ == '__main__':
    unittest.main()
