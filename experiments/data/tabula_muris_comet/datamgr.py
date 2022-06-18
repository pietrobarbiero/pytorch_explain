# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
# import data.additional_transforms as add_transforms
from abc import abstractmethod

from experiments.data.tabula_muris_comet.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 


class SimpleDataManager(DataManager):
    def __init__(self, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size

    def get_data_loader(self, root='./filelists/tabula_muris', mode='train'): #parameters that would change on train/val set
        dataset = SimpleDataset(root=root, mode=mode, min_samples=self.batch_size)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 4, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, n_way, n_support, n_query, n_eposide =100):        
        super(SetDataManager, self).__init__()
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

    def get_data_loader(self, root='./filelists/tabula_muris', mode='train'): #parameters that would change on train/val set
        dataset = SetDataset(root=root, mode=mode, min_samples=self.batch_size)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 4, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


