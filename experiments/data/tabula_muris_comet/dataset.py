# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import os
import torch.utils.data as data
from .preprocess import MacaData
from .map_GO import get_go2gene
identity = lambda x:x

def create_go_mask(adata, go2gene):
    genes = adata.var_names
    gene2index = {g: i for i, g in enumerate(genes)}
    GO_IDs = sorted(go2gene.keys())
    go_mask = []
    for go in GO_IDs:
        go_genes = go2gene[go]
        go_mask.append([gene2index[gene] for gene in go_genes])
    return go_mask

def load_tabular_muris(root='./filelists/tabula_muris', mode='train', min_samples=20):
    train_tissues = ['BAT', 'Bladder', 'Brain_Myeloid', 'Brain_Non-Myeloid',
       'Diaphragm', 'GAT', 'Heart', 'Kidney', 'Limb_Muscle', 'Liver', 'MAT', 'Mammary_Gland',
       'SCAT', 'Spleen', 'Trachea']
    val_tissues = ["Skin", "Lung", "Thymus", "Aorta"]
    test_tissues = ["Large_Intestine", "Marrow", "Pancreas", "Tongue"]
    split = {'train': train_tissues,
                'val': val_tissues,
                'test': test_tissues}
    adata = MacaData(src_file=os.path.join(root, "tabula-muris-comet.h5ad")).adata
    tissues = split[mode]
    # subset data based on target tissues
    adata = adata[adata.obs['tissue'].isin(tissues)]
    
    filtered_index = adata.obs.groupby(["label"]) \
                .filter(lambda group: len(group) >= min_samples) \
                .reset_index()['index']
    adata = adata[filtered_index]

    # convert gene to torch tensor x
    samples = adata.to_df().to_numpy(dtype=np.float32)
    # convert label to torch tensor y
    targets = adata.obs['label'].cat.codes.to_numpy(dtype=np.int32)
    go2gene = get_go2gene(adata=adata, GO_min_genes=32, GO_max_genes=None, GO_min_level=6,
                          GO_max_level=1, data_dir=root)
    go_mask = create_go_mask(adata, go2gene)
    return samples, targets, go_mask


class SimpleDataset:
    def __init__(self, root='./filelists/tabula_muris', mode='train', min_samples=20):
        samples_all, targets_all, go_masks_all = load_tabular_muris(root=root, mode=mode, min_samples=min_samples)
        self.samples = samples_all
        self.targets = targets_all
        self.go_mask = go_masks_all

    def __getitem__(self,i):
        return self.samples[i], self.targets[i]

    def __len__(self):
        return self.samples.shape[0]

    def get_dim(self):
        return self.samples.shape[1]


class SetDataset:
    def __init__(self, root='./filelists/tabula_muris', mode='train', min_samples=20):
        samples_all, targets_all, go_masks = load_tabular_muris(root=root, mode=mode, min_samples=min_samples)
        self.cl_list = np.unique(targets_all)
        self.go_mask = go_masks
        self.x_dim = samples_all.shape[1]
        self.sub_dataloader =[]
        sub_data_loader_params = dict(batch_size = min_samples,
            shuffle = True,
            num_workers = 0, #use main thread only or may receive multiple batches
            pin_memory = False)
        for cl in self.cl_list:
            samples = samples_all[targets_all == cl, ...]
            sub_dataset = SubDataset(samples, cl)
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

    def get_dim(self):
        return self.x_dim

class SubDataset:
    def __init__(self, samples, cl):
        self.samples = samples
        self.cl = cl 
    
    def __getitem__(self,i):
        return self.samples[i], self.cl

    def __len__(self):
        return self.samples.shape[0]

    def get_dim(self):
        return self.samples.shape[1]

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
