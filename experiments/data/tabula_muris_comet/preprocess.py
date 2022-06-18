'''
Created on Jul 26, 2019

@author: maria
'''

from anndata import read_h5ad
import scanpy as sc
import pandas as pd
from collections import Counter
import numpy as np

class MacaData():
    
    def __init__(self, annotation_type='cell_ontology_class_reannotated', src_file = 'dataset/cell_data/tabula-muris-senis-facs-official-annotations.h5ad', filter_genes=True):

        """
        annotation type: cell_ontology_class, cell_ontology id or free_annotation
        """
        self.adata = read_h5ad(src_file)
        self.adata.obs[annotation_type] = self.adata.obs[annotation_type].astype(str)
        self.adata = self.adata[self.adata.obs[annotation_type]!='nan',:]
        self.adata = self.adata[self.adata.obs[annotation_type]!='NA',:]
        
        #print(Counter(self.adata.obs.loc[self.adata.obs['age']=='18m', 'free_annotation']))
        
        self.cells2names = self.cellannotation2ID(annotation_type)
        
        if filter_genes:
            sc.pp.filter_genes(self.adata, min_cells=5)
        
        self.adata = self.preprocess_data(self.adata)
        
    
    def preprocess_data(self, adata):
        sc.pp.filter_cells(adata, min_counts=5000)
        sc.pp.filter_cells(adata, min_genes=500)
        
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4) #simple lib size normalization?
        adata.raw = adata
        adata = sc.pp.filter_genes_dispersion(adata, subset = False, min_disp=.5, max_disp=None, 
                                  min_mean=.0125, max_mean=10, n_bins=20, n_top_genes=None, 
                                  log=True, copy=True)
        adata = adata[:,adata.var.highly_variable]
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0
        #sc.tl.pca(self.adata)
        
        return adata  

    def get_tissue_data(self, tissue, age=None):
        """Select data for given tissue.
        filtered: if annotated return only cells with annotations, if unannotated return only cells without labels, else all
        age: '3m','18m', '24m', if None all ages are included
        """
        
        tiss = self.adata[self.adata.obs['tissue'] == tissue,:]
        
        if age:
            return tiss[tiss.obs['age']==age]
        
        return tiss
    
    
    def cellannotation2ID(self, annotation_type):
        """Adds ground truth clusters data."""
        annotations = list(self.adata.obs[annotation_type])
        annotations_set = sorted(set(annotations))
        
        mapping = {a:idx for idx,a in enumerate(annotations_set)}
        
        truth_labels = [mapping[a] for a in annotations]
        self.adata.obs['label'] = pd.Categorical(values=truth_labels)
        #18m-unannotated
        # 
        return mapping
    
if __name__ == '__main__':
    md = MacaData(src_file='../data/tabula_muris/tabula-muris-senis-facs-official-annotations.h5ad')
    tiss = md.get_tissue_data('Kidney')
    import pdb; pdb.set_trace()
    
