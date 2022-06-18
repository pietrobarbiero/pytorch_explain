'''
Created on Oct 26, 2018

@author: maria
'''
import os
from sys import warnoptions
from goatools import obo_parser
from collections import defaultdict
import scanpy as sc
import numpy as np
import pandas as pd
import torch


def prepare_GO_data(adata, gene2go, GO_file, GO_min_genes=500, GO_max_genes=None, GO_min_level=3, GO_max_level=3):
    """
    Preprocesses data .
    GO terms are propagated to all parents categories so all GO terms satisfying conditions of 
    min and max genes are included.
    gene2go: mapping of gene IDs to GO terms
    count_data: anndata object containing raw count data
    GO_file: GO ontology obo file
    GO_min_genes: minimum number of genes assigned to GO required to keep GO term (default: 500)
    GO_max_genes: maximum number of genes assigned to GO required to keep GO term (default: None)
    GO_min_level: minimum level required to keep GO term (default: 3)
    npcs: number of principal components
    annotations: dictionary containing cell annotations (default: None)
    return: dictionary of GO terms with processed anndata object with calculated knn graph
            of only genes belonging to that GO term
    """
    GOdag = obo_parser.GODag(obo_file=GO_file)
    genes = set(adata.var_names)

    gene2go = {g: gene2go[g] for g in gene2go.keys() if g in genes}
    GOdag.update_association(gene2go)  # propagate through hierarchy
    go2gene = reverse_association(gene2go)
    # return go2gene
    filtered_go2gene = {}

    for GO in go2gene:
        ngenes = len(go2gene[GO])
        if check_conditions(GOdag.get(GO), ngenes, GO_min_genes,
                            GO_max_genes, GO_min_level, GO_max_level):
            filtered_go2gene[GO] = go2gene[GO]
    print("Num filtered GOs:", len(filtered_go2gene))
    return filtered_go2gene


def check_conditions(GOterm, num_genes, min_genes, max_genes, min_level, max_level):
    """Check whether GO term satisfies required conditions."""

    if min_genes != None:
        if num_genes < min_genes:
            return False
    if max_genes != None:
        if num_genes > max_genes:
            return False
    if min_level != None:
        if GOterm.level < min_level:
            return False
    if min_level != None:
        if GOterm.level > min_level:
            return False
    return True


def filter_cells(adata, min_genes=501, min_counts=50001):
    """Removing cells which do not have min_genes and min_counts as done
    in Tabula Muris preprocessing.
    min_genes: minimum number of genes required to retain a cell
    min_counts:  minimum number of counts required to retain a cell
    """
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    return adata


def remove_ERCC_genes(adata):
    """Removing ERCC genes as done in Tabula Muris preprocessing."""
    genes = adata.var_names
    # remove genes starting with ERCC
    idx = [i for i, g in enumerate(genes) if not g.startswith("ERCC")]

    genes = [g for i, g in enumerate(genes) if i in idx]
    adata = adata[:, idx]
    return adata


def anndata_to_df(adata):
    return pd.DataFrame(adata.X.toarray(), dtype=np.float32, index=adata.obs_names,
                        columns=adata.var_names).transpose()


def reverse_association(gene2go):
    """
    For given dictionary of genes mapped to set of GO
    terms, creates mapping of GO terms to gene IDs.
    gene2go: mapping of gene IDs to GO terms
    return: mapping of GO terms to gene IDs
    """
    go2gene = defaultdict(set)
    for gene, go_set in gene2go.items():
        for go in go_set:
            go2gene[go].add(gene)
    return go2gene


def map_mgi2go(filepath):
    """
    Reads from file mapping of MGI mouse gene ID to GO. Takes only genes with
    experimental and high throughput evidence codes.
    filepath: file containing mapping
    return: mapping of MGI to GO
    """
    supported_codes = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP",
                       "HTP", "HDA", "HMP", "HGI", "HEP"}
    mgi2go = defaultdict(set)
    with open(filepath) as f:
        for line in f.readlines():
            if line[0] != '!':
                line = line.split("\t")
                mgi = line[2]
                go = line[4]
                evidence_code = line[6]
                # FIXME experimental, comment out
                mgi2go[mgi].add(go)
                # if evidence_code in supported_codes:
                #     mgi2go[mgi].add(go)
    # print(len(mgi2go))
    return mgi2go


def get_go2gene(adata, GO_min_genes=500, GO_max_genes=None, GO_min_level=3, GO_max_level=3,
                                       data_dir='./filelists/tabula_muris/'):
    """
    Returns processed tabula muris data in AnnData format.
    GO_min_genes: minimum number of genes assigned to GO required to keep GO term, used only if data is separated by GO
                 categories (default: 500)
    GO_max_genes: maximum number of genes assigned to GO required to keep GO term, used only if data is separated by GO
                 categories  (default: None)
    GO_min_level: minimum level required to keep GO term, used only if data is separated by GO categories  (default: 3)
    raw_data_dir: directory contaning raw data
    """

    mgi2go = map_mgi2go(os.path.join(data_dir, "gene_association.mgi"))
    mgi2go_set = set(mgi2go.keys())
    adata_set = set(adata.var_names)
    print("_________Gene count processed_________")
    print("mgi2go_set", len(mgi2go_set))
    print("adata_set", len(adata_set))
    print("union", len(adata_set & mgi2go_set))
    # print("Not found", adata_set - mgi2go_set)
    GOobo_file = os.path.join(data_dir, "go-basic.obo")

    go2gene = prepare_GO_data(adata, mgi2go, GO_file=GOobo_file,
                                     GO_min_genes=GO_min_genes, GO_max_genes=GO_max_genes, GO_min_level=GO_min_level, GO_max_level=GO_max_level)

    return go2gene

