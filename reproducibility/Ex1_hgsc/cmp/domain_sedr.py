# %%
import scanpy as sc
# import squidpy as sq
# import pandas as pd
from tqdm.notebook import tqdm
import scipy as sp
import numpy as np
import pickle as pkl
import torch
import gc
import sklearn.metrics

# %%
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

import os

# %%
import sys
sys.path.append("../SEDR")

# %%
import SEDR
from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable.

# %% [markdown]
# ## SEDR

# %%
# gpu
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# %%
# aris = []
# nmis = []

random_seed = 2023
SEDR.fix_seed(random_seed)
adata = sc.read_h5ad("../Banksy_py/data/hgsc_SMI_T10_F006.h5ad")
name = f'hgsc_{adata.obs["samples"][0]}_sedr_spatial_domain'

sc.pp.scale(adata)
adata_X = PCA(n_components=50, random_state=42).fit_transform(adata.X)
adata.obsm['X_pca'] = adata_X
graph_dict = SEDR.graph_construction(adata, 8)
sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict)
using_dec = True
if using_dec:
    sedr_net.train_with_dec()
else:
    sedr_net.train_without_dec()
sedr_feat, _, _, _ = sedr_net.process()
adata.obsm['SEDR'] = sedr_feat

n_clusters = 3
SEDR.mclust_R(adata, n_clusters, use_rep='SEDR', key_added=name)

adata.obs[[name]].to_csv(f"../Steamboat/revised/saved_results/{name}.csv")
