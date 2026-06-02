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
sys.path.append("../../SEDR/")

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
name = 'sedr_spatial_domain'
for i in tqdm(range(124, 129)):
    try:
        random_seed = 2023
        SEDR.fix_seed(random_seed)
        adata = sc.read_h5ad(f"backup/mmbrain/adata_{i}.h5ad")
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
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
    
        n_clusters = adata.obs['parcellation_division'].nunique()
        SEDR.mclust_R(adata, n_clusters, use_rep='SEDR', key_added=name)
    
        adata.obs[[name]].to_csv(f"backup/mmbrain/{name}_{i}.csv")
        
        ari = sklearn.metrics.adjusted_rand_score(adata.obs['parcellation_division'], adata.obs[name])
        nmi = sklearn.metrics.normalized_mutual_info_score(adata.obs['parcellation_division'], adata.obs[name])
        # aris.append(ari)
        # nmis.append(nmi)
    
        print(i, adata.obs[name].nunique(), ari, nmi)
        # aris.append(ari)
        # nmis.append(nmi)

    except Exception as e:
        print(e)
        
    del adata
    gc.collect()
    
# df = pd.DataFrame({'ARI': aris, 'NMI': nmis})
# df.to_csv(f"backup/mmbrain/{name}.csv")
# df

# %%
import pandas as pd
import os.path

aris = []
nmis = []

name = 'sedr_spatial_domain'
groundtruth_name = 'groundtruth_spatial_domain'
for i in tqdm(range(129)):
    x = pd.read_csv(f"backup/mmbrain/{groundtruth_name}_{i}.csv", index_col=0)
    fname = f"backup/mmbrain/{name}_{i}.csv"
    if os.path.isfile(fname):
        y = pd.read_csv(fname, index_col=0)
        ari = sklearn.metrics.adjusted_rand_score(x['parcellation_division'], y[name])
        nmi = sklearn.metrics.normalized_mutual_info_score(x['parcellation_division'], y[name])
    else:
        print('Sample', i, 'failed.')
        ari = float('nan')
        nmi = float('nan')
    aris.append(ari)
    nmis.append(nmi)

df = pd.DataFrame({'ARI': aris, 'NMI': nmis})
df.to_csv(f"backup/mmbrain/{name}.csv")
df

# %%
print('test')


