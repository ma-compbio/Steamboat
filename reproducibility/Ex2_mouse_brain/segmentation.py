# %%
import scanpy as sc
import squidpy as sq
import pandas as pd
from tqdm.notebook import tqdm
import scipy as sp
import numpy as np
import pickle as pkl
import torch
import gc
import sklearn.metrics

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['font.family'] = 'arial'
matplotlib.rc('pdf', fonttype=42)

# %% [markdown]
# ## Steamboat

# %%
for i in tqdm(range(129)):
    adata = sc.read_h5ad(f"tmp_adata/adata_{i}.h5ad")
    adata.obsm['emb'] = adata.obsm['attn']
    adata.obsm['emb'] /= adata.obsm['emb'].std(axis=0, keepdims=True)
    sc.pp.neighbors(adata, use_rep='emb', key_added='sf', metric='euclidean')
    
    temp = 0
    n_prop = 3
    for j in range(50):
        temp += (adata.obsp[f'local_attn_{j}'] / adata.obsp[f'local_attn_{j}'].data.std()) ** n_prop
    temp = temp.power(1/n_prop)
    temp.data /= temp.data.max()
    temp.eliminate_zeros()
    adata.obsp['sfsd2'] = temp + adata.obsp['sf_connectivities']
    sc.tl.leiden(adata, obsp='sfsd2', key_added='steamboat_spatial_domain', resolution=1.)
    adata.obs[['parcellation_division', 'steamboat_spatial_domain']].to_csv(f"output/steamboat_spatial_domain_{i}.csv")
    
    obs = adata.obs
    ari = sklearn.metrics.adjusted_rand_score(obs['parcellation_division'], obs['steamboat_spatial_domain'])
    nmi = sklearn.metrics.normalized_mutual_info_score(obs['parcellation_division'], obs['steamboat_spatial_domain'])
    
    print(i, adata.obs['steamboat_spatial_domain'].astype(int).max(), obs['parcellation_division'].nunique(), ari, nmi)
    gc.collect()

# %%
aris = []
nmis = []
for i in tqdm(range(129)):
    obs = pd.read_csv(f"output/steamboat_spatial_domain_{i}.csv", index_col=0)
    ari = sklearn.metrics.adjusted_rand_score(obs['parcellation_division'], obs['steamboat_spatial_domain'])
    nmi = sklearn.metrics.normalized_mutual_info_score(obs['parcellation_division'], obs['steamboat_spatial_domain'])
    aris.append(ari)
    nmis.append(nmi)

df = pd.DataFrame({'ARI': aris, 'NMI': nmis})
df.to_csv("output/steamboat_test_spatial_domain_test.csv")
df
