# %% [markdown]
# # Train on mouse brain data
# 
# This could take several hours. The trained model is provided, which can be used to try the interpretation part without training again.

# %%
import os

# specify the path to the root directory of the repository if not installed as a package
# if you seen ModuleNotFoundError: No module named 'steamboat',
# this is likely because you are running from a different working directory. 
# You can change the path below to the absolute root directory of the repository
import sys
sys.path.append("../..")


device = "cuda"
import importlib


# %%
import scanpy as sc
import squidpy as sq
import pandas as pd
from tqdm.notebook import tqdm
import scipy as sp
import numpy as np
import multiprocessing
import pickle as pkl
import torch
import gc
import sklearn.metrics

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'arial'

pltkw = dict(bbox_inches='tight', transparent=True)

# %%
import steamboat as sf
import torch

# %%
if not os.path.exists("../data/Ex2_mouse_brain/Zhuang-ABCA-1-labeled.h5ad"): # regenerate the labeled data
    adata = sc.read_h5ad("../data/Ex2_mouse_brain/Zhuang-ABCA-1-raw.h5ad")
    obs = pd.read_csv("../data/Ex2_mouse_brain/label.csv.gz", index_col=0)
    
    # obs = obs[obs['brain_section_label'].isin(obs['brain_section_label'].value_counts()[:10].index.tolist())]
    # adata = adata[obs.index, :]
    # adata.obs = obs
    # gc.collect()
    
    adata = adata[obs.index, :]
    gc.collect()
    adata.obs = obs
    gc.collect()
    
    adata.obsm['spatial'] = adata.obs[['x', 'y']].to_numpy()
    
    # It will write a new h5ad file with the labeled data; it's larger but faster to load again
    adata.write_h5ad("../data/Ex2_mouse_brain/Zhuang-ABCA-1-labeled.h5ad")
else:
    adata = sc.read_h5ad("../data/Ex2_mouse_brain/Zhuang-ABCA-1-labeled.h5ad")
    
adata

# %%
if False:
    n_genes = 100
    np.random.seed(0)
    chosen_gene_mask = np.zeros(adata.shape[1])
    chosen_gene_mask[:n_genes] = 1.
    np.random.shuffle(chosen_gene_mask)
    chosen_gene_mask = chosen_gene_mask > 0.
    adata = adata[:, chosen_gene_mask]
gc.collect()

# Normalizing to median total counts
sc.pp.normalize_total(adata)
# Logarithmize the data
sc.pp.log1p(adata)


# %%
adatas = []
for i in adata.obs['brain_section_label'].unique():
    adatas.append(adata[adata.obs['brain_section_label'] == i])
    adatas[-1].obs['global'] = 0

adatas = sf.prep_adatas(adatas, norm=False, log1p=False)
dataset = sf.make_dataset(adatas, sparse_graph=True, regional_obs=['global'])

# %%
sf.set_random_seed(0)
model = sf.Steamboat(adata.var_names.tolist(), n_heads=50, n_scales=3)
model = model.to(device)
# model.load_state_dict(torch.load('backup_mmbrain4/model_weights.pth', weights_only=True))

model.fit(dataset, entry_masking_rate=0.1, feature_masking_rate=0.1,
          max_epoch=10000, 
          loss_fun=torch.nn.MSELoss(reduction='sum'),
          opt=torch.optim.Adam, opt_args=dict(lr=0.1), stop_eps=1e-3, report_per=200, stop_tol=200, device=device)

# %%
torch.save(model.state_dict(), '../retrained_models/mmbrain.pth')
