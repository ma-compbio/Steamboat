# %%
import pandas as pd
import numpy as np
import scanpy as sc
import squidpy as sq
import torch

import sys
sys.path.append("../")
import steamboat as sf
import steamboat.tools

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['font.family'] = 'arial'
matplotlib.rcParams["legend.handletextpad"] = 0.
matplotlib.rcParams["legend.labelspacing"] = .3
pltkw = dict(bbox_inches='tight', transparent=True)

# %%
adata = sc.read_h5ad("../data/Ex5_slide_tags_human_tonsil/HumanTonsil_2000.h5ad")

# %%
adata

# %%
adata.obs['global'] = 0
adatas = sf.prep_adatas([adata], norm=False, log1p=False, n_neighs=8)
dataset = sf.make_dataset(adatas, sparse_graph=True, regional_obs=['global'])

# %%
device = 'cuda'
sf.set_random_seed(0)
model = sf.Steamboat(adata.var_names.tolist(), n_heads=60, n_scales=3)
model = model.to(device)


model.fit(dataset.to(device), entry_masking_rate=0.1, feature_masking_rate=0.1,
            max_epoch=10000, 
            loss_fun=torch.nn.MSELoss(reduction='sum'),
            opt=torch.optim.Adam, opt_args=dict(lr=0.1), stop_eps=2e-5, report_per=1000, stop_tol=250, device=device)

torch.save(model.state_dict(), 'saved_models/tonsil.pth')

# [2025-10-02 12:25:00,171::train::INFO] Epoch 1: train_loss 393.33020
# [2025-10-02 12:25:06,773::train::INFO] Epoch 1001: train_loss 206.91405
# [2025-10-02 12:25:13,339::train::INFO] Epoch 2001: train_loss 203.47221
# [2025-10-02 12:25:19,166::train::INFO] Epoch 2886: train_loss 203.11626
# [2025-10-02 12:25:19,167::train::INFO] Stopping criterion met.
