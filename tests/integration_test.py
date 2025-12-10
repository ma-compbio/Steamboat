import sys
sys.path.append('C:/Files/projects/Steamboat/')
from pathlib import Path

# %%
import os
import sys
import pickle as pkl
import matplotlib.pyplot as plt

## Add path to the directory containing steamboat.
sys.path.append("../") 

import torch
import pandas as pd
import numpy as np
import scipy as sp
import scanpy as sc
import squidpy as sq
import steamboat as sf # Steamboat Factorization -> sf
import steamboat.tools

def test_simulation_data():
    # %%
    device = "cpu"

    # %%
    try:
        test_dir = Path(__file__).parent
        adata = sc.read_h5ad(test_dir / "data/simulation.h5ad")
        adata.obs['global'] = 0
        adata
    except Exception as e:
        print("Error reading h5ad (issue in SCANPY?)")
        raise e

    # %%
    try:
        adatas = [adata] # You can include multiple datasets here.
        adatas = sf.prep_adatas(adatas)
        dataset = sf.make_dataset(adatas, regional_obs=[])
    except Exception as e:
        print("Error preprocessing data (issue in steamboat.dataset?):", e)
        raise e

    # %%
    # sq.pl.spatial_scatter(adatas[0], color=adatas[0].var_names, shape=None, figsize=(2, 1), size=1., legend_fontsize=9, cmap='Reds', ncols=4)
    # sq.pl.spatial_scatter(adatas[0], color='celltype', shape=None, figsize=(3, 2), size=1., legend_fontsize=9, cmap='Reds', ncols=4)
    # sq.pl.spatial_scatter(adatas[0], color='R', shape=None, figsize=(3, 2), size=1., legend_fontsize=9, cmap='Reds', ncols=4)

    # %%
    try:
        sf.set_random_seed(2)
        model = sf.Steamboat(adata.var_names.tolist(), n_heads=5, n_scales=2)
        model = model.to(device)
    except Exception as e:
        print("Error creating model (issue in steamboat.model?):", e)
        raise e

    try:
        model.fit(dataset, entry_masking_rate=0.2, feature_masking_rate=0.2,
                max_epoch=1000, 
                loss_fun=torch.nn.MSELoss(reduction='sum'),
                opt=torch.optim.Adam, opt_args=dict(lr=0.01), stop_eps=1e-3, report_per=50, stop_tol=200, device=device)
    except Exception as e:
        print("Error fitting model (issue in steamboat.model.fit?):", e)
        raise e

    try:
        sf.tools.calc_obs(adatas, dataset, model, device=device, get_recon=False)
        sf.tools.neighbors(adata, 'attn')
        sf.tools.leiden(adata, resolution=0.1)
        sf.tools.segment(adata, resolution=0.33)
    except Exception as e:
        print("Error calculating obs (issue in steamboat.tools?):", e)
        raise e
