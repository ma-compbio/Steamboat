# %% [markdown]
# # Evaluating Spatial Perturbation Prediction on Perturb-FISH Data
# 
# - Dataset: [](https://www.nature.com/articles/s41576-025-00857-8)
# - Evaluate in silico perturbation prediction, by perturbing control cancer cells and observing the changes in nearby T cells.
# - Training is performed in a leave-one-out fasion by removing all cancer cells with the KO to be tested.

# %%
import os

import scanpy as sc
import squidpy as sq
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numba
import numpy as np
import torch

import sys
sys.path.append("../../")
import steamboat as sf
import steamboat.tools

import matplotlib
plt.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['font.family'] = 'arial'

import os
os.makedirs("output", exist_ok=True)

# %%
adata = sc.read_h5ad("../data/Ex4_perturb_fish/tumors_qc_test.h5ad")
adata

# The stored data is already transformed. This step is to find the spatial neighbors.
adatas = sf.prep_adatas([adata], norm=False, log1p=False, n_neighs=8)

# %%
test_vars = ['CHUK', 'IRAK1', 'TRAM1', 'LBP', 'IRAK4', 'PELI1', 'TAB2', 'MAP2K2', 'MAP2K6', 'IRF7', 'MYD88']

# %%
res = {}

for test_var in test_vars:
    # find all cells near ko or control tumor cells
    adata.obs['near_ko'] = (adata.obsp['spatial_connectivities'] @ (adata.obs['perturbation'] == test_var).to_numpy()[:, None]) > 0
    adata.obs['near_control'] = (adata.obsp['spatial_connectivities'] @ (adata.obs['perturbation'] == 'Control').to_numpy()[:, None]) > 0
    
    # find T cells near ko or control tumor cells
    adata.obs['t_near_ko'] = (adata.obs['near_ko'] & (adata.obs['celltype2'] == 'T cells'))
    adata.obs['t_near_control'] = (adata.obs['near_control'] & (adata.obs['celltype2'] == 'T cells'))
    
    # Find ground truth DEGs
    sub_adata = adata[adata.obs['t_near_ko'] | adata.obs['t_near_control']].copy()
    sub_adata.obs['test_group'] = sub_adata.obs['t_near_ko'].apply(lambda x: 'test' if x else 'control')
    sc.tl.rank_genes_groups(sub_adata, 'test_group', method='wilcoxon', key_added = "wilcoxon")
    # sc.pl.rank_genes_groups(sub_adata, n_genes=25, sharey=False, key="wilcoxon")
    cmp_df_gt = sc.get.rank_genes_groups_df(sub_adata, group="test", key='wilcoxon')
    # cmp_df_gt
    
    # keep a copy of ko cells and remove those cells (prevent leakage of data in training)
    substitute = adata.X[adata.obs['perturbation'] == test_var]
    train_adata = adata[adata.obs['perturbation'] != test_var, :].copy()
    train_adatas = sf.prep_adatas([train_adata], norm=False, log1p=False, n_neighs=8)
    train_dataset = sf.make_dataset(train_adatas, sparse_graph=True, regional_obs=[])
    
    # Train the model
    device = 'cuda'
    sf.set_random_seed(0)
    model = sf.Steamboat(adata.var_names.tolist(), n_heads=50, n_scales=2)
    model = model.to(device)
    
    model.fit(train_dataset.to(device), entry_masking_rate=0.1, feature_masking_rate=0.1,
              max_epoch=10000, 
              loss_fun=torch.nn.MSELoss(reduction='sum'),
              opt=torch.optim.Adam, opt_args=dict(lr=0.1), stop_eps=1e-4, report_per=500, stop_tol=200, device=device)
    
    # Substitute control cells with random ko cells
    test_adata = train_adata.copy()
    test_adata.X[test_adata.obs['perturbation'] == 'Control'] = substitute[np.random.choice(substitute.shape[0], 
                                                                                           (adata.obs['perturbation'] == 'Control').sum(), 
                                                                                           replace=True)]
    
    # Reconstruct the cells
    test_dataset = sf.make_dataset([test_adata], sparse_graph=True, regional_obs=[])
    sf.tools.calc_obs([test_adata], test_dataset, model, get_recon=True)
    sf.tools.calc_obs([train_adata], train_dataset, model, get_recon=True)
    
    subset = test_adata.obs['t_near_control'] & (~test_adata.obs['t_near_ko'])
    cmp_adata = sc.AnnData(np.vstack([test_adata[subset].obsm['X_recon'], 
                                      train_adata[subset].obsm['X_recon']]), 
                           var=test_adata.var.copy())
    
    cmp_adata.obs['grp'] = ['KO'] * subset.sum() + ['WT'] * subset.sum()
    sc.tl.rank_genes_groups(cmp_adata, groupby='grp', method='wilcoxon')
    cmp_df_sf = sc.get.rank_genes_groups_df(cmp_adata, group="KO")
    # cmp_df_sf
    
    cmp_df_merge = pd.merge(cmp_df_gt, cmp_df_sf, left_on='names', right_on='names', suffixes=['_gt', '_sf'])
    # cmp_df_merge
    
    cmp_df_merge.to_csv(f"output/{test_var}.csv")
    
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    rho = cmp_df_merge[['logfoldchanges_gt', 'logfoldchanges_sf']].corr(method='spearman').iloc[0, 1]
    cmp_df_merge.plot(x='logfoldchanges_gt', y='logfoldchanges_sf', kind='scatter', s=1, ax=ax)
    ax.set_title(f"{test_var}: {rho: .2f}")
    ax.set_xlabel('Ground-truth change')
    ax.set_ylabel('Predicted change')
    fig.savefig(f"output/{test_var}.png", bbox_inches='tight', transparent=True, dpi=300)
    
    print(f"{test_var}: {rho: .2f}")
    res[test_var] = rho

# %%
pd.Series(res).sort_values(ascending=False).plot(kind='bar', figsize=(2, 1))
plt.ylabel('correlation with\nground truth')
plt.axhline(0.3, ls='--', c='r')

# %%
pd.Series(res).to_csv("output/steamboat_summary.csv")
