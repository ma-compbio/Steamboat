# %%
import os

if os.getcwd() != os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))

import scanpy as sc
import squidpy as sq
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numba
import numpy as np
import torch

import sys
sys.path.append("../..")
import steamboat as sf
import steamboat.tools

import matplotlib
plt.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['font.family'] = 'arial'

os.makedirs("output", exist_ok=True)

for filter_multiplier in [0, 1, 2, 5]:
    if os.path.exists(f"output/tumors_moreqc{filter_multiplier}_test.h5ad"):
        print(f"output/tumors_moreqc{filter_multiplier}_test.h5ad already exists. Skipping.")
        continue
    data = pd.read_csv("../data/Ex4_perturb_fish/tumors_expr.csv", index_col=0)
    obs = pd.read_csv("../data/Ex4_perturb_fish/tumors_obs.csv", index_col=0)
    adata = sc.AnnData(data, obs=obs)

    spatial = pd.read_csv("../data/Ex4_perturb_fish/coordinates.csv", header=None)
    adata.obsm['spatial'] = spatial.to_numpy()

    # %%
    sc.pp.filter_genes(adata, min_counts=adata.shape[0] * 0.2)
    sc.pp.filter_cells(adata, min_counts=40)
    sc.pp.filter_cells(adata, max_counts=800)

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)

    # %%
    adata.obs['celltype2'] = [i if i == 'perturbed_tumor' else 'other' for i in adata.obs['celltype']]

    def gating(adata, genes):
        res = None
        for gene in genes:
            temp = ((adata[:, gene].X).flatten() > 0.5)
            if res is None:
                res = temp
            else:
                res = res | temp
        return res

    adata.obs['celltype2'] = adata.obs['celltype2'].astype(str)
    adata.obs.loc[gating(adata, ['MKI67', 'CAV1', 'CTNNB1']), 'celltype2'] = 'cancer'
    adata.obs.loc[gating(adata, ['CD2', 'PTPRC', 'TRAC']), 'celltype2'] = 'T cells'

    adata.obs.loc[adata.obs['celltype'] == 'tumor', 'celltype2'] = 'cancer'

    adata.obs['qc_keep'] = adata.obs['n_perturb'] < 2

    if filter_multiplier > 0:
        for perturbation in tqdm(['CHUK', 'IRAK1', 'TRAM1', 'LBP', 'IRAK4', 'PELI1', 'TAB2', 'MAP2K2', 'MAP2K6', 'IRF7', 'MYD88']):
            n = (adata.obs['perturbation'] == perturbation).sum() * filter_multiplier
            avg_expr = adata.X[adata.obs['perturbation'] == perturbation].mean(axis=0)
            corr = pd.DataFrame(adata.X.T).corrwith(pd.Series(avg_expr))
            
            adata.obs['corr'] = corr.tolist()
            adata.obs['pert'] = adata.obs['perturbation'] == perturbation
        
            cutoff = np.partition(adata.obs['corr'][~adata.obs['pert']], -n)[-n]
            adata.obs.loc[(adata.obs['corr'] >= cutoff) & (~adata.obs['pert']), 'qc_keep'] = False
            # print(n, sum((adata.obs['corr'] >= cutoff) & (~adata.obs['pert'])))

    adata.write_h5ad(f"output/tumors_moreqc{filter_multiplier}_test.h5ad")

# %%
test_vars = ['TRAM1', 'PELI1', 'MAP2K2', 'IRF7', 'CHUK', 'IRAK1', 'LBP', 'IRAK4', 'TAB2', 'MAP2K6', 'MYD88']

# %%
for multiplier in [0, 1, 2, 5]:
    os.makedirs(f"output/robustness-{multiplier}", exist_ok=True)
    adata = sc.read_h5ad(f"output/tumors_moreqc{multiplier}_test.h5ad")
    adata

    # The stored data is already transformed. This step is to find the spatial neighbors.
    adatas = sf.prep_adatas([adata], norm=False, log1p=False, n_neighs=8)

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
                opt=torch.optim.Adam, opt_args=dict(lr=0.1), stop_eps=5e-5, report_per=1000, stop_tol=250, device=device)
        
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
        
        cmp_df_merge.to_csv(f"output/robustness-{multiplier}/{test_var}.csv")
        
        rho = cmp_df_merge[['logfoldchanges_gt', 'logfoldchanges_sf']].corr(method='spearman').iloc[0, 1]

        # Uncomment the following lines to save scatter plots of ground truth vs predicted log fold changes for each perturbation.
        # fig, ax = plt.subplots(figsize=(1.5, 1.5))
        # cmp_df_merge.plot(x='logfoldchanges_gt', y='logfoldchanges_sf', kind='scatter', s=1, ax=ax)
        # ax.set_title(f"{test_var}: {rho: .2f}")
        # ax.set_xlabel('Ground-truth change')
        # ax.set_ylabel('Predicted change')
        # fig.savefig(f"output/robustness-{multiplier}/{test_var}.pdf")
        
        print(f"{test_var}: {rho: .2f}")
        res[test_var] = rho

    pd.Series(res).sort_values(ascending=False).plot(kind='bar', figsize=(2, 1))
    plt.ylabel('correlation with\nground truth')
    plt.axhline(0.3, ls='--', c='r')

    pd.Series(res).to_csv(f"output/robustness-{multiplier}/steamboat_summary.csv")

################################################################################
# Robustness of Steamboat to possible dropouts in perturbation labels          #
################################################################################

t0 = pd.read_csv(f"output/robustness-0/steamboat_summary.csv", index_col=0)
t1 = pd.read_csv(f"output/robustness-1/steamboat_summary.csv", index_col=0)
t2 = pd.read_csv(f"output/robustness-2/steamboat_summary.csv", index_col=0)
t5 = pd.read_csv(f"output/robustness-5/steamboat_summary.csv", index_col=0)

t0.columns = ['0x']
t1.columns = ['1x']
t2.columns = ['2x']
t5.columns = ['5x']

df = pd.merge(t0, t1, left_index=True, right_index=True)
df = pd.merge(df, t2, left_index=True, right_index=True)
df = pd.merge(df, t5, left_index=True, right_index=True)

fig, ax = plt.subplots(figsize=(3.5, 2))
df.sort_values(by='0x', ascending=False).plot(kind='bar', width=0.75, color=['#B87DB7ff', '#B87DB7df', '#B87DB7bf', '#B87DB79f'], ax=ax)
plt.ylabel('Correlation with\nground truth')
ax = plt.gca()
for pos in ['right', 'top']:
    ax.spines[pos].set_visible(False)

plt.axhline(0.7, ls=(0, (5, 3)), c='grey', zorder=-1, lw=.5)
plt.axhline(0.3, ls=(0, (5, 3)), c='grey', zorder=-1, lw=.5)
plt.axhline(0.0, ls=(0, (5, 3)), c='grey', zorder=-1, lw=.5)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

df.to_csv(f"output/robustness_summary.csv")
fig.savefig(f"output/robustness_summary.png", dpi=300, bbox_inches='tight', transparent=True)

########################################################################
# UMAP of perturbations (perturbations do not cluster together)        #
########################################################################

adata = sc.read_h5ad("../../../data/perturbfish/cleaned/tumors_moreqc0_test.h5ad")

test_vars = ['TRAM1', 'PELI1', 'MAP2K2', 'IRF7', 'CHUK', 'IRAK1', 'LBP', 'IRAK4', 'TAB2', 'MAP2K6', 'MYD88', 'Control']
perturb_adata = adata[adata.obs['perturbation'].isin(test_vars)]
sc.pp.neighbors(perturb_adata, n_neighbors=15, use_rep='X_pca')
sc.tl.umap(perturb_adata)
palette = ['#1f77b4', '#e0e0e0', '#ff7f0e', '#279e68', '#d62728', '#aa40fc',
                  '#8c564b', '#e377c2', '#b5bd61', '#17becf', '#aec7e8',
                  '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
                  '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31']
fig = sc.pl.umap(perturb_adata, color=['perturbation'], save='_perturbation.png', palette=palette, return_fig=True, show=False)
fig.savefig(f"output/perturbation_umap.png", dpi=300, bbox_inches='tight', transparent=True)
