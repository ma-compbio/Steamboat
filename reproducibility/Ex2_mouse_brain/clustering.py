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

rerun = False

if rerun:
    # PCA
    from joblib import Parallel, delayed

    aris = []
    nmis = []

    def func(i):
        name = 'pca_cell_type'
        adata = sc.read_h5ad(f"tmp_adata/adata_{i}.h5ad")
        # sc.pp.scale(adata)
        sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata)

        sc.tl.leiden(adata, key_added=name, resolution=0.7)
        adata.obs[[name]].to_csv(f"output/{name}_{i}.csv")

        ari = sklearn.metrics.adjusted_rand_score(adata.obs['class'], adata.obs[name])
        nmi = sklearn.metrics.normalized_mutual_info_score(adata.obs['class'], adata.obs[name])
        
        return adata.obs['class'].nunique(), adata.obs[name].nunique(), ari, nmi

    res = Parallel(n_jobs=20)(delayed(func)(i) for i in tqdm(range(129), total=129))
    for class_nunique, nunique, ari, nmi in res:
        print(class_nunique, nunique, ari, nmi)
        aris.append(ari)
        nmis.append(nmi)
    gc.collect()

    df = pd.DataFrame({'ARI': aris, 'NMI': nmis})
    df.to_csv(f"output/pca_cell_type.csv")

    # NMF
    import sklearn.decomposition

    aris = []
    nmis = []
    name = 'nmf_cell_type'
    for i, adata in enumerate(tqdm(range(129), total=129)):
        adata = sc.read_h5ad(f"output/adata_{i}.h5ad")
        model = sklearn.decomposition.NMF(50, max_iter=500)
        adata.obsm['X_nmf'] = model.fit_transform(adata.X)
        sc.pp.neighbors(adata, use_rep='X_nmf')

        sc.tl.leiden(adata, key_added=name, resolution=1.)
        adata.obs[[name]].to_csv(f"output/{name}_{i}.csv")

        ari = sklearn.metrics.adjusted_rand_score(adata.obs['class'], adata.obs[name])
        nmi = sklearn.metrics.normalized_mutual_info_score(adata.obs['class'], adata.obs[name])

        print(i, adata.obs[name].nunique(), ari, nmi)
        aris.append(ari)
        nmis.append(nmi)

        del adata
        gc.collect()


    df = pd.DataFrame({'ARI': aris, 'NMI': nmis})
    df.to_csv(f"output/{name}.csv")


    # Steamboat
    aris = []
    nmis = []
    for i in tqdm(range(129), total=129):
        adata = sc.read_h5ad(f"output/adata_{i}.h5ad")
        adata.obsm['emb'] = adata.obsm['attn']
        adata.obsm['emb'] /= adata.obsm['emb'].std(axis=0, keepdims=True)
        sc.pp.neighbors(adata, use_rep='emb', key_added='sf', metric='euclidean')
        sc.tl.leiden(adata, obsp='sf_connectivities', key_added='steamboat_clustering', resolution=.8)
        if 'steamboat_clustering' in adata.obs.columns:
            adata.obs[['steamboat_clustering']].to_csv(f"output/steamboat_attn_cell_type_{i}.csv")
        else:
            print(i, 'failed.')

        ari = sklearn.metrics.adjusted_rand_score(adata.obs['class'], adata.obs['steamboat_clustering'])
        nmi = sklearn.metrics.normalized_mutual_info_score(adata.obs['class'], adata.obs['steamboat_clustering'])
        aris.append(ari)
        nmis.append(nmi)

        print(i, adata.obs['steamboat_clustering'].nunique(), adata.obs['class'].nunique(), ari, nmi)

        del adata
        gc.collect()
        
    df = pd.DataFrame({'ARI': aris, 'NMI': nmis})
    df.to_csv("output/steamboat_cell_type.csv")

# For banksy, see reproducibility/Ex2_mouse_brain/run_banksy_clustering.py


# Summarize (Figure 4g, S2bc)
dfs = {}
if rerun:
    dfs['Steamboat'] = pd.read_csv("output/steamboat_cell_type.csv", index_col=0)
    dfs['PCA'] = pd.read_csv("output/pca_cell_type.csv", index_col=0)
    dfs['NMF'] = pd.read_csv("output/nmf_cell_type.csv", index_col=0)
    dfs['BANKSY 0.2'] = pd.read_csv("output/banksy_cell_type.csv", index_col=0)
else:
    dfs['Steamboat'] = pd.read_csv("../data/Ex2_mouse_brain/clustering/steamboat_attn_cell_type.csv", index_col=0)
    dfs['PCA'] = pd.read_csv("../data/Ex2_mouse_brain/clustering/pca_cell_type.csv", index_col=0)
    dfs['NMF'] = pd.read_csv("../data/Ex2_mouse_brain/clustering/nmf_cell_type.csv", index_col=0)
    dfs['BANKSY 0.2'] = pd.read_csv("../data/Ex2_mouse_brain/clustering/banksy_cell_type.csv", index_col=0)

paragon = 'Steamboat'

ari_df = []
nmi_df = []

columns = []

for i in dfs:
    if i != paragon:
        temp = dfs[paragon] - dfs[i]
        columns.append(i)
        ari_df.append(temp['ARI'])
        nmi_df.append(temp['NMI'])
    else:
        columns.append(i)
        temp = dfs[paragon].copy()
        temp[:] = float('nan')
        ari_df.append(temp['ARI'])
        nmi_df.append(temp['NMI'])

ari_df = pd.concat(ari_df, axis=1)
ari_df.columns = columns
nmi_df = pd.concat(nmi_df, axis=1)
nmi_df.columns = columns

orig_ari_df = []
orig_nmi_df = []

columns = []

for i in dfs:
    temp = dfs[i]
    columns.append(i)
    orig_ari_df.append(temp['ARI'])
    orig_nmi_df.append(temp['NMI'])

orig_ari_df = pd.concat(orig_ari_df, axis=1)
orig_ari_df.columns = columns
orig_nmi_df = pd.concat(orig_nmi_df, axis=1)
orig_nmi_df.columns = columns

fig, axes = plt.subplots(1, 4, figsize=(4, .4 + len(columns) * .24), sharey='row')
sns.violinplot(orig_ari_df, orient='h', ax=axes[0], linewidth=.5, color='C0', width=1.)
axes[0].set_xlabel('ARI')
sns.violinplot(orig_nmi_df, orient='h', ax=axes[2], linewidth=.5, color='C0', width=1.)
axes[2].set_xlabel('NMI')

sns.violinplot(ari_df, orient='h', ax=axes[1], linewidth=.5, color='C1', width=1.)
axes[1].set_xlabel('ΔARI')
axes[1].axvline(0, zorder=-1, c='C3', ls='--', linewidth=.5)

sns.violinplot(nmi_df, orient='h', ax=axes[3], linewidth=.5, color='C1', width=1.)
axes[3].set_xlabel('ΔNMI')
axes[3].axvline(0, zorder=-1, c='C3', ls='--', linewidth=.5)

for i in range(4):
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=60)
    for pos in ['right', 'top']:
        axes[i].spines[pos].set_visible(False)

axes[0].set_ylabel('Clustering')
fig.tight_layout(pad=0.4)
fig.align_xlabels()
fig.savefig("output/fig4g_S2c_clustering.png", bbox_inches="tight", transparent=True)