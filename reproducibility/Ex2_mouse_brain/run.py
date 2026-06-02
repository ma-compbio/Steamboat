# %% [markdown]
# # Train on mouse brain data
# 
# This could take several hours. The trained model is provided, which can be used to try the interpretation part without training again.

# %%
import os
if os.getcwd() != os.path.dirname(os.path.realpath(__file__)):
    print("Please run this script from its own directory to avoid path issues.")
    print("Current working directory:", os.getcwd())
    exit(1)

os.makedirs("output", exist_ok=True)
os.makedirs("tmp_adata", exist_ok=True)


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


adatas = []
for i in adata.obs['brain_section_label'].unique():
    adatas.append(adata[adata.obs['brain_section_label'] == i])
    adatas[-1].obs['global'] = 0

adatas = sf.prep_adatas(adatas, norm=False, log1p=False)
dataset = sf.make_dataset(adatas, sparse_graph=True, regional_obs=['global'])


sf.set_random_seed(0)
model = sf.Steamboat(adata.var_names.tolist(), n_heads=50, n_scales=3)
model = model.to(device)
model.load_state_dict(torch.load('../data/saved_models/mmbrain.pth', weights_only=True), strict=False)
sf.tools.calc_obs(adatas, dataset, model)

# Save the annotated adata and individual adatas for future use
# adata.write_h5ad("tmp_adata/mmbrain.h5ad")

# for i, adata in enumerate(tqdm(adatas)):
#     adata.write_h5ad(f"tmp_adata/mmbrain_{i}.h5ad")


#############################################################################
# Fig4a
#############################################################################
sq.pl.spatial_scatter(adatas[0], color='class', shape=None, figsize=(5, 5), size=1., lw=0.0, legend_fontsize=9, title="", frameon=False)
plt.savefig("output/fig4a_insitu_cell_type.png", dpi=300, bbox_inches='tight')


#############################################################################
# Fig4b
#############################################################################
sq.pl.spatial_scatter(adatas[0], color='parcellation_division', shape=None, figsize=(5, 5), size=1., lw=0.0, legend_fontsize=9, title="", frameon=False)
plt.savefig("output/fig4b_insitu_region.png", dpi=300, bbox_inches='tight')


#############################################################################
# Fig4c,d
#############################################################################
for i_comp in [5, 24]: ## Change this to see more
    adatas[0].obs['a'] = adatas[0].obsm['local_attn'][:, i_comp]
    adatas[0].obs['q'] = adatas[0].obsm['q'][:, i_comp]
    adatas[0].obs['k'] = adatas[0].obsm['local_k'][:, i_comp]
    
    axes = sq.pl.spatial_scatter(adatas[0], color=['a', 'q', 'k'], shape=None, figsize=(2, 2), size=.25, 
                          legend_fontsize=9, cmap='Reds', ncols=3, colorbar=False, vmin=0., wspace=.0, outline=False, 
                          frameon=False, title=['Attention', 'Query', 'Key'], return_ax=True)
    axes[0].figure.savefig(f"output/fig4c_insitu_attention_head{i_comp}.png", dpi=200, transparent=False, bbox_inches='tight')
    
    temp_adata = adatas[0].copy()
    good_classes = adatas[0].obs['class'].value_counts()
    good_classes = good_classes[good_classes > temp_adata.shape[0] * 0.01].index.to_list()
    temp_adata = temp_adata[temp_adata.obs['class'].isin(good_classes)]
    
    fig, ax = plt.subplots(figsize=(2, 3.5))
    sc.pl.matrixplot(temp_adata, ['a', 'q', 'k'], 'class', standard_scale='var', ax=ax, cmap='Reds', show=False)
    fig.savefig(f"output/fig4d_heatmap_attention_head{i_comp}.png", dpi=200, transparent=False, bbox_inches='tight')


#############################################################################
# FigS2a
#############################################################################
head_weights = sf.tools.calc_head_weights(adatas, model)
sf.tools.plot_head_weights(head_weights, figsize=(12, 1), multiplier=1000, heatmap_kwargs={'vmax': 50}, save="output/figS2a_head_weights.png")


#############################################################################
# Fig4e
#############################################################################
adata = sc.read_h5ad("tmp_adata/mmbrain_0.h5ad")
adata.obs['class_region'] = adata.obs['class'].astype(str) + '@' + adata.obs['parcellation_division'].astype(str)
astro_adata = adata[adata.obs['class'] == '30 Astro-Epen']

fig, ax = plt.subplots(figsize=(3, 3))
astro_adata = sc.AnnData(astro_adata.obsm['attn'], obs=astro_adata.obs)
sc.pp.scale(astro_adata)
sc.pp.neighbors(astro_adata)
sc.tl.umap(astro_adata)
sc.pl.umap(astro_adata, color=['class_region'], ax=ax, frameon=False, show=False)
fig.savefig(f"output/fig4f_umap_astro_epen_region.png", bbox_inches='tight')


#############################################################################
# Fig4f: global ~ z-axis
#############################################################################
glb = []
z = []
donor = []
for i in range(129):
    adata = sc.read_h5ad(f"tmp_adata/mmbrain_{i}.h5ad")
    glb.append(adata.uns['global_k_0'][0, 2])
    z.append(adata.obs['z'].unique().item())
    donor.append(adata.obs['donor_label'].astype(str).unique().item())

fig, ax = plt.subplots(figsize=(1.2, 1.2))
plt.scatter(z, glb, s=2.)
for pos in ['right', 'top']:
    ax.spines[pos].set_visible(False)
ax.set_xlabel('Z-coordinate')
ax.set_ylabel("Global env't score\nhead 2")
fig.savefig(f"output/fig4f_zcoord.png", transparent=False, bbox_inches='tight')

#############################################################################
# Fig4g, S2b, S3
#############################################################################
# This takes a long time so we put it in a separate script: run_clusters_and_domains.py

#############################################################################
# Fig4h, S7: Ligand-receptor pairs
#############################################################################
lrdb = pd.read_csv("../data/Ex2_mouse_brain/CellChatDB.mouse.csv.gz", index_col=0)

def parse_complex(s):
    if s[0] != '(':
        return [s]
    else:
        return s[1:-1].split('+')
        
def parse_lr(s):
    l, r = s.split(' - ')
    return parse_complex(l), parse_complex(r)

lrp = []
for i in lrdb['interaction_name_2']:
    ls, rs = parse_lr(i)
    for l in ls:
        for r in rs:
            lrp.append((l.strip(), r.strip()))

n_heads = 50

k_local = model.spatial_gather.k_local.weight.detach().cpu().numpy()
k_global = model.spatial_gather.k_regionals[0].weight.detach().cpu().numpy()
q = model.spatial_gather.q.weight.detach().cpu().numpy()
v = model.spatial_gather.v.weight.detach().cpu().numpy().T

adata = sc.read_h5ad(f"tmp_adata/mmbrain_{0}.h5ad")

index = ([f'k_local_{i}' for i in range(n_heads)] + 
         [f'k_global_{i}' for i in range(n_heads)] + 
         [f'q_{i}' for i in range(n_heads)] + 
         [f'v_{i}' for i in range(n_heads)])
gene_df = pd.DataFrame(np.vstack([k_local, k_global, q, v]), 
                       index=index, columns=adata.var['gene_symbol']).T

normalized_gene_df = gene_df.multiply(adata.X.mean(axis=0), axis=0)
normalized_gene_df /= normalized_gene_df.max(axis=0)
lrp_dfs = []
for i in tqdm(range(n_heads)):
    lrp_df = pd.DataFrame(lrp, columns=['ligand', 'receptor'])
    lrp_df = lrp_df.drop_duplicates()
    lrp_df['lr'] = lrp_df['ligand'] + '-' + lrp_df['receptor']
    lrp_df = lrp_df[(lrp_df['ligand'].isin(adata.var['gene_symbol'].values)) & (lrp_df['receptor'].isin(adata.var['gene_symbol'].values))]

    lrp_df['kl_score'] = np.log(normalized_gene_df.loc[lrp_df['ligand'].tolist(), f'k_local_{i}'].tolist())
    lrp_df['qr_score'] = np.log(normalized_gene_df.loc[lrp_df['receptor'].tolist(), f'q_{i}'].tolist())
    lrp_df['lr_score'] = lrp_df['kl_score'] + lrp_df['qr_score']
    
    lrp_df['ql_score'] = np.log(normalized_gene_df.loc[lrp_df['ligand'].tolist(), f'q_{i}'].tolist())
    lrp_df['kr_score'] = np.log(normalized_gene_df.loc[lrp_df['receptor'].tolist(), f'k_local_{i}'].tolist())
    lrp_df['rl_score'] = lrp_df['kr_score'] + lrp_df['ql_score']

    lrp_df['k_to_q'] = lrp_df['lr_score'] > lrp_df['rl_score']
    
    lrp_df['score'] = np.maximum(lrp_df['lr_score'], lrp_df['rl_score'])
    
    xy = np.maximum((lrp_df['kl_score'].to_numpy() + lrp_df['qr_score'].to_numpy()[:, None]).flatten(),
                    (lrp_df['kr_score'].to_numpy() + lrp_df['ql_score'].to_numpy()[:, None]).flatten())
    lrp_df['p'] = (lrp_df['score'].to_numpy()[:, None] < xy).sum(axis=1) / len(xy)

    # xy = (lrp_df['kl_score'].to_numpy() + lrp_df['qr_score'].to_numpy()[:, None]).flatten(),
    # lrp_df['p_lr'] = (lrp_df['lr_score'].to_numpy()[:, None] < xy).sum(axis=1) / len(xy)

    # lrp_df['p_rl'] = float('nan')
    # xy = np.maximum((lrp_df['kl_score'].to_numpy() + lrp_df['qr_score'].to_numpy()[:, None]).flatten(),
    #                 (lrp_df['kr_score'].to_numpy() + lrp_df['ql_score'].to_numpy()[:, None]).flatten())
    
    # lrp_df['p'] = (lrp_df['lr_score'].to_numpy()[:, None] < xy).sum(axis=1) / len(xy)
    
    lrp_df['adj_p'] = sp.stats.false_discovery_control(lrp_df['p'])
    
    lrp_dfs.append(lrp_df.sort_values('p'))
    
# Fig S7
good_lrs = {}
for i in range(n_heads):
    temp = lrp_dfs[i][lrp_dfs[i]['adj_p'] <= 0.2]
    for lr, p in zip(temp['lr'], temp['adj_p']):
        if lr not in good_lrs:
            good_lrs[lr] = p
        else:
            good_lrs[lr] = min(good_lrs[lr], p)

sorted_lrs = sorted(good_lrs.items(), key=lambda x: x[1])
print(len(sorted_lrs))
inv_sorted_lrs = {x[0]: idx for idx, x in enumerate(sorted_lrs)}

heatmap_df = pd.DataFrame(float('nan'), index=[x[0] for x in sorted_lrs], columns=list(range(n_heads)))
for i in range(n_heads):
    temp = lrp_dfs[i][lrp_dfs[i]['adj_p'] <= 0.2]
    for lr, p in zip(temp['lr'], temp['adj_p']):
        if lr in heatmap_df.index:
            heatmap_df.loc[lr, i] = p

heatmap_df = heatmap_df.loc[:, heatmap_df.isna().sum(axis=0) < heatmap_df.shape[0]]

fig, ax = plt.subplots(figsize=(4.5, 3.25))
sns.heatmap(heatmap_df, cmap='Reds_r', ax=ax, cbar_kws={'shrink': 0.5}, linewidths=.5, linecolor='lightgray')
ax.set_xticks(np.arange(len(heatmap_df.columns)) + 0.5)
ax.set_xticklabels(heatmap_df.columns.tolist(), rotation=0)
ax.set_xlabel('Attention Heads')
ax.set_ylabel('Ligand-Receptor Pairs')
fig.savefig('output/figS7_lr_summary_heatmap.png', **pltkw)
    
# Fig 4h
fig, ax = plt.subplots(figsize=(1.2, 0.65))
lrp_dfs[5][lrp_dfs[5]['p'] < 0.05].plot.scatter(y='lr', x='p',  ax=ax)
ax.set_xticks([0.01, 0.03])
ax.set_ylabel('Ligand-\nreceptor')
ax.set_xlabel('P value')
for pos in ['right', 'top']:
    ax.spines[pos].set_visible(False)
ax.set_ylim([-0.5, 2.5])
ax.set_axisbelow(True)
ax.grid(axis='both', zorder=0)
fig.savefig("output/fig4h_lrp_pvalue.png", **pltkw)

def plot_lr(adata, l, r, l_cutoff=0, r_cutoff=0, figsize=(10, 10)):
    adata.obs[f'{l}-{r}'] = ''
    adata.obs.loc[(adata[:, l].X > l_cutoff).squeeze(), f'{l}-{r}'] += f'{l}+'
    adata.obs.loc[(adata[:, r].X > r_cutoff).squeeze(), f'{l}-{r}'] += f'{r}+'
    adata.obs.loc[adata.obs[f'{l}-{r}'] == '', f'{l}-{r}'] = 'Neither'
    adata.uns[f'{l}-{r}_colors'] = {'#f0f0f0', 'C0', 'C3', 'C4'}
    adata.obs[f'{l}-{r}'] = adata.obs[f'{l}-{r}'].astype('category')

    any = adata.obs_names[adata.obs[f'{l}-{r}'] != 'Neither'].tolist()
    neither = adata.obs_names[adata.obs[f'{l}-{r}'] == 'Neither'].tolist()

    adata = adata[neither + any, :]
    color_map = {'Neither': '#f0f0f0', f'{l}+': 'C0', f'{r}+': 'C3', f'{l}+{r}+': 'C4'}
    adata.uns[f'{l}-{r}_colors'] = [color_map[i] for i in adata.obs[f'{l}-{r}'].cat.categories]
    return sq.pl.spatial_scatter(adata, color=[f'{l}-{r}'], shape=None, figsize=figsize, size=1., 
                          legend_fontsize=12, na_color='#f0f0f0', frameon=False, title='', alpha=.75, outline=False, return_ax=True)

adata.var_names = adata.var['gene_symbol']

for i in lrp_dfs[5][lrp_dfs[5]['p'] < 0.05].index:
    ax = plot_lr(adata, l=lrp_dfs[5].loc[i, 'ligand'], r=lrp_dfs[24].loc[i, 'receptor'], figsize=(3, 2), l_cutoff=.75, r_cutoff=.75)
    ax.figure.savefig(f"output/fig4hS2d_{lrp_dfs[5].loc[i, 'ligand']}_{lrp_dfs[24].loc[i, 'receptor']}.png", dpi=300, bbox_inches='tight', transparent=False)


#############################################################################
# FigS4
#############################################################################
adata = adatas[0].copy()
adata.var_names = adata.var['gene_symbol']
ax = sq.pl.spatial_scatter(adata, color='Mog', shape=None, figsize=(3, 3), size=1., legend_fontsize=9, cmap='Reds', return_ax=True)
ax.figure.savefig("output/figS4_insitu_Mog.png", dpi=300, bbox_inches='tight', transparent=False)

adata.obs['OPC-Oligo'] = "Other"
adata.obs.loc[adata.obs['class'].isin(['31 OPC-Oligo']), 'OPC-Oligo'] = 'OPC-Oligo'
ax = sq.pl.spatial_scatter(adata, color='OPC-Oligo', shape=None, figsize=(3, 3), size=1., legend_fontsize=9, cmap='Reds', return_ax=True)
ax.figure.savefig("output/figS4_insitu_OPCOligo.png", dpi=300, bbox_inches='tight', transparent=False)
