
# %%
import os
import sys
sys.path.append("../../")
device = "cuda"

cwd = os.getcwd()
if not cwd.endswith("reproducibility/Ex3_crc") and not cwd.endswith("reproducibility\\Ex3_crc"):
    print("Please run this script from the reproducibility/Ex3_crc directory")
    sys.exit(1)
os.makedirs("output", exist_ok=True)

# %%
import scanpy as sc
import squidpy as sq
import pandas as pd
from tqdm.notebook import tqdm
import scipy as sp
import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator

# %%
import torch

# %%
import steamboat as sf
import steamboat.tools

# %%
import pickle as pkl

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import matplotlib
plt.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['font.family'] = 'arial'

# %%
import importlib
importlib.reload(steamboat.tools)

# %%
data_path = "../data/Ex3_crc/"

# %% [markdown]
# ## Process dataset
# 
# You can download the dataset [here](https://data.mendeley.com/datasets/mpjzbtfgfr/1)

# %%
patient_info = pd.read_excel(data_path + "mmc2.xlsx", sheet_name=0, skipfooter=5)

# %%
data = pd.read_csv(data_path + "CRC_clusters_neighborhoods_markers.csv", index_col=0)
features = [
       'CD44 - stroma:Cyc_2_ch_2', 'FOXP3 - regulatory T cells:Cyc_2_ch_3',
       'CD8 - cytotoxic T cells:Cyc_3_ch_2',
       'p53 - tumor suppressor:Cyc_3_ch_3',
       'GATA3 - Th2 helper T cells:Cyc_3_ch_4',
       'CD45 - hematopoietic cells:Cyc_4_ch_2', 'T-bet - Th1 cells:Cyc_4_ch_3',
       'beta-catenin - Wnt signaling:Cyc_4_ch_4', 'HLA-DR - MHC-II:Cyc_5_ch_2',
       'PD-L1 - checkpoint:Cyc_5_ch_3', 'Ki67 - proliferation:Cyc_5_ch_4',
       'CD45RA - naive T cells:Cyc_6_ch_2', 'CD4 - T helper cells:Cyc_6_ch_3',
       'CD21 - DCs:Cyc_6_ch_4', 'MUC-1 - epithelia:Cyc_7_ch_2',
       'CD30 - costimulator:Cyc_7_ch_3', 'CD2 - T cells:Cyc_7_ch_4',
       'Vimentin - cytoplasm:Cyc_8_ch_2', 'CD20 - B cells:Cyc_8_ch_3',
       'LAG-3 - checkpoint:Cyc_8_ch_4', 'Na-K-ATPase - membranes:Cyc_9_ch_2',
       'CD5 - T cells:Cyc_9_ch_3', 'IDO-1 - metabolism:Cyc_9_ch_4',
       'Cytokeratin - epithelia:Cyc_10_ch_2',
       'CD11b - macrophages:Cyc_10_ch_3', 'CD56 - NK cells:Cyc_10_ch_4',
       'aSMA - smooth muscle:Cyc_11_ch_2', 'BCL-2 - apoptosis:Cyc_11_ch_3',
       'CD25 - IL-2 Ra:Cyc_11_ch_4', 'CD11c - DCs:Cyc_12_ch_3',
       'PD-1 - checkpoint:Cyc_12_ch_4',
       'Granzyme B - cytotoxicity:Cyc_13_ch_2', 'EGFR - signaling:Cyc_13_ch_3',
       'VISTA - costimulator:Cyc_13_ch_4', 'CD15 - granulocytes:Cyc_14_ch_2',
       'ICOS - costimulator:Cyc_14_ch_4',
       'Synaptophysin - neuroendocrine:Cyc_15_ch_3',
       'GFAP - nerves:Cyc_16_ch_2', 'CD7 - T cells:Cyc_16_ch_3',
       'CD3 - T cells:Cyc_16_ch_4',
       'Chromogranin A - neuroendocrine:Cyc_17_ch_2',
       'CD163 - macrophages:Cyc_17_ch_3', 'CD45RO - memory cells:Cyc_18_ch_3',
       'CD68 - macrophages:Cyc_18_ch_4', 'CD31 - vasculature:Cyc_19_ch_3',
       'Podoplanin - lymphatics:Cyc_19_ch_4', 'CD34 - vasculature:Cyc_20_ch_3',
       'CD38 - multifunctional:Cyc_20_ch_4',
       'CD138 - plasma cells:Cyc_21_ch_3', 
       'CDX2 - intestinal epithelia:Cyc_2_ch_4',
       'Collagen IV - bas. memb.:Cyc_12_ch_2',
       'CD194 - CCR4 chemokine R:Cyc_14_ch_3',
       'MMP9 - matrix metalloproteinase:Cyc_15_ch_2',
       'CD71 - transferrin R:Cyc_15_ch_4', 'CD57 - NK cells:Cyc_17_ch_4',
       'MMP12 - matrix metalloproteinase:Cyc_21_ch_4']

metadata = data.drop(features, axis=1)
data = data[features]

# %%
p2g_dict = {'CD44': 'CD44',
 'FOXP3': 'FOXP3',
 'CD8': 'CD8A',
 'p53': 'TP53',
 'GATA3': 'GATA3',
 'CD45': 'PTPRC',
 'T-bet': 'TBX21',
 'beta-catenin': 'CTNNB1',
 'HLA-DR': 'HLA-DRA',
 'PD-L1': 'CD274',
 'Ki67': 'MKI67',
 'CD45RA': 'PTPRC',
 'CD4': 'CD4',
 'CD21': 'CR2',
 'MUC-1': 'MUC1',
 'CD30': 'TNFRSF8',
 'CD2': 'CD2',
 'Vimentin': 'VIM',
 'CD20': 'MS4A1',
 'LAG-3': 'LAG3' ,
 'Na-K-ATPase': 'ATP1A1',
 'CD5': 'CD5',
 'IDO-1': 'IDO1',
 'Cytokeratin': 'KRT20',
 'CD11b': 'ITGAM',
 'CD56': 'NCAM',
 'aSMA': 'ACTA2',
 'BCL-2': 'BCL2',
 'CD25': 'IL2RA',
 'CD11c': 'ITGAX',
 'PD-1': 'PDCD1',
 'Granzyme B': 'GZMB',
 'EGFR': 'EGFR',
 'VISTA': 'VSIR',
 'CD15': 'FUT4',
 'ICOS': 'ICOS',
 'Synaptophysin': 'SYP',
 'GFAP': 'GFAP',
 'CD7': 'CD7',
 'CD3': 'CD3D',
 'Chromogranin A': 'CHGA',
 'CD163': 'CD163',
 'CD45RO': 'PTPRC',
 'CD68': 'CD68',
 'CD31': 'PECAM1',
 'Podoplanin': 'PDPN',
 'CD34': 'CD34',
 'CD38': 'CD38',
 'CD138': 'SDC1',
 'CDX2': 'CDX2',
 'Collagen IV': 'COL4A1',
 'CD194': 'CCR4',
 'MMP9': 'MMP9',
 'CD71': 'TFRC',
 'CD57': 'B3GAT1',
 'MMP12': 'MMP12'}

# %%
def feature_map(s):
    if ':' in s:
        s = s.split(':')[0]
    if ' - ' in s:
        s = s.split(' - ')[0]
    return s
short_features = list(map(feature_map, features))
# short_features

# %%
metadata.head().style

# %%
all_adata = sc.AnnData(data, obs=metadata)
all_adata.var_names = short_features
all_adata.obsm['spatial'] = all_adata.obs[['X:X', 'Y:Y']].to_numpy()

adatas = []
for i in all_adata.obs['File Name'].unique():
    adatas.append(all_adata[all_adata.obs['File Name'] == i].copy())
    adatas[-1].obs['global'] = 0
adatas = sf.prep_adatas(adatas)
dataset = sf.make_dataset(adatas, sparse_graph=True, regional_obs=['global'])

# %%
pop_cell_types = all_adata.obs["ClusterName"].value_counts()[all_adata.obs["ClusterName"].value_counts() > all_adata.shape[0] * .005].index.tolist()
pop_cell_types

# %% [markdown]
# ## Train model

# %%
cuda_dataset = dataset.to('cuda')

# %%
sf.set_random_seed(0)
model = sf.Steamboat(short_features, n_heads=10, n_scales=3)
model = model.to(device)
model.load_state_dict(torch.load('../data/saved_models/crc_codex.pth', weights_only=True), strict=False)
sf.tools.calc_obs(adatas, dataset, model, get_recon=False)

# %%
head_weights = sf.tools.calc_head_weights(adatas, model)
sf.tools.plot_head_weights(head_weights, save="output/fig6d_head_weights.png")

# %%
sf.tools.plot_all_transforms(model, top=1, save="output/figS5a_weights.png")

# %%
# T cells (CD3+)
# cytotoxic T cells (CD3+CD8+)
# T helper cells (CD3+CD4+), 
# Tregs (CD3+CD4+CD25+FOXP3+), 
# B cells (CD3-CD20+), 
# plasma cells (CD3-CD20-CD56-CD68-CD163-CD38+)
# NK cells (CD3-CD20-CD56+), 
# CD68+ macrophages (CD3-CD20-CD56-CD68+), 
# CD163 +macrophages (CD3-CD20-CD56-CD163+), 
# CD68+CD163+double-positive macrophages (CD3-CD20-CD56-CD68+CD163+), 
# lymphatics (CD3-CD20-CD56-Podoplanin+),
# vasculature (CD3-CD20-CD56-CD31+), 
# dendritic cells (CD3-CD20-CD56-CD11c+), 
# granulocytes (CD3-CD20-CD56-CD15+)
#############################################################
# Fig6c
#############################################################
chosen_features = ['CD3', 'CD4', 'CD8', 'CD25', 'FOXP3',  # T
                   'CD20', 'CD38',  # B/Plasma
                   'CD56',  # NK
                   'CD68', 'CD163', # Macro
                   'Podoplanin', #lymphatics
                   'CD31', #vasculature
                   'CD11c', #dendritic
                   'CD15', #granulocytes
                   'Cytokeratin'
                  ]

sf.tools.plot_vq(model, chosen_features, save="output/fig6c_reconstruction_metagenes.png")

#############################################################
# Fig6fg
#############################################################
# %%
sample_df = pd.crosstab(all_adata.obs['File Name'], all_adata.obs['neighborhood name'])
sample_df = sample_df.div(sample_df.sum(axis=1), axis=0)
sample_df['TLS'] = sample_df['Follicle'] > 0.00

sample_os = []
for i in patient_info['OS']:
    sample_os.extend([i] * 4)
sample_df['OS'] = sample_os

sample_osc = []
for i in patient_info['OS_Censor']:
    sample_osc.extend([i] * 4)
sample_df['OS_Censor'] = sample_osc

sample_dfs = []
for i in patient_info['DFS']:
    sample_dfs.extend([i] * 4)
sample_df['DFS'] = sample_dfs

sample_df['Patient'] = [i.obs['patients'].unique().item() for i in adatas]
sample_df['Group'] = [i.obs['groups'].unique().item() for i in adatas]

sample_df['Group'] = np.array(['', 'CLR', 'DII'])[sample_df['Group'].astype(int)]
sample_df['TLS'] = np.array(['No', 'Yes'])[sample_df['TLS'].astype(int)]
sample_df.loc[sample_df['Group'] == 'DII', 'TLS'] = 'No'

# %%
for i_head in [2, 8]:
    score_name = f'Global_score_{i_head}'
    sample_df[score_name] = [i.uns['global_k_0'][0, i_head] for i in adatas]
    
    g1 = (sample_df['Group'] == 'CLR') & (sample_df['TLS'] == 'No')
    g2 = (sample_df['Group'] == 'CLR') & (sample_df['TLS'] == 'Yes')
    s, p = sp.stats.mannwhitneyu(sample_df.loc[g1, score_name], 
                                 sample_df.loc[g2, score_name])
    print('CLR Yes vs No:', s / g1.sum() / g2.sum(), p)
    
    g1 = (sample_df['Group'] == 'DII')
    g2 = (sample_df['Group'] == 'CLR') & (sample_df['TLS'] == 'No')
    s, p = sp.stats.mannwhitneyu(sample_df.loc[g1, score_name], 
                                 sample_df.loc[g2, score_name])
    print('DII vs CLR-No:', s / g1.sum() / g2.sum(), p)
    
    g1 = (sample_df['Group'] == 'DII')
    g2 = (sample_df['Group'] == 'CLR')
    s, p = sp.stats.mannwhitneyu(sample_df.loc[g1, score_name], 
                                 sample_df.loc[g2, score_name])
    print('DII vs CLR:', s / g1.sum() / g2.sum(), p)
    
    fig, ax = plt.subplots(figsize=(1.6, 2.4))
    sns.violinplot(sample_df, x='Group', hue='TLS', y=score_name, ax=ax, legend="brief")
    plt.ylabel(f'Global_score_{i_head}')
    ax.grid(axis='y')
    # ax.set_ylim([-49, 180])
    for pos in ['right', 'top']:
        ax.spines[pos].set_visible(False)
    fig.savefig(f"output/fig6fS5b_global_score_{i_head}.png", bbox_inches="tight", transparent=True)

def plot_global_transform(model, d, 
                   top: int = 3, reorder: bool = False, 
                   figsize: str | tuple[float, float] = 'auto'):
    
    q = model.spatial_gather.q.weight[d, :].detach().cpu().numpy()
    k = model.spatial_gather.k_regionals[0].weight[d, :].detach().cpu().numpy()
    v = model.spatial_gather.v.weight[:, d].detach().cpu().numpy()
    
    if top > 0:
        rank_q = np.argsort(-q)[:top]
        rank_k = np.argsort(-k)[:top]
        rank_v = np.argsort(-v)[:top]
        feature_mask = {}
        for j in rank_k:
            feature_mask[j] = None
        for j in rank_q:
            feature_mask[j] = None
        for j in rank_v:
            feature_mask[j] = None
        feature_mask = list(feature_mask.keys())
        chosen_features = np.array(model.features)[feature_mask]
    else:
        feature_mask = list(range(len(model.features)))
        chosen_features = np.array(model.features)

    if figsize == 'auto':
        figsize = (.65, len(chosen_features) * 0.15 + .75)
    # print(figsize)
    fig, ax = plt.subplots(figsize=figsize)
    common_params = {'linewidths': .05, 'linecolor': 'gray', 'yticklabels': chosen_features, 
                     'cmap': 'Reds'}

    to_plot = np.vstack((k[feature_mask],
                         q[feature_mask],
                         v[feature_mask])).T
    true_vmax = to_plot.max(axis=0)
    # print(true_vmax)
    to_plot /= true_vmax

    sns.heatmap(to_plot, xticklabels=['global env', 'center cell', 'reconstruction'], square=True, ax=ax, **common_params)
    ax.set_xticklabels(['global env',  'center cell', 'reconstruction'], rotation=45, ha='right', va='center', rotation_mode='anchor')
    # ax.set_xticklabels(plot_axes[i].get_xticklabels(), rotation=0)
    # ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    return fig

fig = plot_global_transform(model, 2, top=8)
fig.savefig("output/fig6gS5b_global_transform_2.png", bbox_inches="tight", transparent=True)
fig = plot_global_transform(model, 8, top=8)
fig.savefig("output/fig6gS5b_global_transform_8.png", bbox_inches="tight", transparent=True)


#############################################################
# Fig6a,e: Survival analysis
#############################################################
def summarize_samples(df, entity, keys):
    res = {key: [] for key, _ in keys}
    res[entity] = []
    for i in df[entity].unique():
        res[entity].append(i)
        for (key, how) in keys:
            # print(df.loc[df[entity] == i, key])
            if how == 'mean':
                res[key].append(df.loc[df[entity] == i, key].mean())
            elif how == 'median':
                res[key].append(df.loc[df[entity] == i, key].median())
            elif how == 'item':
                res[key].append(df.loc[df[entity] == i, key].unique().item())
    return pd.DataFrame(res)

patient_df = summarize_samples(sample_df, 'Patient', [('OS', 'item'), ('OS_Censor', 'item'), ('Group', 'item')])
patient_df['OS_Censor'] = patient_df['OS_Censor'].astype(bool)
fig, ax = plt.subplots(figsize=(1.2, 1.2))

for group in ("CLR", "DII"):
    mask = patient_df["Group"] == group
    time_survival, survival_prob, conf_int = kaplan_meier_estimator(
        patient_df["OS_Censor"][mask], 
        patient_df["OS"][mask], 
        conf_type="log-log"
    )

    ax.step(time_survival, survival_prob, where="post", label=f"{group}")
    ax.fill_between(time_survival, conf_int[0], conf_int[1], alpha=0.25, step="post")

import sksurv.compare
y = np.array([(i, j) for i, j in zip(patient_df['OS_Censor'], patient_df["OS"])], dtype=[('OS_Censor', '?'), ('Survival_in_months', '<f8')])
chi2, pval = sksurv.compare.compare_survival(y, patient_df["Group"], return_stats=False)

ax.set_title(f'CLR vs DII p={pval:.3f}')
ax.set_ylim(0, 1)
ax.set_ylabel("est. prob. of survival\n$\\hat{S}(t)$")
ax.set_xlabel("time $t$")
ax.legend(loc="best")
ax.spines[['right', 'top']].set_visible(False)

regional = []
for i in range(len(adatas)):
    regional.append(np.mean(adatas[i].obsm['global_attn_0'], axis=0))
fig.savefig("output/fig6a_survival.png", bbox_inches="tight", transparent=True)

for i in [2, 8]:
    sample_df['head_weight'] = np.vstack(regional)[:, i]
    patient_df = summarize_samples(sample_df, 'Patient', [('OS', 'item'), ('OS_Censor', 'item'), ('Group', 'item'), ('head_weight', 'mean')])
    patient_df['OS_Censor'] = patient_df['OS_Censor'].astype(bool)
    patient_df['head_weight_binary'] = 'mid'
    patient_df.loc[patient_df['head_weight'] > patient_df['head_weight'].quantile(.5), 'head_weight_binary'] = 'high'
    patient_df.loc[patient_df['head_weight'] < patient_df['head_weight'].quantile(.5), 'head_weight_binary'] = 'low'
    
    patient_df = patient_df[patient_df['head_weight_binary'] != 'mid']
    
    y = np.array([(i, j) for i, j in zip(patient_df['OS_Censor'], patient_df["OS"])], dtype=[('OS_Censor', '?'), ('Survival_in_months', '<f8')])
    chi2, pval = sksurv.compare.compare_survival(y, patient_df["head_weight_binary"], return_stats=False)
    print(i, pval)
    
    fig, ax = plt.subplots(figsize=(1.2, 1.2))
    for group in ("high", "low"):
        mask = patient_df["head_weight_binary"] == group
        time_survival, survival_prob, conf_int = kaplan_meier_estimator(
            patient_df["OS_Censor"][mask], 
            patient_df["OS"][mask], 
            conf_type="log-log"
        )
    
        ax.step(time_survival, survival_prob, where="post", label=f"{group}", lw=1.)
        ax.fill_between(time_survival, conf_int[0], conf_int[1], alpha=0.25, step="post")
    
    ax.set_title(f'Head #{i} p={pval:.3f}')
    ax.set_ylim(0, 1)
    ax.set_ylabel("Est. prob. of survival\n$\\hat{S}(t)$")
    ax.set_xlabel("time $t$ (months)")
    ax.legend(loc="best")
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(f"output/fig6a_survival_head{i}.png", bbox_inches="tight", transparent=True)

#############################################################
# Fig6hi, S5b: head #8 and #2
#############################################################
sf.tools.plot_cell_type_enrichment(all_adata, adatas, 8, 'ClusterName', pop_cell_types, save="output/fig6h_head8_celltype.png")
sf.tools.plot_cell_type_enrichment(all_adata, adatas, 8, 'neighborhood name', save="output/fig6i_head8_neighborhood.png")

sf.tools.plot_cell_type_enrichment(all_adata, adatas, 2, 'ClusterName', pop_cell_types, save="output/figS5b_head2_celltype.png")
sf.tools.plot_cell_type_enrichment(all_adata, adatas, 2, 'neighborhood name', save="output/figS5b_head2_neighborhood.png")


#############################################################
# Fig6b: cell types and neighborhoods
#############################################################
i_sample = 10

major_cell_type_map = {'tumor cells': 'tumor',
                         'CD68+CD163+ macrophages': 'macrophages',
                         'smooth muscle': 'muscle',
                         'granulocytes': 'granulocytes',
                         'stroma': 'stroma',
                         'CD8+ T cells': 'T cells',
                         'CD4+ T cells CD45RO+': 'T cells',
                         'B cells': 'B cells',
                         'vasculature': 'vasculature',
                         'plasma cells': 'plasma',
                         'dirt': 'dirt',
                         'undefined': 'unclear',
                         'immune cells': 'other immune',
                         'Tregs': 'T cells',
                         'CD4+ T cells': 'T cells',
                         'immune cells / vasculature': 'unclear',
                         'CD68+ macrophages': 'macrophages',
                         'adipocytes': 'adipocytes',
                         'tumor cells / immune cells': 'unclear',
                         'CD11b+CD68+ macrophages': 'macrophages',
                         'CD11b+ monocytes': 'monocytes',
                         'nerves': 'nerves',
                         'CD11c+ DCs': 'DCs',
                         'lymphatics': 'lymphatics',
                         'NK cells': 'NK cells',
                         'CD3+ T cells': 'T cells',
                         'CD68+ macrophages GzmB+': 'macrophages',
                         'CD4+ T cells GATA3+': 'T cells',
                         'CD163+ macrophages': 'macrophages'}

coarse_cell_type_map = {'tumor cells': 'tumor',
                         'CD68+CD163+ macrophages': 'immune',
                         'smooth muscle': 'muscle',
                         'granulocytes': 'immune',
                         'stroma': 'stroma',
                         'CD8+ T cells': 'immune',
                         'CD4+ T cells CD45RO+': 'immune',
                         'B cells': 'immune',
                         'vasculature': 'vasculature',
                         'plasma cells': 'immune',
                         'dirt': 'dirt',
                         'undefined': 'unclear',
                         'immune cells': 'immune',
                         'Tregs': 'immune',
                         'CD4+ T cells': 'immune',
                         'immune cells / vasculature': 'unclear',
                         'CD68+ macrophages': 'immune',
                         'adipocytes': 'adipocytes',
                         'tumor cells / immune cells': 'unclear',
                         'CD11b+CD68+ macrophages': 'immune',
                         'CD11b+ monocytes': 'immune',
                         'nerves': 'nerves',
                         'CD11c+ DCs': 'immune',
                         'lymphatics': 'immune',
                         'NK cells': 'immune',
                         'CD3+ T cells': 'immune',
                         'CD68+ macrophages GzmB+': 'immune',
                         'CD4+ T cells GATA3+': 'immune',
                         'CD163+ macrophages': 'immune'}

all_adata.obs['MajorCellTypes'] = all_adata.obs['ClusterName'].apply(major_cell_type_map.__getitem__)
all_adata.obs['CoarseCellTypes'] = all_adata.obs['ClusterName'].apply(coarse_cell_type_map.__getitem__)

neighborhood_palette = {i: j for i, j in zip(sorted(all_adata.obs['neighborhood name'].dropna().unique().tolist()), sc.pl.palettes.vega_10_scanpy)}
adatas[i_sample].uns['neighborhood name_colors'] = [neighborhood_palette[i] for i in sorted(adatas[i_sample].obs['neighborhood name'].dropna().unique().tolist())]
adatas[i_sample].obs['CoarseCellTypes'] = adatas[i_sample].obs['ClusterName'].apply(coarse_cell_type_map.__getitem__)
ax = sq.pl.spatial_scatter(adatas[i_sample], color='CoarseCellTypes', shape=None, figsize=(4, 2), size=1., cmap='Reds', frameon=False, return_ax=True)
ax.figure.savefig("output/fig6b_insitu_cell_types.png", bbox_inches="tight", transparent=True)
ax = sq.pl.spatial_scatter(adatas[i_sample], color=['neighborhood name'], shape=None, figsize=(5, 2), size=1., cmap='Reds', frameon=False, return_ax=True)
ax.figure.savefig("output/fig6b_insitu_neighborhoods.png", bbox_inches="tight", transparent=True)

####################################################################################
# Fig6j GATA3 ~ DII/CLR
####################################################################################

adatas[0].uns['neighborhood name_colors'] = [neighborhood_palette[i] for i in sorted(adatas[0].obs['neighborhood name'].dropna().unique().tolist())]
axes = sq.pl.spatial_scatter(adatas[0], color=['GATA3', 'neighborhood name'], shape=None, figsize=(2, 1.5), size=1., cmap='Reds', frameon=False, vmin=2, vmax=6, return_ax=True)
axes[0].figure.savefig("output/fig6j_insitu_GATA3.png", bbox_inches="tight", transparent=True)

adatas[10].uns['neighborhood name_colors'] = [neighborhood_palette[i] for i in sorted(adatas[10].obs['neighborhood name'].dropna().unique().tolist())]
axes = sq.pl.spatial_scatter(adatas[10], color=['GATA3', 'neighborhood name'], shape=None, figsize=(2, 1.5), size=1., cmap='Reds', frameon=False, vmin=2, vmax=6, return_ax=True)
axes[0].figure.savefig("output/fig6j_insitu_GATA3.png", bbox_inches="tight", transparent=True)


################################################################################################
# Comparing STEAMBOAT, PCA, and NMF on disentangling sample-level features.
################################################################################################
# %%
avg_exprs = {adata.obs['File Name'].astype(str).unique().item(): adata.X.mean(axis=0) for adata in adatas}
avg_exprs = np.vstack([avg_exprs[i] for i in sample_df.index])

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

X = avg_exprs.copy()
nmf = NMF(n_components=10, init='random', random_state=0)
nmf_scores = nmf.fit_transform(X)
nmf_cols = [f"NMF{i+1}" for i in range(nmf_scores.shape[1])]
nmf_scores = pd.DataFrame(nmf_scores, index=sample_df.index, columns=nmf_cols)

# z-score features then run PCA
scaler = StandardScaler()
X = scaler.fit_transform(avg_exprs)

pca = PCA()
scores = pca.fit_transform(X)

pc_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
pca_scores = pd.DataFrame(scores, index=sample_df.index, columns=pc_cols)

explained_variance = pd.Series(pca.explained_variance_ratio_, index=pc_cols)

# %%
sample_df = pd.merge(sample_df, pca_scores, left_index=True, right_index=True)
sample_df = pd.merge(sample_df, nmf_scores, left_index=True, right_index=True)

# %%
sample_df

# %% [markdown]
# ## Survival analysis

# %%
import sksurv.compare

# %%
def summarize_samples(df, entity, keys):
    res = {key: [] for key, _ in keys}
    res[entity] = []
    for i in df[entity].unique():
        res[entity].append(i)
        for (key, how) in keys:
            # print(df.loc[df[entity] == i, key])
            if how == 'mean':
                res[key].append(df.loc[df[entity] == i, key].mean())
            elif how == 'median':
                res[key].append(df.loc[df[entity] == i, key].median())
            elif how == 'item':
                res[key].append(df.loc[df[entity] == i, key].unique().item())
    return pd.DataFrame(res)

patient_df = summarize_samples(sample_df, 'Patient', [('OS', 'item'), ('OS_Censor', 'item'), ('Group', 'item')])
patient_df['OS_Censor'] = patient_df['OS_Censor'].astype(bool)
patient_df

## steamboat
steamboat_p = []
regional = []
for i in range(len(adatas)):
    regional.append(np.mean(adatas[i].obsm['global_attn_0'], axis=0))

for i in range(10):
    sample_df['head_weight'] = np.vstack(regional)[:, i]
    patient_df = summarize_samples(sample_df, 'Patient', [('OS', 'item'), ('OS_Censor', 'item'), ('Group', 'item'), ('head_weight', 'mean')])
    patient_df['OS_Censor'] = patient_df['OS_Censor'].astype(bool)
    patient_df['head_weight_binary'] = 'mid'
    patient_df.loc[patient_df['head_weight'] > patient_df['head_weight'].quantile(.5), 'head_weight_binary'] = 'high'
    patient_df.loc[patient_df['head_weight'] < patient_df['head_weight'].quantile(.5), 'head_weight_binary'] = 'low'
    
    patient_df = patient_df[patient_df['head_weight_binary'] != 'mid']
    
    y = np.array([(i, j) for i, j in zip(patient_df['OS_Censor'], patient_df["OS"])], dtype=[('OS_Censor', '?'), ('Survival_in_months', '<f8')])
    chi2, pval = sksurv.compare.compare_survival(y, patient_df["head_weight_binary"], return_stats=False)
    print(i, pval)
    steamboat_p.append(pval)

    # If you want to plot the Kaplan-Meier curves for each head, you can uncomment the following code:
    # fig, ax = plt.subplots(figsize=(1.2, 1.2))
    # for group in ("high", "low"):
    #     mask = patient_df["head_weight_binary"] == group
    #     time_survival, survival_prob, conf_int = kaplan_meier_estimator(
    #         patient_df["OS_Censor"][mask], 
    #         patient_df["OS"][mask], 
    #         conf_type="log-log"
    #     )
    # 
    #     ax.step(time_survival, survival_prob, where="post", label=f"{group}", lw=1.)
    #     ax.fill_between(time_survival, conf_int[0], conf_int[1], alpha=0.25, step="post")
    # 
    # ax.set_title(f'Head #{i} p={pval:.3f}')
    # ax.set_ylim(0, 1)
    # ax.set_ylabel("Est. prob. of survival\n$\\hat{S}(t)$")
    # ax.set_xlabel("time $t$ (months)")
    # ax.legend(loc="best")
    # ax.spines[['right', 'top']].set_visible(False)
    
pd.DataFrame({'steamboat_p': steamboat_p}).to_csv('output/steamboat_pvals.csv')

## PCA
pca_baseline = []
for i in range(10):
    sample_df['head_weight'] = sample_df[f'PC{i+1}']
    patient_df = summarize_samples(sample_df, 'Patient', [('OS', 'item'), ('OS_Censor', 'item'), ('Group', 'item'), ('head_weight', 'mean')])
    patient_df['OS_Censor'] = patient_df['OS_Censor'].astype(bool)
    patient_df['head_weight_binary'] = 'mid'
    patient_df.loc[patient_df['head_weight'] > patient_df['head_weight'].quantile(.5), 'head_weight_binary'] = 'high'
    patient_df.loc[patient_df['head_weight'] < patient_df['head_weight'].quantile(.5), 'head_weight_binary'] = 'low'
    
    patient_df = patient_df[patient_df['head_weight_binary'] != 'mid']
    
    y = np.array([(i, j) for i, j in zip(patient_df['OS_Censor'], patient_df["OS"])], dtype=[('OS_Censor', '?'), ('Survival_in_months', '<f8')])
    chi2, pval = sksurv.compare.compare_survival(y, patient_df["head_weight_binary"], return_stats=False)
    print(i, pval)
    pca_baseline.append(pval)
    
    # If you want to plot the Kaplan-Meier curves for each head, you can uncomment the following code:
    # fig, ax = plt.subplots(figsize=(1.2, 1.2))
    # for group in ("high", "low"):
    #     mask = patient_df["head_weight_binary"] == group
    #     time_survival, survival_prob, conf_int = kaplan_meier_estimator(
    #         patient_df["OS_Censor"][mask], 
    #         patient_df["OS"][mask], 
    #         conf_type="log-log"
    #     )
    # 
    #     ax.step(time_survival, survival_prob, where="post", label=f"{group}", lw=1.)
    #     ax.fill_between(time_survival, conf_int[0], conf_int[1], alpha=0.25, step="post")
    # 
    # ax.set_title(f'Head #{i} p={pval:.3f}')
    # ax.set_ylim(0, 1)
    # ax.set_ylabel("Est. prob. of survival\n$\\hat{S}(t)$")
    # ax.set_xlabel("time $t$ (months)")
    # ax.legend(loc="best")
    # ax.spines[['right', 'top']].set_visible(False)

pd.DataFrame({'pca_p': pca_baseline}).to_csv('output/pca_pvals.csv')

## NMF
nmf_baseline = []
for i in range(10):
    sample_df['head_weight'] = sample_df[f'NMF{i+1}']
    patient_df = summarize_samples(sample_df, 'Patient', [('OS', 'item'), ('OS_Censor', 'item'), ('Group', 'item'), ('head_weight', 'mean')])
    patient_df['OS_Censor'] = patient_df['OS_Censor'].astype(bool)
    patient_df['head_weight_binary'] = 'mid'
    patient_df.loc[patient_df['head_weight'] > patient_df['head_weight'].quantile(.5), 'head_weight_binary'] = 'high'
    patient_df.loc[patient_df['head_weight'] < patient_df['head_weight'].quantile(.5), 'head_weight_binary'] = 'low'
    
    patient_df = patient_df[patient_df['head_weight_binary'] != 'mid']
    
    y = np.array([(i, j) for i, j in zip(patient_df['OS_Censor'], patient_df["OS"])], dtype=[('OS_Censor', '?'), ('Survival_in_months', '<f8')])
    chi2, pval = sksurv.compare.compare_survival(y, patient_df["head_weight_binary"], return_stats=False)
    print(i, pval)
    nmf_baseline.append(pval)
    
    # If you want to plot the Kaplan-Meier curves for each head, you can uncomment the following code:
    # fig, ax = plt.subplots(figsize=(1.2, 1.2))
    # for group in ("high", "low"):
    #     mask = patient_df["head_weight_binary"] == group
    #     time_survival, survival_prob, conf_int = kaplan_meier_estimator(
    #         patient_df["OS_Censor"][mask], 
    #         patient_df["OS"][mask], 
    #         conf_type="log-log"
    #     )
    # 
    #     ax.step(time_survival, survival_prob, where="post", label=f"{group}", lw=1.)
    #     ax.fill_between(time_survival, conf_int[0], conf_int[1], alpha=0.25, step="post")
    # 
    # ax.set_title(f'Head #{i} p={pval:.3f}')
    # ax.set_ylim(0, 1)
    # ax.set_ylabel("Est. prob. of survival\n$\\hat{S}(t)$")
    # ax.set_xlabel("time $t$ (months)")
    # ax.legend(loc="best")
    # ax.spines[['right', 'top']].set_visible(False)

pd.DataFrame({'nmf_p': nmf_baseline}).to_csv('output/nmf_pvals.csv')

# Summarizing
pca_p_df = pd.read_csv('output/pca_pvals.csv', index_col=0)
steamboat_p_df = pd.read_csv('output/steamboat_pvals.csv', index_col=0)
nmf_p_df = pd.read_csv('output/nmf_pvals.csv', index_col=0)


disp_df = pca_p_df.sort_values('pca_p').reset_index()
disp_df = pd.merge(disp_df, nmf_p_df.sort_values('nmf_p').reset_index(), left_index=True, right_index=True, suffixes=('_pca', '_nmf'))
disp_df = pd.merge(disp_df, steamboat_p_df.sort_values('steamboat_p').reset_index(), left_index=True, right_index=True, suffixes=('', '_steamboat'))

disp_df.columns = ["pca_index", "pca_p", "nmf_index", "nmf_p", "steamboat_index", "steamboat_p"]
# disp_df.to_csv('output/steamboat_crc_pca_nmf_head_pvals.csv', index=False)
disp_df


fig, axes = plt.subplots(1, 3, figsize=(5.25, 2.5), sharey=True)

sns.barplot(x=list(range(0, 10)), y=-np.log10(pca_p_df['pca_p'].sort_values()), ax=axes[0], color='C0')
axes[0].axhline(-np.log10(0.05), color='red', linestyle='--')
axes[0].set_xlabel('Principal components')
axes[0].set_xticklabels(pca_p_df['pca_p'].sort_values().index)

sns.barplot(x=list(range(0, 10)), y=-np.log10(steamboat_p_df['steamboat_p'].sort_values()), ax=axes[2], color='C1')
axes[2].axhline(-np.log10(0.05), color='red', linestyle='--')
axes[2].set_xlabel('Steamboat global')
axes[2].set_xticklabels(steamboat_p_df['steamboat_p'].sort_values().index)

sns.barplot(x=list(range(0, 10)), y=-np.log10(nmf_p_df['nmf_p'].sort_values()), ax=axes[1], color='C2')
axes[1].axhline(-np.log10(0.05), color='red', linestyle='--')
axes[1].set_xlabel('NMF components')
axes[1].set_xticklabels(nmf_p_df['nmf_p'].sort_values().index)

ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
axes[0].set_yticks(-np.log10(ticks))
axes[0].set_yticklabels(ticks)

axes[0].grid(axis='y')
axes[1].grid(axis='y')
axes[2].grid(axis='y')

for ax in axes:
    ax.set_ylabel('Survival p-value\n(Negative log scale)')
    #ax.set_xlabel('Head index')
    ax.spines[['right', 'top']].set_visible(False)
fig.tight_layout()

fig.savefig('output/steamboat_pca_nmf.png', bbox_inches='tight')