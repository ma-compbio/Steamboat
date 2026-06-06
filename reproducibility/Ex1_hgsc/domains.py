# %% [markdown]
# # Ovarian cancer data analysis
# Train steamboat model on HGSC data.

# %%
import os
import scanpy as sc
import squidpy as sq
import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json

# %%
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'arial'

pltkw = dict(bbox_inches='tight', transparent=True)

# %%
import sys
sys.path.append("..")
import steamboat as sf
import steamboat.tools
import torch
device = 'cuda'

# %%
import importlib
importlib.reload(steamboat.tools)

# %%
# https://www.nature.com/articles/s41590-024-01943-5

# %% [markdown]
# ## Load data

# %%
regenerate = False

h5ad_file = "../data/Ex1_hgsc/ST_Discovery_so.h5ad"
if (not os.path.exists(h5ad_file)) or regenerate:
    adata = sc.read_mtx("../data/Ex1_hgsc/ST_Discovery_so_counts.mtx").T
    metadata = pd.read_csv("../data/Ex1_hgsc/ST_Discovery_so_metadata.csv", index_col=0)
    features = pd.read_csv("../data/Ex1_hgsc/ST_Discovery_so_features.txt", index_col=0, header=None)
    features.index = features.index.str.strip() # remove trailing white space in gene names
    features.index.name = 'gene_symbol'
    adata.obs = metadata
    adata.var = features
    adata.obsm['spatial'] = adata.obs[['x', 'y']].to_numpy()
    adata.write_h5ad("../data/Ex1_hgsc/h5ad/ST_Discovery_so.h5ad")
else:
    adata = sc.read_h5ad(h5ad_file)

# %%
adata.obs

# %%
TNK_info = adata.obs[adata.obs['cell.types'] == 'Monocyte']
TNK_info['cell.subtypes'].value_counts()

# %%
## Metadata and gene sets

sample_metadata = pd.read_excel("../data/Ex1_hgsc/sample_metadata.xlsx", index_col=0, sheet_name='Table 2b', skiprows=1)
sample_metadata = sample_metadata[sample_metadata['dataset'] == 'Discovery']

celltype_signatures = pd.read_excel("../data/Ex1_hgsc/sample_metadata.xlsx", sheet_name='Table 3a', skiprows=2)
mtil_signautures = pd.read_excel("../data/Ex1_hgsc/sample_metadata.xlsx", sheet_name='Table 6a', skiprows=2)
desmoplasia_signautures = pd.read_excel("../data/Ex1_hgsc/sample_metadata.xlsx", sheet_name='Table 5a', skiprows=2)

def purge_gene_sets(df, prefix=''):
    res = {}
    for i in df.columns:
        res[prefix + i] = df[i].dropna().tolist()
    return res
celltype_signatures = pd.read_excel("../data/Ex1_hgsc/sample_metadata.xlsx", index_col=0, sheet_name='Table 3b', skiprows=2).iloc[:, :-3]
genesets = (purge_gene_sets(celltype_signatures, 'sig_') | 
            purge_gene_sets(mtil_signautures, 'mtil_') | 
            purge_gene_sets(mtil_signautures, 'mtil_'))
genesets.keys()
del genesets['sig_Mast.cell']

sample_metadata

# %%
## Find untreated, adnexa samples

columns_of_interest = ['sites_binary', 'stage', 'treatment']
fig, axes = plt.subplots(1, len(columns_of_interest), figsize=(len(columns_of_interest) * 1.5, 3))
for i, column in enumerate(columns_of_interest):
    sample_metadata[column].value_counts().plot(kind='bar', ax=axes[i])
    axes[i].set_title(column)
plt.tight_layout()

mask = (sample_metadata['sites_binary'] == 'Adnexa') & (sample_metadata['treatment'] == 'Untreated')
samples_of_interest = sample_metadata.index[mask].tolist()

all_adata = adata[adata.obs['samples'].isin(samples_of_interest)].copy()
all_adata.obs['cell.types.nolc'] = all_adata.obs['cell.types'].str.replace('_LC', '')

# %%
# selected_samples = np.random.choice(all_adata.obs['samples'].unique(), size=10, replace=False)
# all_adata = all_adata[all_adata.obs['samples'].isin(selected_samples)].copy()

# %%


# %% [markdown]
# ### Process data to create torch dataset

# %%
# Separate individual slides
adatas = []
for i in all_adata.obs['samples'].unique():
    temp = all_adata[all_adata.obs['samples'] == i].copy()
    if temp.shape[0] < 100:
        continue
    adatas.append(temp)
    adatas[-1].obs['global'] = 0

adatas[0].write_h5ad(f"output/Banksy_py/data/hgsc_{adatas[0].obs['samples'][0]}.h5ad")
adatas[2].write_h5ad(f"output/Banksy_py/data/hgsc_{adatas[2].obs['samples'][0]}.h5ad")

# normalize and log transformation
adatas = sf.prep_adatas(adatas, norm=True, log1p=True, scale=False, renorm=False)

# create torch dataset
dataset = sf.make_dataset(adatas, sparse_graph=True, regional_obs=['global'])

# %%
print(*adata.obs['cell.types'].unique())

# %%
immune_types = ['TNK.cell', 'B.cell', 'Monocyte', 'Mast.cell']
fibroblast_types = ['Fibroblast']



# %%
cuda_dataset = None

load_data_into_gpu = True # if you run into OOM on GPU, set this to False
if device == 'cuda' and load_data_into_gpu:
    cuda_dataset = dataset.to('cuda')

# %%
n_heads = 25

sf.set_random_seed(0)
model = sf.Steamboat(adata.var_names.tolist(), n_heads=n_heads, n_scales=3)
model = model.to(device)

use_dataset = cuda_dataset
if use_dataset is None:
    use_dataset = dataset

model.load_state_dict(torch.load('../examples/saved_models/hgsc.pth', weights_only=True), strict=False)

# model.fit(cuda_dataset, entry_masking_rate=0.1, feature_masking_rate=0.1,
#           max_epoch=10000, 
#           loss_fun=torch.nn.MSELoss(reduction='sum'),
#           opt=torch.optim.Adam, opt_args=dict(lr=0.1), stop_eps=1e-3, report_per=200, stop_tol=200, device=device)

# %%
sf.tools.calc_obs(adatas, dataset, model, get_recon=True)

# %%
for i in range(len(adatas)):
    adata = adatas[i]
    if 'steamboat_spatial_domain_colors' in adata.uns:
        adata.uns.pop('steamboat_spatial_domain_colors')
    sf.tools.segment(adata, resolution=0.35, key_added="steamboat_spatial_domain", n_prop=2)
    
    pd.crosstab(adata.obs['steamboat_spatial_domain'], adata.obs['cell.types.nolc']).to_csv(f"./saved_results/hgsc_spatial_domain/hgsc_{adata.obs['samples'][0]}_steamboat_spatial_domain_crosstab.csv")

    adata.obs['steamboat_spatial_domain'].to_csv(f"./saved_results/hgsc_spatial_domain/hgsc_{adata.obs['samples'][0]}_steamboat_spatial_domain.csv")
    sq.pl.spatial_scatter(adata, color=["steamboat_spatial_domain", "cell.types.nolc"], size=.1, shape=None, legend_loc='right margin', frameon=False, figsize=(3, 3))
    plt.savefig(f"./saved_results/hgsc_spatial_domain/hgsc_{adata.obs['samples'][0]}_steamboat_spatial_domain.pdf")

# %%
for i in range(len(adatas)):
    adata = adatas[i]
    sq.pl.spatial_scatter(adata, color=["steamboat_spatial_domain", "cell.types.nolc"], size=.1, shape=None, legend_loc='right margin', frameon=False, figsize=(3, 3), ncols=1)
    plt.savefig(f"./saved_results/hgsc_spatial_domain/hgsc_{adata.obs['samples'][0]}_steamboat_spatial_domain.png", bbox_inches='tight')

# %%
df = pd.crosstab(adatas[2].obs['steamboat_spatial_domain'], adatas[2].obs['cell.types.nolc'])
n = df.loc['0', ['B.cell', 'Mast.cell', 'Monocyte', 'TNK.cell']].sum()
N = n + df.loc['0', 'Malignant']
m = df.loc['1', ['B.cell', 'Mast.cell', 'Monocyte', 'TNK.cell']].sum()
M = m + df.loc['1', 'Malignant']

proportion_test(N, n, M, m)
df

# %%
df = pd.crosstab(adatas[2].obs['steamboat_spatial_domain'], adatas[2].obs['cell.types.nolc'])
n = df.loc['0', ['Fibroblast']].sum()
N = n + df.loc['0', 'Malignant']
m = df.loc['1', ['Fibroblast']].sum()
M = m + df.loc['1', 'Malignant']

proportion_test(N, n, M, m)
df

# %%
import matplotlib.font_manager as fm


vega_20_scanpy = ['#1f77b4', '#ff7f0e', '#279e68', '#d62728', '#aa40fc',
                  '#8c564b', '#e377c2', '#b5bd61', '#17becf', '#aec7e8',
                  '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
                  '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31']

def polished_spatial_scatter(adata, color, palette=None, size=size, ax=None, rasterized=None, frameon=False, 
                             legend_loc='right margin', legend_ncols=1, legend_shift=(0,0), legend_frameon=False, legend_title=None):
    if rasterized is None:
        rasterized = adata.shape[0] > 500

    if palette is None:
        palette = vega_20_scanpy
    if isinstance(palette, list):
        palette = {cat: palette[i % len(palette)] for i, cat in enumerate(adata.obs[color].astype('category').cat.categories)}
    
    for cat in adata.obs[color].astype('category').cat.categories:
        subset = adata[adata.obs[color] == cat]
        x = subset.obsm['spatial'][:, 0]
        y = subset.obsm['spatial'][:, 1]
        x = x
        y = -y
        ax.scatter(x, y, c=palette[cat], 
                   s=size, cmap='tab20', alpha=0.8, rasterized=rasterized, label=cat, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    if not frameon:
        for pos in ['right', 'top', 'left', 'bottom']:
            ax.spines[pos].set_visible(False)
    ax.set_aspect('equal')

    if legend_loc != 'off':
        if legend_loc == 'right margin':
            loc = 'center left'
            bbox_to_anchor = (1 + legend_shift[0], 0.5 + legend_shift[1])
        elif legend_loc == 'bottom margin':
            loc = 'upper center'
            bbox_to_anchor = (0.5 + legend_shift[0], -0.1 + legend_shift[1])
        
        bold_font = fm.FontProperties(size=9, weight='semibold')
        legend = ax.legend(*ax.get_legend_handles_labels(), 
                           fontsize=9, 
                           title=legend_title, title_fontproperties=bold_font,
                        loc=loc, bbox_to_anchor=bbox_to_anchor, 
                        ncol=legend_ncols, columnspacing=0.5,
                        frameon=legend_frameon, handletextpad=0.1, labelspacing=0.3)
        legend.get_frame().set_linewidth(0.5)
        for handle in legend.legend_handles:
            handle.set_sizes([25])  # Adjust dot size in legend


# %%
temp_df = pd.read_csv("../../Banksy_py/hgsc_output/SMI_T10_F001.csv", index_col=0)
adatas[0].obs['banksy'] = temp_df['banksy'].map(['0', '2', '1'].__getitem__).astype('category')

temp_df = pd.read_csv("../../Banksy_py/hgsc_output/SMI_T10_F006.csv", index_col=0)
adatas[2].obs['banksy'] = temp_df['banksy'].map(['1', '0', '2'].__getitem__).astype('category')

# %%
temp_df = pd.read_csv("./saved_results/hgsc_SMI_T10_F001_stagate_spatial_domain.csv", index_col=0)
adatas[0].obs['STAGATE'] = temp_df['stagate'].map({0: '1', 1: '0', 2: '2'}.__getitem__).astype('category')

temp_df = pd.read_csv("./saved_results/hgsc_SMI_T10_F006_stagate_spatial_domain.csv", index_col=0)
adatas[2].obs['STAGATE'] = temp_df['stagate'].map({0: '1', 1: '2', 2: '0'}.__getitem__).astype('category')


# %%
adatas[0].obs['SEDR']

# %%
temp_df = pd.read_csv("./saved_results/hgsc_SMI_T10_F001_sedr_spatial_domain.csv", index_col=0)
adatas[0].obs['SEDR'] = temp_df['hgsc_SMI_T10_F001_sedr_spatial_domain'].map(['', '0', '1', '2'].__getitem__).astype('category')

temp_df = pd.read_csv("./saved_results/hgsc_SMI_T10_F006_sedr_spatial_domain.csv", index_col=0)
adatas[2].obs['SEDR'] = temp_df['hgsc_SMI_T10_F006_sedr_spatial_domain'].map(['', '1', '2', '0'].__getitem__).astype('category')

# %%
from matplotlib.gridspec import GridSpec
import matplotlib


domain_palette = [vega_20_scanpy[7], vega_20_scanpy[8], vega_20_scanpy[18], vega_20_scanpy[19]]

fig = plt.figure(figsize=(6.5, 2.2))
gs = GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 1], hspace=.2, wspace=.01, figure=fig)
axes = [fig.add_subplot(gs[0, i]) for i in range(5)] + [fig.add_subplot(gs[1, i]) for i in range(5)]

size = .35

################
ax = axes[0]
polished_spatial_scatter(adatas[0], color="cell.types.nolc", size=size, legend_loc='off', frameon=False, ax=ax)
ax.set_title('Cell type', size=10, pad=3)
ax.set_ylabel('T10_F001')

################
ax = axes[1]
polished_spatial_scatter(adatas[0], color="steamboat_spatial_domain", size=size, palette=domain_palette,
                         legend_loc='off', frameon=False, ax=ax)
ax.set_title('Steamboat', size=10, pad=3)

################
ax = axes[2]
polished_spatial_scatter(adatas[0], color="banksy", size=size, palette=domain_palette,
                         legend_loc='off', frameon=False, ax=ax)
ax.set_title('BANKSY 0.8', size=10, pad=3)

################
ax = axes[3]
polished_spatial_scatter(adatas[0], color="SEDR", size=size, palette=domain_palette,
                         legend_loc='off', frameon=False, ax=ax)
ax.set_title('SEDR', size=10, pad=3)

################
ax = axes[4]
polished_spatial_scatter(adatas[0], color="STAGATE", size=size, palette=domain_palette,
                         legend_loc='off', frameon=False, ax=ax)
ax.set_title('STAGATE', size=10, pad=3)

################
ax = axes[5]
polished_spatial_scatter(adatas[2], color="cell.types.nolc", size=size, 
                         legend_loc='bottom margin', legend_ncols=4, legend_shift=(1.3, 0), legend_frameon=True, legend_title='Cell types',
                         frameon=False, ax=ax)
ax.set_title('Cell type', size=10, pad=3)
ax.set_ylabel('T10_F006')

################
ax = axes[6]
polished_spatial_scatter(adatas[2], color="steamboat_spatial_domain", size=size, palette=domain_palette,
                         legend_loc='bottom margin', legend_ncols=3, legend_shift=(2.9, 0),  legend_frameon=True, legend_title = 'Spatial domains',
                         frameon=False, ax=ax)
ax.set_title('Steamboat', size=10, pad=3)

################
ax = axes[7]
polished_spatial_scatter(adatas[2], color="banksy", size=size, legend_loc='off', palette=domain_palette,
                         frameon=False, ax=ax)
ax.set_title('BANKSY 0.8', size=10, pad=3)

################

ax = axes[8]
polished_spatial_scatter(adatas[2], color="SEDR", size=size, palette=domain_palette,
                         legend_loc='off', frameon=False, ax=ax)
ax.set_title('SEDR', size=10, pad=3)

################
ax = axes[9]
polished_spatial_scatter(adatas[2], color="STAGATE", size=size, palette=domain_palette,
                         legend_loc='off', frameon=False, ax=ax)
ax.set_title('STAGATE', size=10, pad=3)

#axes[0].text(x = -0.05, y = 1.35, s = 'a', 
#         transform = axes[0].transAxes,
#         fontsize = 18, fontweight = 'bold',
#        va = 'top', ha = 'right'
# )


plt.savefig(f"./figures/hgsc_steamboat_spatial_domain_comparison.pdf", bbox_inches='tight', pad_inches=0.05, dpi=300)

# %%
def proportion_test(N, n, M, m):
    """
    Perform a proportion test between two groups.

    Parameters:
    n (int): Number of successes in group 1
    N (int): Total number of trials in group 1
    m (int): Number of successes in group 2
    M (int): Total number of trials in group 2

    Returns:
    p-value (float): The p-value from the chosen statistical test
    """
    from scipy import stats

    # Step 1: Create contingency table
    immune_counts = [n, m]
    other_counts = [N - n, M - m]
    contingency_table = [immune_counts, other_counts]

    print(f"Contingency Table: {contingency_table}")
    print(f"Proportions: Group 1 = {n}/{N} = {n/N:.4f}, Group 2 = {m}/{M} = {m/M:.4f}")

    # Step 2: Choose your test

    # OPTION A: Chi-Squared Test (Best for typical, large datasets)
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"Chi-Squared p-value: {p_val}")

    # OPTION B: Fisher's Exact Test (Best if any cell count is < 5)
    # Note: This is computationally more intensive but exact.
    odds_ratio, p_val_fisher = stats.fisher_exact(contingency_table)
    print(f"Fisher's Exact p-value: {p_val_fisher}")

    # Step 3: Interpretation
    alpha = 0.05
    if p_val < alpha:
        print("Result: SIGNIFICANT. The proportions are different.")
    else:
        print("Result: NOT SIGNIFICANT. The difference could be due to chance.")

# %%


# %%
df = pd.crosstab(adatas[0].obs['steamboat_spatial_domain'], adatas[0].obs['cell.types.nolc'])
n = df.loc['0', ['B.cell', 'Mast.cell', 'Monocyte', 'TNK.cell']].sum()
N = n + df.loc['0', 'Malignant']
m = df.loc['2', ['B.cell', 'Mast.cell', 'Monocyte', 'TNK.cell']].sum()
M = m + df.loc['2', 'Malignant']

proportion_test(N, n, M, m)
df

# %%
df = pd.crosstab(adatas[0].obs['SEDR'], adatas[0].obs['cell.types.nolc'])
n = df.loc['0', ['B.cell', 'Mast.cell', 'Monocyte', 'TNK.cell']].sum()
N = n + df.loc['0', 'Malignant']
m = df.loc['2', ['B.cell', 'Mast.cell', 'Monocyte', 'TNK.cell']].sum()
M = m + df.loc['2', 'Malignant']

proportion_test(N, n, M, m)
df

# %%
df = pd.crosstab(adatas[0].obs['banksy'], adatas[0].obs['cell.types.nolc'])
n = df.loc['0', ['B.cell', 'Mast.cell', 'Monocyte', 'TNK.cell']].sum()
N = n + df.loc['0', 'Malignant']
m = df.loc['2', ['B.cell', 'Mast.cell', 'Monocyte', 'TNK.cell']].sum()
M = m + df.loc['2', 'Malignant']

proportion_test(N, n, M, m)
df

# %%
adatas[0].obs['STAGATE']

# %%
df = pd.crosstab(adatas[0].obs['STAGATE'], adatas[0].obs['cell.types.nolc'])
n = df.loc['0', ['B.cell', 'Mast.cell', 'Monocyte', 'TNK.cell']].sum()
N = n + df.loc['0', 'Malignant']
m = df.loc['2', ['B.cell', 'Mast.cell', 'Monocyte', 'TNK.cell']].sum()
M = m + df.loc['2', 'Malignant']

proportion_test(N, n, M, m)
df

# %%
for method in ['steamboat_spatial_domain', 'banksy', 'SEDR', 'STAGATE']:
    print(f"Method: {method}")
    df = pd.crosstab(adatas[0].obs[method], adatas[0].obs['cell.types.nolc'])
    n = df.loc['0', ['B.cell', 'Mast.cell', 'Monocyte', 'TNK.cell']].sum()
    N = df.loc['0'].sum()
    m = df.loc['2', ['B.cell', 'Mast.cell', 'Monocyte', 'TNK.cell']].sum()
    M = df.loc['2'].sum()
    proportion_test(N, n, M, m)
    print()

# %%
for method in ['steamboat_spatial_domain', 'banksy', 'SEDR', 'STAGATE']:
    print(f"Method: {method}")
    df = pd.crosstab(adatas[0].obs[method], adatas[0].obs['cell.types.nolc'])
    n = df.loc['1', ['Fibroblast']].sum()
    N = df.loc['1'].sum()
    m = df.loc['2', ['Fibroblast']].sum()
    M = df.loc['2'].sum()
    proportion_test(N, n, M, m)
    print()

# %%
for method in ['steamboat_spatial_domain', 'banksy', 'SEDR', 'STAGATE']:
    print(f"Method: {method}")
    df = pd.crosstab(adatas[2].obs[method], adatas[2].obs['cell.types.nolc'])
    n = df.loc['0', ['Fibroblast']].sum()
    N = df.loc['0'].sum()
    m = df.loc['1', ['Fibroblast']].sum()
    M = df.loc['1'].sum()
    proportion_test(N, n, M, m)
    print()

# %%
for method in ['steamboat_spatial_domain', 'banksy', 'SEDR', 'STAGATE']:
    print(f"Method: {method}")
    df = pd.crosstab(adatas[2].obs[method], adatas[2].obs['cell.types.nolc'])
    n = df.loc['0', ['Endothelial']].sum()
    N = df.loc['0'].sum()
    m = df.loc['2', ['Endothelial']].sum()
    M = df.loc['2'].sum()
    proportion_test(N, n, M, m)
    print()

# %%
df

# %%
attn_by_domain = np.zeros((len(adatas[2].obs['steamboat_spatial_domain'].unique()), n_heads))
for domain in adatas[2].obs['steamboat_spatial_domain'].unique():
    subset = adatas[2][adatas[2].obs['steamboat_spatial_domain'] == domain]
    attn_by_domain[int(domain), :] = subset.obsm['local_attn'].mean(axis=0)

# %%
adatas[2].obs['temp'] = adatas[2].obsm['local_attn'][:, 12]
sq.pl.spatial_scatter(adatas[2], color=["temp"], size=.1, shape=None, legend_loc='right margin', frameon=False, figsize=(3, 2), cmap='Reds', vmin=0.005, vmax=0.025)

# %%
adatas[2].var_names[adatas[2].var_names.str.contains('^F')]

# %%
var.loc[['FGF1', 'FGF']]

# %%
var.sort_values(by='k_local_12', ascending=False)

# %%
var = sf.tools.calc_var(model)
fig, ax = plt.subplots(figsize=(12, 6))
var = var.div(var.max(axis=0), axis=1)
temp = var.loc[adatas[2].var_names[adatas[2].var_names.str.contains('^FGF|TGFB|WNT|FZD')], var.columns.str.contains('^q_|k_local_')]
# temp = temp.div(temp.max(axis=0), axis=1)
sns.heatmap(temp, cmap='Reds', ax=ax)

# %%


# %%
fig, ax = plt.subplots(figsize=(6, 1))
sns.heatmap(attn_by_domain, ax=ax)

# %%
from matplotlib.gridspec import GridSpec
import matplotlib


domain_palette = [vega_20_scanpy[7], vega_20_scanpy[8], vega_20_scanpy[18], vega_20_scanpy[19]]

fig = plt.figure(figsize=(6.5, 3.6))
gs = GridSpec(4, 5, width_ratios=[1, 1, 1, 1, 1], hspace=.2, wspace=.01, figure=fig)
axes = [fig.add_subplot(gs[0, i]) for i in range(5)] + [fig.add_subplot(gs[1, i]) for i in range(5)]
axes += [fig.add_subplot(gs[3, 0])]
size = .35

################
ax = axes[0]
polished_spatial_scatter(adatas[0], color="cell.types.nolc", size=size, legend_loc='off', frameon=False, ax=ax)
ax.set_title('Cell type', size=10, pad=3)
ax.set_ylabel('T10_F001')

################
ax = axes[1]
polished_spatial_scatter(adatas[0], color="steamboat_spatial_domain", size=size, palette=domain_palette,
                         legend_loc='off', frameon=False, ax=ax)
ax.set_title('Steamboat', size=10, pad=3)

################
ax = axes[2]
polished_spatial_scatter(adatas[0], color="banksy", size=size, palette=domain_palette,
                         legend_loc='off', frameon=False, ax=ax)
ax.set_title('BANKSY 0.8', size=10, pad=3)

################
ax = axes[3]
polished_spatial_scatter(adatas[0], color="SEDR", size=size, palette=domain_palette,
                         legend_loc='off', frameon=False, ax=ax)
ax.set_title('SEDR', size=10, pad=3)

################
ax = axes[4]
polished_spatial_scatter(adatas[0], color="STAGATE", size=size, palette=domain_palette,
                         legend_loc='off', frameon=False, ax=ax)
ax.set_title('STAGATE', size=10, pad=3)

################
ax = axes[5]
polished_spatial_scatter(adatas[2], color="cell.types.nolc", size=size, 
                         legend_loc='bottom margin', legend_ncols=4, legend_shift=(1.3, 0), legend_frameon=True, legend_title='Cell types',
                         frameon=False, ax=ax)
ax.set_title('Cell type', size=10, pad=3)
ax.set_ylabel('T10_F006')

################
ax = axes[6]
polished_spatial_scatter(adatas[2], color="steamboat_spatial_domain", size=size, palette=domain_palette,
                         legend_loc='bottom margin', legend_ncols=3, legend_shift=(2.9, 0),  legend_frameon=True, legend_title = 'Spatial domains',
                         frameon=False, ax=ax)
ax.set_title('Steamboat', size=10, pad=3)

################
ax = axes[7]
polished_spatial_scatter(adatas[2], color="banksy", size=size, legend_loc='off', palette=domain_palette,
                         frameon=False, ax=ax)
ax.set_title('BANKSY 0.8', size=10, pad=3)

################

ax = axes[8]
polished_spatial_scatter(adatas[2], color="SEDR", size=size, palette=domain_palette,
                         legend_loc='off', frameon=False, ax=ax)
ax.set_title('SEDR', size=10, pad=3)

################
ax = axes[9]
polished_spatial_scatter(adatas[2], color="STAGATE", size=size, palette=domain_palette,
                         legend_loc='off', frameon=False, ax=ax)
ax.set_title('STAGATE', size=10, pad=3)

axes[0].text(x = -0.05, y = 1.35, s = 'a', 
         transform = axes[0].transAxes,
         fontsize = 18, fontweight = 'bold',
        va = 'top', ha = 'right'
)

plt.savefig(f"output/hgsc_steamboat_spatial_domain_comparison.png", bbox_inches='tight', pad_inches=0.05, dpi=300)


