import os
import scanpy as sc
import squidpy as sq
import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.decomposition import NMF
import sklearn.metrics
from tqdm.auto import tqdm
from matplotlib.gridspec import GridSpec
import itertools
import pickle as pkl

# specify the path to the root directory of the repository if not installed as a package
# if you seen ModuleNotFoundError: No module named 'steamboat',
# this is likely because you are running from a different working directory. 
# You can change the path below to the absolute root directory of the repository
import sys
sys.path.append("../..") 


import steamboat as sf
import steamboat.tools
import torch
device = 'cuda'

pltkw = dict(bbox_inches='tight', transparent=True, dpi=400)

# number of parallel jobs for clustering evaluation. You can change this number based on your CPU cores.
n_jobs = 20 

##################################################################################
# Load data                                                                      #
##################################################################################

regenerate = False

# Data curated from https://www.nature.com/articles/s41590-024-01943-5
# You can download the processed data from:
# https://drive.google.com/drive/folders/1PbLOhYRXp1TKVfPNPWiO4-F3ucsc4u8T?usp=sharing
data_path = "../data/Ex1_hgsc/"
h5ad_file = data_path + "ST_Discovery_so.h5ad"
adata = sc.read_h5ad(h5ad_file)

# If you prefer to start from the raw data, you can use the following code to generate the h5ad files
# data_path = "PATH/TO/DATA/"
# adata = sc.read_mtx(data_path + "ST_Discovery_so_counts.mtx").T
# metadata = pd.read_csv(data_path + "ST_Discovery_so_metadata.csv", index_col=0)
# features = pd.read_csv(data_path + "ST_Discovery_so_features.txt", index_col=0, header=None)
# features.index = features.index.str.strip() # remove trailing white space in gene names
# features.index.name = 'gene_symbol'
# adata.obs = metadata
# adata.var = features
# adata.obsm['spatial'] = adata.obs[['x', 'y']].to_numpy()
# adata.write_h5ad("../data/Ex1_hgsc/ST_Discovery_so.h5ad")

sample_metadata = pd.read_excel(data_path + "sample_metadata.xlsx", index_col=0, sheet_name='Table 2b', skiprows=1)
sample_metadata = sample_metadata[sample_metadata['dataset'] == 'Discovery']

celltype_signatures = pd.read_excel(data_path + "sample_metadata.xlsx", sheet_name='Table 3a', skiprows=2)
mtil_signautures = pd.read_excel(data_path + "sample_metadata.xlsx", sheet_name='Table 6a', skiprows=2)
desmoplasia_signautures = pd.read_excel(data_path + "sample_metadata.xlsx", sheet_name='Table 5a', skiprows=2)

def purge_gene_sets(df, prefix=''):
    res = {}
    for i in df.columns:
        res[prefix + i] = df[i].dropna().tolist()
    return res
celltype_signatures = pd.read_excel(data_path + "sample_metadata.xlsx", index_col=0, sheet_name='Table 3b', skiprows=2).iloc[:, :-3]
genesets = (purge_gene_sets(celltype_signatures, 'sig_') | 
            purge_gene_sets(mtil_signautures, 'mtil_') | 
            purge_gene_sets(mtil_signautures, 'mtil_'))
genesets.keys()
del genesets['sig_Mast.cell']

mask = (sample_metadata['sites_binary'] == 'Adnexa') & (sample_metadata['treatment'] == 'Untreated')
samples_of_interest = sample_metadata.index[mask].tolist()

all_adata = adata[adata.obs['samples'].isin(samples_of_interest)].copy()
all_adata.obs['cell.types.nolc'] = all_adata.obs['cell.types'].str.replace('_LC', '')

##################################################################################
# Prepare torch dataset                                                          
##################################################################################

# Separate individual slides
adatas = []
for i in all_adata.obs['samples'].unique():
    temp = all_adata[all_adata.obs['samples'] == i].copy()
    if temp.shape[0] < 100:
        continue
    adatas.append(temp)
    adatas[-1].obs['global'] = 0

# normalize and log transformation
adatas = sf.prep_adatas(adatas, norm=True, log1p=True, scale=False, renorm=False)

# create torch dataset
dataset = sf.make_dataset(adatas, sparse_graph=True, regional_obs=['global'])

# Preload the whole dataset into GPU memory to speed up training. 
# This is optional and can be disabled if you run into out-of-memory (OOM) issues on the GPU.
load_data_into_gpu = True 
if device == 'cuda' and load_data_into_gpu:
    cuda_dataset = dataset.to('cuda')

######################################################################################
#     Load the model                                                                  
######################################################################################

n_heads = 25

sf.set_random_seed(0)
model = sf.Steamboat(adata.var_names.tolist(), n_heads=n_heads, n_scales=3)
model = model.to(device)

cuda_dataset = None

use_dataset = cuda_dataset
if use_dataset is None:
    use_dataset = dataset

model.load_state_dict(torch.load('../data/saved_models/hgsc.pth', weights_only=True))

sf.tools.calc_obs(adatas, dataset, model, get_recon=True)
sf.tools.gather_obs(all_adata, adatas)

os.makedirs('output', exist_ok=True)

if False:
    ######################################################################################
    #     Figure 3a: Spatial distribution of cell types                                   
    #     Also: Figure S1a                                                               
    ######################################################################################
    ax = sq.pl.spatial_scatter(adatas[0], color=['cell.types.nolc'], shape=None, figsize=(3.5, 3.5), size=.5, 
                        legend_fontsize=10, cmap='Reds', ncols=3, colorbar=False, vmin=0., wspace=.0, outline=False, frameon=False, title="", return_ax=True)
    ax.figure.savefig('output/fig3a_spatial_celltypes.png', **pltkw)

    ######################################################################################
    #     Figure 3b: UMAP
    ######################################################################################
    adata = all_adata[~all_adata.obs['cell.types'].str.contains('_LC')]
    adata.obsm['std_attn'] = adata.obsm['attn'] / adata.obsm['attn'].std(axis=0, keepdims=True)
    sc.pp.neighbors(adata, use_rep='std_attn', key_added='sf', metric='cosine')
    sc.tl.umap(adata, neighbors_key='sf')
    fig = sc.pl.umap(adata, color=['cell.types', 'samples'], frameon=False, show=False, return_fig=True)
    fig.savefig('output/fig3b_umap_celltypes.png', **pltkw)

    ######################################################################################
    #     Figure 3c: Head weights & gene programs
    #     Also: Figure S1c
    ######################################################################################
    head_weights = sf.tools.calc_head_weights(adatas, model)
    sf.tools.plot_head_weights(head_weights, save='output/fig3c_head_weights.png')

    gene_df = sf.tools.calc_var(model)
    sig_df = sf.tools.calc_geneset_auroc(gene_df, genesets)
    metagene_order = sf.tools.calc_geneset_auroc_order(sig_df)
    sf.tools.plot_geneset_auroc(sig_df, metagene_order, save='output/fig3c_metagene_auroc.png')

    ######################################################################################
    #     Figure 3d,e: Attention maps
    #     Also: Figure S1e
    ######################################################################################
    for i_head in [0, 5, 15]:
        adatas[0].obs[f'q_{i_head}'] = adatas[0].obsm['q'][:, i_head]
        adatas[0].obs[f'local_k_{i_head}'] = adatas[0].obsm['local_k'][:, i_head]
        adatas[0].obs[f'global_k_0_{i_head}'] = adatas[0].obsm['global_k_0'][:, i_head]
        ax = sq.pl.spatial_scatter(adatas[0], color=[f'q_{i_head}', f'local_k_{i_head}', f'global_k_0_{i_head}'], shape=None, figsize=(2, 2), size=.5, 
                            legend_fontsize=10, cmap='Reds', ncols=3, colorbar=False, vmin=0., wspace=.0, outline=False, frameon=False, title="", return_ax=True)
        ax[0].figure.savefig(f'output/fig3d_attention_head_{i_head}.png', **pltkw)

    def plot_helper(all_adata, i_head, celltypes):
        plt_df = all_adata.obs.copy()
        plt_df[f'q_{i_head}'] = all_adata.obsm['q'][:, i_head]
        plt_df = plt_df[plt_df['cell.types.nolc'].isin(celltypes)]
        plt_df = pd.pivot_table(plt_df, index=["samples"], columns=["cell.types.nolc"], values=[f"q_{i_head}"], aggfunc=np.median)
        plt_df = plt_df.sort_values(by=(f"q_{i_head}", celltypes[0]))
        
        fig, ax = plt.subplots(figsize=(2, 0.75))
        sns.heatmap((plt_df / plt_df.max()).T, ax=ax, cmap='Reds')
        ax.set_xticks([])
        ax.set_ylabel('')
        ax.set_xlabel('Samples')
        return fig, ax

    for head in [0, 5]:
        fig, _ = plot_helper(all_adata, head, ['Malignant', 'Monocyte'])
        fig.savefig(f'output/fig3e_attention_head_{head}.png', **pltkw)

######################################################################################
#     Figure 3f,g: CCI
######################################################################################
steamboat_cci = sf.tools.calc_interaction(adatas, model, 'samples', 'cell.types.nolc')
adjacency_cci = sf.tools.calc_adjacency_freq(adatas, 'samples', 'cell.types.nolc')
def melt_helper(x):
    x_melt = x.melt(ignore_index=False)
    x_melt['variable'] = x_melt.index + '_' + x_melt['variable']
    return x_melt

cellchat_corr_dict = {}
cellchat_melt_dict = {}

cellchat_dict = {}

for k, v in steamboat_cci.items():
    cellchat_res_path = f"../data/Ex1_hgsc/cellchat/{k}.csv"
    if os.path.isfile(cellchat_res_path):
        # print(cellchat_res_path)
        cellchat = pd.read_csv(cellchat_res_path, index_col=0)
        if (cellchat == 0).all().all():
            print(cellchat_res_path, 'all zero. Ignore.')
            continue
            
        cellchat_dict[k] = cellchat
        mine = v.copy()
        cellchat_melt = melt_helper(cellchat + cellchat.T)
        mine_melt = melt_helper(mine)
        
        adjacency_melt = melt_helper(adjacency_cci[k])
        adjacency_melt.columns = ['variable', 'value_adjacency']
        
        all_melt = pd.merge(cellchat_melt, mine_melt, on='variable', suffixes=['_cellchat', '_steamboat'])
        all_melt = pd.merge(all_melt, adjacency_melt, on='variable')
        
        corr_res = sp.stats.spearmanr(all_melt['value_cellchat'], all_melt['value_steamboat'])
        cellchat_melt_dict[k] = all_melt
        cellchat_corr_dict[k] = [corr_res.statistic, corr_res.pvalue]
        
        corr_res = sp.stats.spearmanr(all_melt['value_cellchat'], all_melt['value_adjacency'])
        cellchat_corr_dict[k].extend([corr_res.statistic, corr_res.pvalue])
    else:
        print(cellchat_res_path, 'not found!')
        if os.path.isfile('../data/Ex1_hgsc/cellchat.zip'):
            print('Please unzip the cellchat.zip file to the cellchat folder.')


cellchat_corr_df = pd.DataFrame(cellchat_corr_dict, index=['steamboat_r', 'steamboat_p', 
                                                           'adjacency_r', 'adjacency_p'])

cellchat_corr_df = cellchat_corr_df.T
g = sns.jointplot(
    data=cellchat_corr_df,
    x="adjacency_r", y="steamboat_r",
    kind="scatter",
    height=2,
    xlim=[0.1, .9],
    ylim=[0.1, .9]
)

ax = g.ax_joint

ax.set_xlabel('Neighboring cell types')
ax.set_ylabel('Steamboat')
ax.plot([.0, 1.], [.0, 1.], ls='--', lw=1., c='k')
g.figure.savefig(f"output/fig3g_steamboat_vs_adjacency.png", **pltkw)

test_res = sp.stats.wilcoxon(cellchat_corr_df['adjacency_r'], cellchat_corr_df['steamboat_r'])

ax.text(0.3, 0.15, f'p = {test_res.pvalue:.1e}')
(cellchat_corr_df['adjacency_r'] < cellchat_corr_df['steamboat_r']).sum() / cellchat_corr_df.shape[0]

sample = 'SMI_T10_F001'
fig, ax = plt.subplots(figsize=(1., 1.))
cellchat_melt_dict[sample].plot(kind='scatter', x='value_cellchat', y='value_steamboat', ax=ax, title=sample, s=5.)

ax.set_xlabel('CellChat')
ax.set_ylabel('Steamboat')
for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)
fig.savefig(f"output/fig3f_steamboat.png", **pltkw)

fig, ax = plt.subplots(figsize=(1., 1.))
cellchat_melt_dict[sample].plot(kind='scatter', x='value_cellchat', y='value_adjacency', ax=ax, title=sample, s=5.)

ax.set_xlabel('CellChat')
ax.set_ylabel('Baseline\n(adjacency)')
for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)
fig.savefig(f"output/fig3g_adjacency.png", **pltkw)

######################################################################################
#     Figure S1f,g: CCI for all methods
######################################################################################

steamboat_cci = sf.tools.calc_interaction(adatas, model, 'samples', 'cell.types.nolc')
adjacency_cci = sf.tools.calc_adjacency_freq(adatas, 'samples', 'cell.types.nolc')

def melt_helper(x, symmetricize=True):
    if symmetricize:
        x = x + x.T
    x_melt = x.melt(ignore_index=False)
    x_melt['variable'] = x_melt.index + '_' + x_melt['variable']
    return x_melt

def calc_corr(df1, df2):
    merged = pd.merge(df1, df2, on='variable', suffixes=('_1', '_2'))
    merged = merged[(merged['value_1'] > 0) & (merged['value_2'] > 0)]
    if merged.shape[0] < 10:
        return np.nan, np.nan
    r, p = sp.stats.spearmanr(merged['value_1'], merged['value_2'])
    return r, p

methods = ['STEAMBOAT', 'CellChat', 'Adjacency',  'HistoCAT-style', 'CellPhoneDB', 'COMMOT', 'MISTy',]
method_pairs = list(itertools.combinations(methods, 2))

# %%
# Read all matrices

cci_matrices = {}

cci_matrices['STEAMBOAT'] = steamboat_cci

cci_matrices['CellChat'] = {}
for k in samples_of_interest:
    cellchat_res_path = f"../data/Ex1_hgsc/cellchat/{k}.csv"
    if os.path.isfile(cellchat_res_path):
        # print(cellchat_res_path)
        cci_matrices['CellChat'][k] = pd.read_csv(cellchat_res_path, index_col=0)
    else:
        cci_matrices['CellChat'][k] = pd.DataFrame()

cci_matrices['Adjacency'] = adjacency_cci

cci_matrices['COMMOT'] = {}
for k in samples_of_interest:
    commot_res_path = f"../data/Ex1_hgsc/cmp/commot/hgsc_commot_cci_{k}.csv"
    if os.path.isfile(commot_res_path):
        # print(commot_res_path)
        cci_matrices['COMMOT'][k] = pd.read_csv(commot_res_path, index_col=0)
    else:
        cci_matrices['COMMOT'][k] = pd.DataFrame()

# %%
cci_matrices['MISTy'] = {}
for k in samples_of_interest:
    misty_res_path = f"../data/Ex1_hgsc/cmp/misty/misty_cci_hgsc_{k}.csv"
    if os.path.isfile(misty_res_path):
        # print(misty_res_path)
        temp = pd.read_csv(misty_res_path, index_col=0)
        cci_matrices['MISTy'][k] = pd.DataFrame(temp, index=temp['Predictor'].unique(), columns=temp['Predictor'].unique())
        for i in cci_matrices['MISTy'][k].index:
            for j in cci_matrices['MISTy'][k].columns:
                cci_matrices['MISTy'][k].loc[i, j] = temp[(temp['Predictor'] == i) & (temp['Target'] == j)]['Importance'].values[0]
    else:
        cci_matrices['MISTy'][k] = pd.DataFrame()

# %%

def unmelt(df):
    celltypes = {}
    for i in df.index:
        a, b = i.split('|')
        celltypes[a] = None
        celltypes[b] = None
    
    celltypes = list(celltypes.keys())
    df2 = pd.DataFrame(index=celltypes, columns=celltypes)
    for i in df2.index:
        for j in df2.columns:
            df2.loc[i, j] = cpdb_summary[i + '|' + j]
    return df2

cci_matrices['CellPhoneDB'] = {}
ks = [adata.obs['samples'].astype(str).unique().item() for adata in adatas]
for i, k in enumerate(ks):
    cellphonedb_res_path = f"../data/Ex1_hgsc/cmp/cellphonedb/hgsc_{i}"
    with open(cellphonedb_res_path, 'rb') as f:
        cpdb_results = pkl.load(f)    
        cpdb_summary = cpdb_results['means_result'].loc[:, cpdb_results['means_result'].columns.str.contains('\|')].sum(axis=0)
        cci_matrices['CellPhoneDB'][k] = unmelt(cpdb_summary)
        

# %%
cci_matrices['HistoCAT-style'] = {}
for k in samples_of_interest:
    histocat_res_path = f"../data/Ex1_hgsc/cmp/histocat-style/hgsc_histocat_results_{k}.pkl"
    if os.path.isfile(histocat_res_path):
        # print(histocat_res_path)
        real_counts, p_interaction, p_avoidance = pkl.load(open(histocat_res_path, "rb"))
        p_interaction.index.name = None
        p_interaction.columns.name = None
        p_avoidance.index.name = None
        p_avoidance.columns.name = None
        temp = np.log(p_avoidance) - np.log(p_interaction)
        cci_matrices['HistoCAT-style'][k] = temp
    else:
        print("", histocat_res_path, "not found!")
        cci_matrices['HistoCAT-style'][k] = pd.DataFrame()

# %%
# Get pairwise correlation between methods
r_matrix = pd.DataFrame(index=samples_of_interest, columns=method_pairs)

for k in r_matrix.index:
    for m1, m2 in method_pairs:
        if cci_matrices[m1][k].empty or cci_matrices[m2][k].empty:
            r_matrix.loc[k, (m1, m2)] = np.nan
            continue
        df1 = melt_helper(cci_matrices[m1][k])
        df2 = melt_helper(cci_matrices[m2][k])
        r, p = calc_corr(df1, df2)
        r_matrix.loc[k, ((m1, m2), )] = r

r_matrix2 = r_matrix.copy()
r_matrix2.columns = [' & '.join(col) for col in r_matrix2.columns]

quantile_r_matrix = pd.DataFrame(index=methods, columns=methods)
for m1, m2 in method_pairs:
    quantile_r_matrix.loc[m1, m2] = r_matrix[(m1, m2)].median()
    quantile_r_matrix.loc[m2, m1] = r_matrix[(m1, m2)].median()

fig = plt.figure(figsize=(6.5, 3.75))
gs = GridSpec(2, 2, height_ratios=[3, 1.75], width_ratios=[1.2, 1], hspace=0.05, wspace=0.35, figure=fig)
axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]


########################################
ax = axes[0]
sns.heatmap(quantile_r_matrix.astype(float), annot=True, annot_kws={"size": 8}, fmt='.2f', 
            lw=.5, vmin=-.75, vmax=.75, cmap='vlag', square=True, ax=ax, cbar_kws={"shrink": 0.3})
ax.set_xlim(0, len(methods))
ax.set_ylim(len(methods), 0)

for pos in ['right', 'top']:
    ax.spines[pos].set_visible(False)

for pos in ['left', 'bottom']:
    ax.spines[pos].set_visible(False)

# ax.set_title('Median of correlations\n', size=12)
ax.set_anchor('N')

########################################

ax = axes[1]
# ax.set_title('Distribution of correlations', size=12)
sns.boxplot(data=r_matrix2, ax=ax, fliersize=1.)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
ax.set_xticklabels([])

ax.grid(axis='y')
ax.set_ylabel('Spearman Correlation')

for pos in ['right', 'top']:
    ax.spines[pos].set_visible(False)
ax.set_anchor('N')

ax = axes[2]
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods[::-1])
ax.set_ylim([-0.5, len(methods)-0.5])

method2idx = {m: len(methods) - i - 1 for i, m in enumerate(methods)}

for i, m in enumerate(r_matrix2.columns):
    a, b = m.split(' & ')
    ax.plot([i, i], [method2idx[a], method2idx[b]], color='gray', zorder=100)
    ax.scatter([i, i], [method2idx[a], method2idx[b]], color='gray', s=10, zorder=100)

for pos in ['right', 'top', 'left', 'bottom']:
    ax.spines[pos].set_visible(False)

ax.grid(axis='y', zorder=0)

ax.tick_params(axis=u'both', which=u'both',length=0)
ax.set_xticklabels([])


###########################################

axes[0].text(x = -0.53, y = 1.05, s = 'a', 
        transform = axes[0].transAxes,
        fontsize = 18, fontweight = 'bold',
        va = 'top', ha = 'right'
)

axes[1].text(x = -0.25, y = 1.05, s = 'b', 
        transform = axes[1].transAxes,
        fontsize = 18, fontweight = 'bold',
        va = 'top', ha = 'right'
)

fig.savefig('output/figS1fg_cci_summary.png', **pltkw)


######################################################################################
#     Figure S1b: Clustering
######################################################################################
temp_file_names = []
try:
    os.makedirs('temp_adata', exist_ok=False)
except FileExistsError:
    print("temp_adata already exists in the working directory. If it's from an interrupted run, please delete it first.")
for i, adata in enumerate(adatas):
    adata.write_h5ad(f"temp_adata/hgsc_{i}.h5ad")
    temp_file_names.append(f"temp_adata/hgsc_{i}.h5ad")

# PCA

aris = []
nmis = []

def func(i):
    name = 'pca_cell_type_test'
    adata = sc.read_h5ad(f"temp_adata/hgsc_{i}.h5ad")
    # sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=25)
    sc.pp.neighbors(adata)

    sc.tl.leiden(adata, key_added=name, resolution=0.6)
    # If you want to save the clustering results for each slide, uncomment the following line
    # adata.obs[[name]].to_csv(f"temp_adata/{name}_{i}.csv")
    # temp_file_names.append(f"temp_adata/{name}_{i}.csv")

    ari = sklearn.metrics.adjusted_rand_score(adata.obs['cell.types.nolc'], adata.obs[name])
    nmi = sklearn.metrics.normalized_mutual_info_score(adata.obs['cell.types.nolc'], adata.obs[name])
    
    return adata.obs['cell.types.nolc'].nunique(), adata.obs[name].nunique(), ari, nmi

res = Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in tqdm(range(27), total=27))
for class_nunique, nunique, ari, nmi in res:
    print(class_nunique, nunique, ari, nmi)
    aris.append(ari)
    nmis.append(nmi)

df = pd.DataFrame({'ARI': aris, 'NMI': nmis})
df.to_csv(f"temp_adata/pca_cell_type_test.csv")
temp_file_names.append(f"temp_adata/pca_cell_type_test.csv")

# NMF
aris = []
nmis = []
name = 'nmf_cell_type_test'
def func(i):
    name = 'nmf_cell_type_test'
    adata = sc.read_h5ad(f"temp_adata/hgsc_{i}.h5ad")
    adata.obsm['X_nmf'] = NMF(n_components=25, init='random', random_state=0).fit_transform(adata.X)
    sc.pp.neighbors(adata, use_rep='X_nmf', key_added='sf', metric='cosine')
    sc.tl.leiden(adata, resolution=0.5, key_added=name, neighbors_key='sf')

    # If you want to save the clustering results for each slide, uncomment the following line
    # adata.obs[[name]].to_csv(f"temp_adata/{name}_{i}.csv")
    # temp_file_names.append(f"temp_adata/{name}_{i}.csv")

    ari = sklearn.metrics.adjusted_rand_score(adata.obs['cell.types.nolc'], adata.obs[name])
    nmi = sklearn.metrics.normalized_mutual_info_score(adata.obs['cell.types.nolc'], adata.obs[name])
    
    return adata.obs['cell.types.nolc'].nunique(), adata.obs[name].nunique(), ari, nmi

res = Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in tqdm(range(27), total=27))
for class_nunique, nunique, ari, nmi in res:
    print(class_nunique, nunique, ari, nmi)
    aris.append(ari)
    nmis.append(nmi)

df = pd.DataFrame({'ARI': aris, 'NMI': nmis})
df.to_csv(f"temp_adata/{name}.csv")
temp_file_names.append(f"temp_adata/{name}.csv")

# Steamboat
aris = []
nmis = []
name = 'steamboat_cell_type_test'
def func(i):
    name = 'steamboat_cell_type_test'
    adata = sc.read_h5ad(f"temp_adata/hgsc_{i}.h5ad")

    adata.obsm['std_attn'] = adata.obsm['attn'] / adata.obsm['attn'].std(axis=0, keepdims=True)
    sc.pp.neighbors(adata, use_rep='std_attn', key_added='sf', metric='cosine')
    sc.tl.leiden(adata, resolution=0.55, key_added=name, neighbors_key='sf')

    # If you want to save the clustering results for each slide, uncomment the following line
    # adata.obs[[name]].to_csv(f"temp_adata/{name}_{i}.csv")
    # temp_file_names.append(f"temp_adata/{name}_{i}.csv")

    ari = sklearn.metrics.adjusted_rand_score(adata.obs['cell.types.nolc'], adata.obs[name])
    nmi = sklearn.metrics.normalized_mutual_info_score(adata.obs['cell.types.nolc'], adata.obs[name])
    
    return adata.obs['cell.types.nolc'].nunique(), adata.obs[name].nunique(), ari, nmi

res = Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in tqdm(range(27), total=27))
for class_nunique, nunique, ari, nmi in res:
    print(class_nunique, nunique, ari, nmi)
    aris.append(ari)
    nmis.append(nmi)

df = pd.DataFrame({'ARI': aris, 'NMI': nmis})
df.to_csv(f"temp_adata/{name}.csv")
temp_file_names.append(f"temp_adata/{name}.csv")

# Summarize the results
dfs = {}
dfs['Steamboat'] = pd.read_csv("temp_adata/steamboat_cell_type_test.csv", index_col=0)
dfs['PCA'] = pd.read_csv("temp_adata/pca_cell_type_test.csv", index_col=0)
dfs['NMF'] = pd.read_csv("temp_adata/nmf_cell_type_test.csv", index_col=0)

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

# Plotting the results
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
fig.savefig("output/figs1b_clustering.png", **pltkw)

# Cleaning
# if you want to keep the temp_adata folder for further analysis, comment out the following lines.
for file_name in temp_file_names:
    os.remove(file_name)
os.rmdir('temp_adata')
