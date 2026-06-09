# %% [markdown]
# # Ovarian cancer data analysis
# Train steamboat model on HGSC data.

# %%
import os
import scanpy as sc
import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

# %%
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'arial'

pltkw = dict(bbox_inches='tight', transparent=True)

# %% [markdown]
# ## Load data

# %%
regenerate = False

h5ad_file = "../../../data/HGSC/ST_Discovery_so.h5ad"
if (not os.path.exists(h5ad_file)) or regenerate:
    adata = sc.read_mtx("G:/data/HGSC/Csv/ST_Discovery_so_counts.mtx").T
    metadata = pd.read_csv("G:/data/HGSC/Csv/ST_Discovery_so_metadata.csv", index_col=0)
    features = pd.read_csv("G:/data/HGSC/Csv/ST_Discovery_so_features.txt", index_col=0, header=None)
    features.index = features.index.str.strip() # remove trailing white space in gene names
    features.index.name = 'gene_symbol'
    adata.obs = metadata
    adata.var = features
    adata.obsm['spatial'] = adata.obs[['x', 'y']].to_numpy()
    adata.write_h5ad("G:/data/HGSC/h5ad/ST_Discovery_so.h5ad")
else:
    adata = sc.read_h5ad(h5ad_file)

# %%
adata.obs

# %%
## Metadata and gene sets

sample_metadata = pd.read_excel("../../../data/HGSC/sample_metadata.xlsx", index_col=0, sheet_name='Table 2b', skiprows=1)
sample_metadata = sample_metadata[sample_metadata['dataset'] == 'Discovery']

celltype_signatures = pd.read_excel("G:/data/HGSC/sample_metadata.xlsx", sheet_name='Table 3a', skiprows=2)
mtil_signautures = pd.read_excel("G:/data/HGSC/sample_metadata.xlsx", sheet_name='Table 6a', skiprows=2)
desmoplasia_signautures = pd.read_excel("G:/data/HGSC/sample_metadata.xlsx", sheet_name='Table 5a', skiprows=2)

def purge_gene_sets(df, prefix=''):
    res = {}
    for i in df.columns:
        res[prefix + i] = df[i].dropna().tolist()
    return res
celltype_signatures = pd.read_excel("G:/data/HGSC/sample_metadata.xlsx", index_col=0, sheet_name='Table 3b', skiprows=2).iloc[:, :-3]
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
# Separate individual slides
adatas = []
for i in all_adata.obs['samples'].unique():
    temp = all_adata[all_adata.obs['samples'] == i].copy()
    if temp.shape[0] < 100:
        continue
    sc.pp.normalize_total(temp)
    sc.pp.log1p(temp)
    adatas.append(temp)
    adatas[-1].obs['global'] = 0

# %%
### Have a look at the spatial distribution of cells

# %% [markdown]
# ## Attention map interpretation: Cell-cell interaction

# %%
from cellphonedb.src.core.methods import cpdb_analysis_method
import pickle as pkl
from tqdm.notebook import tqdm

# %%
for i_sample in tqdm(range(len(adatas))):
    abundant_cell_types = (adatas[i_sample].obs['cell.types.nolc'].value_counts() >= 10).where(lambda x: x).dropna().index.tolist()
    adatas[i_sample] = adatas[i_sample][adatas[i_sample].obs['cell.types.nolc'].isin(abundant_cell_types)]

    df_meta = pd.DataFrame(data={'Cell':list(adatas[i_sample].obs.index),
                                 'cell.types.nolc':[ i for i in adatas[i_sample].obs['cell.types.nolc']]
                                })
    df_meta.set_index('Cell', inplace=True)
    df_meta.to_csv('cellphonedb-res/hgsc_meta.tsv', sep = '\t')
    df_meta

    # deg_dfs = []
    # sc.tl.rank_genes_groups(adatas[i_sample], groupby='cell.types', method='wilcoxon')
    # for celltype in df_meta['cell.types.nolc'].unique():
    #     deg_df = sc.get.rank_genes_groups_df(adatas[0], group="Malignant")
    #     deg_df = deg_df[(deg_df['logfoldchanges'] > 0.1) & (deg_df['pvals'] < 0.05)]
    #     deg_df['cluster'] = celltype
    #     deg_dfs.append(deg_df)

    # deg_df = pd.concat(deg_dfs)
    # deg_df[['cluster', 'names', 'scores']].to_csv('cellphonedb-res/hgsc_DEGs.tsv', index=False, sep='\t')

    cpdb_results = cpdb_analysis_method.call(
        cpdb_file_path = 'cellphonedb-res/db/v5.0.0/cellphonedb.zip',           # mandatory: CellphoneDB database zip file.
        meta_file_path = 'cellphonedb-res/hgsc_meta.tsv',           # mandatory: tsv file defining barcodes to cell label.
        counts_file_path = adatas[i_sample],       # mandatory: normalized count matrix - a path to the counts file, or an in-memory AnnData object
        counts_data = 'hgnc_symbol',               # defines the gene annotation in counts matrix.
        microenvs_file_path = None, # optional (default: None): defines cells per microenvironment.
        score_interactions = True,                 # optional: whether to score interactions or not. 
        output_path = 'cellphonedb-res/',                    # Path to save results    microenvs_file_path = None,
        separator = '|',                           # Sets the string to employ to separate cells in the results dataframes "cellA|CellB".
        threads = 5,                               # number of threads to use in the analysis.
        threshold = 0.1,                           # defines the min % of cells expressing a gene for this to be employed in the analysis.
        result_precision = 3,                      # Sets the rounding for the mean values in significan_means.
        debug = False,                             # Saves all intermediate tables emplyed during the analysis in pkl format.
        output_suffix = None                       # Replaces the timestamp in the output files by a user defined string in the  (default: None)
    )

    with open(f'cellphonedb-res/hgsc_{i_sample}', 'wb') as f:
        pkl.dump(cpdb_results, f)
    

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



# %%
ks = [adata.obs['samples'].astype(str).unique().item() for adata in adatas]

# %%
def melt_helper(x):
    x_melt = x.melt(ignore_index=False)
    x_melt['variable'] = x_melt.index + '_' + x_melt['variable']
    return x_melt

cellchat_corr_dict = {}
cellchat_melt_dict = {}

cellchat_dict = {}

for i, k in enumerate(ks):
    cellchat_res_path = f"../data/Ex1_hgsc/cellchat/{k}.csv"
    cellphonedb_res_path = f"cellphonedb-res/hgsc_{i}"
    if os.path.isfile(cellchat_res_path):
        # print(cellchat_res_path)
        cellchat = pd.read_csv(cellchat_res_path, index_col=0)
        if (cellchat == 0).all().all():
            print(cellchat_res_path, 'all zero. Ignore.')
            continue
            
        with open(cellphonedb_res_path, 'rb') as f:
            cpdb_results = pkl.load(f)
            
        cpdb_summary = cpdb_results['means_result'].loc[:, cpdb_results['means_result'].columns.str.contains('\|')].sum(axis=0)
        cpdb_unmelt = unmelt(cpdb_summary)
        cpdb_melt = melt_helper(cpdb_unmelt + cpdb_unmelt.T)
        cpdb_melt.columns = ['variable', 'value_cpdb']
            
        nichenet_unmelt = pd.read_csv(f"nichenet-res/nichenet_{k}.csv", index_col=0)
        nichenet_melt = melt_helper(nichenet_unmelt + nichenet_unmelt.T)
        nichenet_melt.columns = ['variable', 'value_nichenet']
            
        cellchat_dict[k] = cellchat
        cellchat_melt = melt_helper(cellchat + cellchat.T)
        cellchat_melt.columns = ['variable', 'value_cellchat']
        
        
        all_melt = pd.merge(cellchat_melt, cpdb_melt, on='variable', suffixes=['_cellchat', '_cpdb'])
        all_melt = pd.merge(all_melt, nichenet_melt, on='variable', suffixes=['**', '_nichenet'])
        
        corr_res = sp.stats.spearmanr(all_melt['value_cellchat'], all_melt['value_cpdb'])
        # cellchat_melt_dict[k] = all_melt
        cellchat_corr_dict[k] = [corr_res.statistic, corr_res.pvalue]
        corr_res = sp.stats.spearmanr(all_melt['value_cellchat'], all_melt['value_nichenet'])
        # cellchat_melt_dict[k] = all_melt
        cellchat_corr_dict[k].extend([corr_res.statistic, corr_res.pvalue])
        
    else:
        print(cellchat_res_path, 'not found!')

cellchat_corr_df = pd.DataFrame(cellchat_corr_dict, index=['cpdb_r', 'cpdb_p', 'nichenet_r', 'nichenet_p'])

cellchat_corr_df = cellchat_corr_df.T
cellchat_corr_df

# %%
orig_df = pd.read_csv("../data/Ex1_hgsc/res.csv", index_col=0)
cellchat_corr_df = orig_df.merge(cellchat_corr_df, left_index=True, right_index=True)

# %%
# fig, ax = plt.subplots(figsize=(2, 2))
# cellchat_corr_df.plot(kind='scatter', x='normneigh_r', y='steamboat_r', ax=ax)
g = sns.jointplot(
    data=cellchat_corr_df,
    x="cpdb_r", y="steamboat_r",
    kind="scatter",
    height=2,
    xlim=[0.1, .9],
    ylim=[0.1, .9]
)

ax = g.ax_joint

ax.set_xlabel('CellphoneDB')
ax.set_ylabel('Steamboat')
ax.plot([.0, 1.], [.0, 1.], ls='--', lw=1., c='k')

test_res = sp.stats.wilcoxon(cellchat_corr_df['cpdb_r'], cellchat_corr_df['steamboat_r'])
print(test_res)
ax.text(0.3, 0.15, f'p = {test_res.pvalue:.1e}')
(cellchat_corr_df['cpdb_r'] < cellchat_corr_df['steamboat_r']).sum() / cellchat_corr_df.shape[0]

# for pos in ['right', 'top']:
#    ax.spines[pos].set_visible(False)

# g.savefig("C:/Users/lshh/OneDrive/Publications/Steamboat/pub/fig-hgsc-elements/steamboat_vs_adjacency.pdf", **pltkw)

# %%
cellchat_corr_df['cpdb_r'] > cellchat_corr_df['steamboat_r']

# %%
cellchat_corr_df[['steamboat_r', 'adjacency_r', 'cpdb_r']].plot(kind='bar', figsize=(5, 2), width=.75)
ax = plt.gca()
for pos in ['right', 'top']:
    ax.spines[pos].set_visible(False)
plt.legend('')
plt.xlabel('Correlation with CellChat')
plt.ylabel('Samples')
plt.savefig("cci-by-sample.pdf")

# %%
fig, ax = plt.subplots(figsize=(1.5, 2))
sns.violinplot(cellchat_corr_df[['steamboat_r', 'adjacency_r', 'cpdb_r']], bw_adjust=0.5, orient='v', ax=ax)

for pos in ['right', 'top']:
    ax.spines[pos].set_visible(False)

ax.set_yticks([0, 0.5, 1])
ax.set_xticklabels(['Steamboat', 'Adjacency', 'CellphoneDB v5'], rotation=45, va='top', ha='right', rotation_mode='anchor') 
                  # rotation=45, rotation_mode='anchor', ha='right', va='center')
ax.set_ylabel('Correlation with CellChat')
fig.savefig("cci.pdf")

# %%
nichenet_unmelt = pd.read_csv("nichenet-res/nichenet_SMI_T10_F001.csv", index_col=0)
nichenet_melt = melt_helper(nichenet_unmelt + nichenet_unmelt.T)
nichenet_melt.columns = ['variable', 'value_nichenet']

cellchat_res_path = f"../data/Ex1_hgsc/cellchat/SMI_T10_F001.csv"
cellchat = pd.read_csv(cellchat_res_path, index_col=0)
cellchat_dict[k] = cellchat
cellchat_melt = melt_helper(cellchat + cellchat.T)
cellchat_melt.columns = ['variable', 'value_cellchat']

all_melt = pd.merge(cellchat_melt, nichenet_melt, on='variable', suffixes=['_cellchat', '_nichenet'])
corr_res = sp.stats.spearmanr(all_melt['value_cellchat'], all_melt['value_nichenet'])
corr_res


