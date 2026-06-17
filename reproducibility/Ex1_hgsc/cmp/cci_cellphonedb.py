# Ovarian cancer data analysis with CellPhoneDB
# Reproduces results in ../../data/Ex1_hgsc/cmp/cellphonedb

# %%
import os
import scanpy as sc
import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import os

# %%
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'arial'

pltkw = dict(bbox_inches='tight', transparent=True)

os.makedirs('../output/cellphonedb-res', exist_ok=True)

# %% [markdown]
# ## Load data

# %%
regenerate = False

h5ad_file = "../../data/Ex1_hgsc/ST_Discovery_so.h5ad"
if (not os.path.exists(h5ad_file)) or regenerate:
    # Local expected paths:
    # C:/Files/projects/Steamboat_cleanup/Steamboat/reproducibility/data/Ex1_hgsc/ST_Discovery_so_counts.mtx
    # C:/Files/projects/Steamboat_cleanup/Steamboat/reproducibility/data/Ex1_hgsc/ST_Discovery_so_metadata.csv
    # C:/Files/projects/Steamboat_cleanup/Steamboat/reproducibility/data/Ex1_hgsc/ST_Discovery_so_features.txt
    adata = sc.read_mtx("../../data/Ex1_hgsc/ST_Discovery_so_counts.mtx").T
    metadata = pd.read_csv("../../data/Ex1_hgsc/ST_Discovery_so_metadata.csv", index_col=0)
    features = pd.read_csv("../../data/Ex1_hgsc/ST_Discovery_so_features.txt", index_col=0, header=None)
    features.index = features.index.str.strip() # remove trailing white space in gene names
    features.index.name = 'gene_symbol'
    adata.obs = metadata
    adata.var = features
    adata.obsm['spatial'] = adata.obs[['x', 'y']].to_numpy()
    adata.write_h5ad(h5ad_file)
else:
    adata = sc.read_h5ad(h5ad_file)

## Metadata and gene sets

sample_metadata = pd.read_excel("../../data/Ex1_hgsc/sample_metadata.xlsx", index_col=0, sheet_name='Table 2b', skiprows=1)
sample_metadata = sample_metadata[sample_metadata['dataset'] == 'Discovery']

celltype_signatures = pd.read_excel("../../data/Ex1_hgsc/sample_metadata.xlsx", sheet_name='Table 3a', skiprows=2)
mtil_signautures = pd.read_excel("../../data/Ex1_hgsc/sample_metadata.xlsx", sheet_name='Table 6a', skiprows=2)
desmoplasia_signautures = pd.read_excel("../../data/Ex1_hgsc/sample_metadata.xlsx", sheet_name='Table 5a', skiprows=2)

def purge_gene_sets(df, prefix=''):
    res = {}
    for i in df.columns:
        res[prefix + i] = df[i].dropna().tolist()
    return res
celltype_signatures = pd.read_excel("../../data/Ex1_hgsc/sample_metadata.xlsx", index_col=0, sheet_name='Table 3b', skiprows=2).iloc[:, :-3]

genesets = (purge_gene_sets(celltype_signatures, 'sig_') | 
            purge_gene_sets(mtil_signautures, 'mtil_') | 
            purge_gene_sets(mtil_signautures, 'mtil_'))
genesets.keys()
del genesets['sig_Mast.cell']

sample_metadata

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
    df_meta.to_csv('../output/cellphonedb-res/hgsc_meta.tsv', sep = '\t')
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
        cpdb_file_path = '../../data/misc/cellphonedb/v5.0.0/cellphonedb.zip',           # mandatory: CellphoneDB database zip file.
        meta_file_path = '../output/cellphonedb-res/hgsc_meta.tsv',           # mandatory: tsv file defining barcodes to cell label.
        counts_file_path = adatas[i_sample],       # mandatory: normalized count matrix - a path to the counts file, or an in-memory AnnData object
        counts_data = 'hgnc_symbol',               # defines the gene annotation in counts matrix.
        microenvs_file_path = None, # optional (default: None): defines cells per microenvironment.
        score_interactions = True,                 # optional: whether to score interactions or not. 
        output_path = '../output/cellphonedb-res/',                    # Path to save results    microenvs_file_path = None,
        separator = '|',                           # Sets the string to employ to separate cells in the results dataframes "cellA|CellB".
        threads = 5,                               # number of threads to use in the analysis.
        threshold = 0.1,                           # defines the min % of cells expressing a gene for this to be employed in the analysis.
        result_precision = 3,                      # Sets the rounding for the mean values in significan_means.
        debug = False,                             # Saves all intermediate tables emplyed during the analysis in pkl format.
        output_suffix = None                       # Replaces the timestamp in the output files by a user defined string in the  (default: None)
    )

    with open(f'cellphonedb-res/hgsc_{i_sample}', 'wb') as f:
        pkl.dump(cpdb_results, f)
    
