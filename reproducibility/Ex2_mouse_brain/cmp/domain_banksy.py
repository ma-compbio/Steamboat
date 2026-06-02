# %%
import warnings
warnings.filterwarnings("ignore") 
import os, time
import pandas as pd
import numpy as np
import random
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, issparse

from banksy.initialize_banksy import initialize_banksy
from banksy.run_banksy import run_banksy_multiparam
from banksy_utils.color_lists import spagcn_color

start = time.perf_counter_ns()
random_seed = 1234
cluster_algorithm = 'leiden'
np.random.seed(random_seed)
random.seed(random_seed)

# %%
from tqdm.notebook import tqdm

# %%
import sklearn.metrics
import gc

# %% [markdown]
# ## Input-Ouput (IO) Options
# 1. Loading '.h5ad' file
# 2. Saving output images and '.csv' files in 'output_folder'

# %%
import scanpy as sc
all_adata = sc.read_h5ad("E:/allen-brain-cell-atlas/abc_atlas_data/temp/Zhuang-ABCA-1-labeled.h5ad")

# %%
adatas = []
for i in all_adata.obs['brain_section_label'].unique():
    adatas.append(all_adata[all_adata.obs['brain_section_label'] == i])

# %%
del all_adata
gc.collect()

# %%
coord_keys = ('x', 'y', 'spatial')
num_clusters = 20
annotation_key = 'parcellation_division'
output_folder = 'merfish_output/'

# %%
resolutions = [.9] # clustering resolution for Leiden clustering
pca_dims = [20] # number of dimensions to keep after PCA
lambda_list = [.8] # lambda
k_geom = 15 # 15 spatial neighbours
max_m = 1 # use AGF
nbr_weight_decay = "scaled_gaussian" # can also be "reciprocal", "uniform" or "ranked"

# %%
for i, adata in enumerate(tqdm(adatas)):
    if i < 26:
        continue
    try:
        banksy_dict = initialize_banksy(
            adata,
            coord_keys,
            k_geom,
            nbr_weight_decay=nbr_weight_decay,
            max_m=max_m,
            plt_edge_hist=True,
            plt_nbr_weights=True,
            plt_agf_angles=False,
            plt_theta=False,
        )
        results_df = run_banksy_multiparam(
        adata,
        banksy_dict,
        lambda_list,
        resolutions,
        color_list = spagcn_color * 10,
        max_m = max_m,
        filepath = output_folder,
        key = coord_keys,
        pca_dims = pca_dims,
        annotation_key = annotation_key,
        max_labels = num_clusters,
        cluster_algorithm = cluster_algorithm,
        match_labels = False,
        savefig = False,
        add_nonspatial = False,
        variance_balance = False,
        )
        adata.obs['banksy'] = results_df['labels'][0].dense
    except:
        pass

# %%
for i, adata in enumerate(tqdm(adatas)):
    if 'banksy' in adata.obs.columns:
        adata.obs[['banksy']].to_csv(f"../SpaceFormer/experiments/backup/mmbrain/banksy_spaital_domain_{i}.csv")
    else:
        print(i, 'failed.')

# %% [markdown]
# # Run BANKSY using defined parameters

# %%
aris = []
nmis = []
for adata in adatas:
    if 'banksy' in adata.obs.columns:
        ari = sklearn.metrics.adjusted_rand_score(adata.obs['parcellation_division'], adata.obs['banksy'])
        nmi = sklearn.metrics.adjusted_mutual_info_score(adata.obs['parcellation_division'], adata.obs['banksy'])
        aris.append(ari)
        nmis.append(nmi)
    else:
        aris.append(float('nan'))
        nmis.append(float('nan'))

# %%
df = pd.DataFrame({'ARI': aris, 'NMI': nmis})
df

# %%
df.to_csv("../SpaceFormer/experiments/backup/mmbrain/banksy_spatial_domain.csv")


