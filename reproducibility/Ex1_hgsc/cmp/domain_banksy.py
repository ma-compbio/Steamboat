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

import scanpy as sc
import squidpy as sq
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

start = time.perf_counter_ns()
random_seed = 1234
cluster_algorithm = 'leiden'
np.random.seed(random_seed)
random.seed(random_seed)

# %%
print(os.environ['CONDA_DEFAULT_ENV'])

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
adata = sc.read_h5ad("data/hgsc_SMI_T10_F001.h5ad")

# %%
! mkdir hgsc_output

# %%
adata.obs['dummy'] = 0
adata.obs['dummy'] = adata.obs['dummy'].astype('category')

# %%
coord_keys = ('x', 'y', 'spatial')
num_clusters = 3
annotation_key = 'dummy'
output_folder = 'hgsc_output/'

# %% [markdown]
# # Run BANKSY using defined parameters

# %%
resolutions = [.1] # clustering resolution for Leiden clustering
pca_dims = [60] # number of dimensions to keep after PCA
lambda_list = [.8] # lambda
k_geom = 15 # 15 spatial neighbours
max_m = 1 # use AGF
nbr_weight_decay = "scaled_gaussian" # can also be "reciprocal", "uniform" or "ranked"

# %%
adata.obs['x'] = adata.obsm['spatial'] [:,0]
adata.obs['y'] = adata.obsm['spatial'] [:,1]

# %%
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
max_labels = 3,
cluster_algorithm = cluster_algorithm,
match_labels = False,
savefig = False,
add_nonspatial = False,
variance_balance = False,
)
adata.obs['banksy'] = results_df['labels'][0].dense

adata.obs[['banksy']].to_csv(f"./hgsc_output/{adata.obs['samples'][0]}.csv")

# %%
adata.obs['banksy'].value_counts()

