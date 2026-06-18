import os
import scanpy as sc
import squidpy as sq
import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

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
# Training the model                                                                 # 
######################################################################################

n_heads = 25

sf.set_random_seed(0)
model = sf.Steamboat(adata.var_names.tolist(), n_heads=n_heads, n_scales=3)
model = model.to(device)

cuda_dataset = None

use_dataset = cuda_dataset
if use_dataset is None:
    use_dataset = dataset

model.fit(use_dataset, entry_masking_rate=0.1, feature_masking_rate=0.1,
          max_epoch=10000, 
          loss_fun=torch.nn.MSELoss(reduction='sum'),
          opt=torch.optim.Adam, opt_args=dict(lr=0.1), stop_eps=1e-3, report_per=200, stop_tol=200, device=device)

##################################################################################
#                         Saving the model                                       # 
##################################################################################

# create a directory to save the model if it doesn't exist
os.makedirs('../data/retrained_models', exist_ok=True)
torch.save(model.state_dict(), '../data/retrained_models/hgsc.pth')

print("Training completed and model saved as '../data/retrained_models/hgsc.pth'.")

# Output example:
# [2025-03-02 16:34:31,571::train::INFO] Epoch 1: train_loss 108.29949
# [2025-03-02 16:34:52,082::train::INFO] Epoch 201: train_loss 65.23699
# [2025-03-02 16:35:13,249::train::INFO] Epoch 401: train_loss 64.43046
# [2025-03-02 16:35:35,547::train::INFO] Epoch 601: train_loss 64.02565
# [2025-03-02 16:35:58,622::train::INFO] Epoch 801: train_loss 63.11885
# [2025-03-02 16:36:21,533::train::INFO] Epoch 1001: train_loss 62.52837
# [2025-03-02 16:36:43,575::train::INFO] Epoch 1201: train_loss 61.90503
# [2025-03-02 16:37:05,656::train::INFO] Epoch 1401: train_loss 61.92073
# [2025-03-02 16:37:26,926::train::INFO] Epoch 1601: train_loss 61.44772
# [2025-03-02 16:37:48,203::train::INFO] Epoch 1801: train_loss 61.36187
# [2025-03-02 16:38:09,090::train::INFO] Epoch 2001: train_loss 61.31289
# [2025-03-02 16:38:31,168::train::INFO] Epoch 2201: train_loss 61.13449
# [2025-03-02 16:38:52,523::train::INFO] Epoch 2401: train_loss 61.05229
# [2025-03-02 16:39:13,759::train::INFO] Epoch 2601: train_loss 61.05418
# [2025-03-02 16:39:18,513::train::INFO] Epoch 2647: train_loss 61.26666
# [2025-03-02 16:39:18,513::train::INFO] Stopping criterion met.
# Training completed and model saved as '../data/retrained_models/hgsc.pth'.
