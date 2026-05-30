# %% [markdown]
# # Investigating Global Attention and Prognosis on Colorectal CODEX data
# 
# - Dataset: [Schürch, Christian M., et al. "Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front." Cell 182.5 (2020): 1341-1359.](https://doi.org/10.1016/j.cell.2020.07.005)
# - Tasks:
#   - Perform standard processing of the dataset
#   - Confirm that attention heads roughly correspond to cell types.
#   - Invstigate the relationship of global attention and prognosis.

# %%
import os
import sys
sys.path.append("../../")
device = "cuda"

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

model.fit(cuda_dataset, entry_masking_rate=0.1, feature_masking_rate=0.0,
          max_epoch=10000, 
          loss_fun=torch.nn.MSELoss(reduction='sum'),
          opt=torch.optim.Adam, opt_args=dict(lr=0.1), stop_eps=1e-3, report_per=200, stop_tol=200, device=device)

torch.save(model.state_dict(), '../data/retrained_models/crc_codex.pth')