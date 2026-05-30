# %%
# import relevant packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as ss
import seaborn as sns
from tqdm import tqdm

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

# %%
# load model
from celcomen.models.celcomen import celcomen
from celcomen.models.simcomen import simcomen

from celcomen.training_plan.train import train
from celcomen.datareaders.datareader import get_dataset_loaders

from celcomen.utils.helpers import calc_gex, get_pos, get_pos, calc_sphex, normalize_g2g

# set figure parameters
sc.settings.set_figure_params(dpi=100)

# %% [markdown]
# Advancements in genomics and machine learning have revolutionized both fields and for instance have culminated in models of Virtual Cells which predict the effect that changes in the micro- and macro-environment of the cell (such as perturbing the age of the donor, the tissue the cell is in, the drug treatment, knock-outs from guide RNAs etc) have on gene expression. 
# 
# Conversely, models of Virtual Tissues aim to not only estimate the effect the environment has on the cell but also the effect that the cell has on its environment and overall tissue.
# 
# Celcomen attempts to handle this problem using a causally identifiable GenAI framework which consists of two parts: the inference module (CCE) which uses spatial transcriptomics data to learn the values of gene-gene forces (up to their Markov equivalence class) and disentangle them to intra- and inter-cellular components; and the generative module (SCE) which uses the learned gene-gene forces to predict the effect of spatial counterfactuals, such as gene knokouts and cell injections, on the tissue.
# 
# The required inputs are raw gene counts stored as anndata. Download the cell_feature_matrix and cells.csv.gz from [10x_human_glioblastoma](https://www.10xgenomics.com/datasets/ffpe-human-brain-cancer-data-with-human-immuno-oncology-profiling-panel-and-custom-add-on-1-standard) and extract them into a directory labelled "data" in the same directory as this notebook, before proceeding. 
# 
# 
# 

# %%
perturbations = pd.read_csv("G:/Projects/Steamboat/experiments/perturbfish-surrogate/all_down_genes.csv", index_col=0)
perturbations['surrogate'] = perturbations['0'].apply(eval)
perturbations

# %% [markdown]
# # Pre-processing

# %%
avis = sc.read_h5ad("G:/data/perturbfish/cleaned/tumors_qc_test_spatial.h5ad")
avis

# %%
avis.obs['is_control'] = avis.obs['perturbation'] == 'Control'

# %%
x = 57500
y = 117500
mask = ((avis.obsm['spatial'][:, 0] > x) & (avis.obsm['spatial'][:, 0] < x + 25000) & 
        (avis.obsm['spatial'][:, 1] > y) & (avis.obsm['spatial'][:, 1] < y + 25000))
avis.obs['mask'] = mask
sc.pl.embedding(avis, basis='spatial', color=['mask', 'celltype2', 'is_control'], use_raw=False, s=1)
mask.sum()

# %%
avis.var['highly_variable'] = True
avis = avis[mask].copy()

# %%
avis.obs['is_control'].sum()

# %%
avis

# %%
from scipy.spatial.distance import pdist, squareform
from scipy import sparse

# create a gene subset for testing
genes = avis.var_names[avis.var['highly_variable']].tolist()
# avis = avis[:, genes].copy()
avis.X = sparse.csr_matrix(avis.X)
# retrieve positions from the data # only needed for simcomen later
pos = torch.from_numpy(avis.obsm['spatial'])
# convert the gene expression data to numpy
x = torch.from_numpy(avis[:, genes].X.todense())
# sphere normalize the data (just in case)
norm_factor = torch.pow(x, 2).sum(1).reshape(-1,1)
assert (norm_factor > 0).all()
x = torch.div(x, norm_factor)
# compute the distances
distances = squareform(pdist(avis.obsm['spatial']))
# compute the edges as two cell widths apart so 30µm
edge_index = torch.from_numpy(np.array(np.where((distances < 120) & (distances != 0))))

avis.obs["sangerID"] = "sample1"

avis.write_h5ad(
    'data/merfish/avis_preprocessed.h5',
)

# %% [markdown]
# # Run inference module to learn intra- and inter-cellular gene-gene forces


# %%
h5ad_path='data/merfish/avis_preprocessed.h5'
n_neighbors=8
loader = get_dataset_loaders(h5ad_path, "sangerID", n_neighbors, 120, 'cuda', True)

# %% [markdown]
# ### Select the hyperparameters of the model.
# zmft_scalar should be the highest value in the range (0,1) such that the loss can be stably minimized. 
# n_neighbors should be selected based on what disentanglement we want to achieve in the data. n_neighbors=6 targets disentanglement between intracellular and intercellular forces of short range.

# %%
n_genes = len(genes)
learning_rate = 1e-2
zmft_scalar = 1e-1
seed = 0
epochs = 50


model = celcomen(input_dim=n_genes, output_dim=n_genes, n_neighbors=n_neighbors, seed=seed)
model.to("cuda")
input_g2g = np.random.uniform(size=(n_genes, n_genes)).astype('float32')
input_g2g = normalize_g2g((input_g2g + input_g2g.T) / 2)

model.set_g2g(torch.from_numpy(input_g2g))
model.set_g2g_intra(torch.from_numpy(input_g2g))
model.to("cuda")

losses = train(epochs, learning_rate, model, loader, zmft_scalar=zmft_scalar, seed=1, device="cuda")

# %%
# create the plot
fig, ax = plt.subplots(figsize=[6, 4])
ax.grid(False)
ax.plot(losses, lw=2, color='#fe86a4')
ax.set_xlim(0, epochs)
vmin, vmax = min(min(losses), 0), max(losses)
vstep = (vmax - vmin) * 0.01
ax.set_ylim(vmin-vstep, vmax+vstep)
ax.set(xlabel='epochs', ylabel='loss')

# %% [markdown]
# The stable optimization of Celcomen indicates that zmft_scalar has been chosen appropriately.

# %% [markdown]
# # Run the generative module: to predict the effect of gene knock-outs 
# 
# The process of making a counterfactual prediction consists in: 
# 
# 1) creating a prompt spatial transcriptomic dataset
# 2) use Simcomen (SCE) to make the counterfactual prediction based on the prompt

# %%
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# %% [markdown]
# #### Create prompt 

# %%
res = {}

for test_var in ['IRAK4']:
    # define the genes that distinguish ISG15 spots currently
    # surrogate = perturbations.loc[test_var, 'surrogate']
    
    avis_sub = avis.copy()
    avis_sub.obs['near_ko'] = (avis_sub.obsp['spatial_connectivities'] @ 
                               (avis_sub.obs['perturbation'] == test_var).to_numpy()[:, None]) > 0
    avis_sub.obs['near_control'] = (avis_sub.obsp['spatial_connectivities'] @ 
                                    (avis_sub.obs['perturbation'] == 'Control').to_numpy()[:, None]) > 0
    
    # find T cells near ko or control tumor cells
    avis_sub.obs['t_near_ko'] = (avis_sub.obs['near_ko'] & 
                                 (avis_sub.obs['celltype2'] == 'T cells'))
    avis_sub.obs['t_near_control'] = (avis_sub.obs['near_control'] & 
                                      (avis_sub.obs['celltype2'] == 'T cells'))

    avis_sub1 = avis_sub.copy()
    
    ## without ko
    # propose an X 
    prompt_x = avis_sub.X.toarray().copy()
    # adjust the X so we artificially introduce signaling to the center left side of the tissue
    np.random.seed(0)
    # df_gex = sc.get.obs_df(avis_sub, keys=['ITGA1'])['ITGA1']
    # mask = avis_sub.obs.index == np.random.choice(avis_sub.obs.index[df_gex > 0], size=1)[0]
    
    mask = avis_sub.obs['perturbation'] == 'Control'
    # for j in surrogate:
    #     idx = np.where(avis_sub.var_names == j)[0][0]
    #     # prompt_x[mask, idx] = 0
    
    avis_sub.obs['perturbed'] = 'unperturbed'
    avis_sub.obs['perturbed'].iloc[np.where(squareform(pdist(avis_sub.obsm['spatial']))[mask, :] < 120 * 2)[1]] = 'perturbed-neighbors1'
    avis_sub.obs['perturbed'].iloc[np.where(squareform(pdist(avis_sub.obsm['spatial']))[mask, :] < 120)[1]] = 'perturbed-neighbors0'
    avis_sub.obs.loc[mask, 'perturbed'] = 'perturbed'
    avis_sub.uns['perturbed_colors'] = ['#ff47a6','#f593c2','#f7cbe0','#f7ebf1']
    sc.pl.embedding(avis_sub, basis='spatial', color=['perturbed'], use_raw=False, s=1e2)
    
    # define the parameters of the model
    n_genes = avis_sub.shape[1]
    learning_rate = 1e-7
    zmft_scalar = 1e-1
    seed = 0
    epochs = 25
    # instantiate the model, input and output will be the same
    simmodel = simcomen(input_dim=n_genes, output_dim=n_genes, n_neighbors=n_neighbors, seed=seed)
    # now perform the simulation
    np.random.seed(seed)
    # convert the gene expression data to numpy
    x = torch.from_numpy(prompt_x)
    # sphere normalize the data (just in case)
    norm_factor = torch.sqrt(torch.pow(x, 2).sum(1)).reshape(-1,1)
    assert (norm_factor > 0).all()
    x = torch.div(x, norm_factor)
    # artifically set the g2g matrix
    simmodel.set_g2g(model.conv1.lin.weight.clone().detach())
    simmodel.set_g2g_intra(model.lin.weight.clone().detach())
    # initialize a gene expression matrix
    assert np.isnan(x.detach().numpy()).sum() == 0
    input_sphex = calc_sphex(x.to('cuda')).clone()
    
    # move tensors and model to device cuda
    device="cuda"
    simmodel.set_sphex(input_sphex)
    simmodel.to(device)
    
    # set up the optimizer
    optimizer = torch.optim.SGD(simmodel.parameters(), lr=learning_rate, momentum=0)
    
    # keep track of the losses per data object
    loss, losses = None, []
    # train the model
    simmodel.train()
    tmp_gexs = []
    # work through epochs
    edge_index = edge_index.to('cuda')
    for epoch in tqdm(range(epochs), total=epochs):
        # derive the message as well as the mean field approximation
        msg, msg_intra, log_z_mft = simmodel(edge_index, 1)
        if (epoch % 5) == 0:
            tmp_gex = simmodel.gex.clone().detach().cpu().numpy()
            tmp_gexs.append(tmp_gex)
        # compute the loss and track it
        loss = -(-log_z_mft + zmft_scalar * torch.trace(torch.mm(msg, torch.t(simmodel.gex))) + zmft_scalar * torch.trace(torch.mm(msg_intra, torch.t(model.gex))) )
        losses.append(loss.detach().cpu().numpy()[0][0])
        # derive the gradients, update, and clear
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    fig, ax = plt.subplots(figsize=[6, 4])
    ax.grid(False)
    ax.plot(losses, lw=2, color='#fe86a4')
    ax.set_xlim(0, epochs)
    vmin, vmax = min(min(losses), 0), max(losses)
    vstep = (vmax - vmin) * 0.01
    ax.set_ylim(vmin-vstep, vmax+vstep)
    ax.set(xlabel='epochs', ylabel='loss')
    
    output_gex = simmodel.gex.detach().cpu().numpy()
    
    for idx, tmp_gex in enumerate(tmp_gexs):
        avis_sub.layers[f'input{idx}'] = tmp_gex
    avis_sub.layers['output'] = output_gex
    
    avis_sub0 = avis_sub

    avis_sub = avis_sub1
    ## with ko
    # propose an X 
    prompt_x = avis_sub.X.toarray().copy()
    # adjust the X so we artificially introduce signaling to the center left side of the tissue
    np.random.seed(0)
    # df_gex = sc.get.obs_df(avis_sub, keys=['ITGA1'])['ITGA1']
    # mask = avis_sub.obs.index == np.random.choice(avis_sub.obs.index[df_gex > 0], size=1)[0]

    substitute_x = prompt_x[avis_sub.obs['perturbation'] == test_var, :]
    mask = avis_sub.obs['perturbation'] == 'Control'
    # for j in surrogate:
        # idx = np.where(avis_sub.var_names == j)[0][0]
        # prompt_x[mask, idx] = 0
    prompt_x[mask, :] = substitute_x[np.random.choice(substitute_x.shape[0], (avis_sub.obs['perturbation'] == 'Control').sum(), 
                                                                                           replace=True), :]
    
    avis_sub.obs['perturbed'] = 'unperturbed'
    avis_sub.obs['perturbed'].iloc[np.where(squareform(pdist(avis_sub.obsm['spatial']))[mask, :] < 120 * 2)[1]] = 'perturbed-neighbors1'
    avis_sub.obs['perturbed'].iloc[np.where(squareform(pdist(avis_sub.obsm['spatial']))[mask, :] < 120)[1]] = 'perturbed-neighbors0'
    avis_sub.obs.loc[mask, 'perturbed'] = 'perturbed'
    avis_sub.uns['perturbed_colors'] = ['#ff47a6','#f593c2','#f7cbe0','#f7ebf1']
    sc.pl.embedding(avis_sub, basis='spatial', color=['perturbed'], use_raw=False, s=1e2)
    
    # define the parameters of the model
    n_genes = avis_sub.shape[1]
    learning_rate = 1e-7
    zmft_scalar = 1e-1
    seed = 0
    epochs = 25
    # instantiate the model, input and output will be the same
    simmodel = simcomen(input_dim=n_genes, output_dim=n_genes, n_neighbors=n_neighbors, seed=seed)
    # now perform the simulation
    np.random.seed(seed)
    # convert the gene expression data to numpy
    x = torch.from_numpy(prompt_x)
    # sphere normalize the data (just in case)
    norm_factor = torch.sqrt(torch.pow(x, 2).sum(1)).reshape(-1,1)
    assert (norm_factor > 0).all()
    x = torch.div(x, norm_factor)
    # artifically set the g2g matrix
    simmodel.set_g2g(model.conv1.lin.weight.clone().detach())
    simmodel.set_g2g_intra(model.lin.weight.clone().detach())
    # initialize a gene expression matrix
    assert np.isnan(x.detach().numpy()).sum() == 0
    input_sphex = calc_sphex(x.to('cuda')).clone()
    
    # move tensors and model to device cuda
    device="cuda"
    simmodel.set_sphex(input_sphex)
    simmodel.to(device)
    
    # set up the optimizer
    optimizer = torch.optim.SGD(simmodel.parameters(), lr=learning_rate, momentum=0)
    
    # keep track of the losses per data object
    loss, losses = None, []
    # train the model
    simmodel.train()
    tmp_gexs = []
    # work through epochs
    edge_index = edge_index.to('cuda')
    for epoch in tqdm(range(epochs), total=epochs):
        # derive the message as well as the mean field approximation
        msg, msg_intra, log_z_mft = simmodel(edge_index, 1)
        if (epoch % 5) == 0:
            tmp_gex = simmodel.gex.clone().detach().cpu().numpy()
            tmp_gexs.append(tmp_gex)
        # compute the loss and track it
        loss = -(-log_z_mft + zmft_scalar * torch.trace(torch.mm(msg, torch.t(simmodel.gex))) + zmft_scalar * torch.trace(torch.mm(msg_intra, torch.t(model.gex))) )
        losses.append(loss.detach().cpu().numpy()[0][0])
        # derive the gradients, update, and clear
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    fig, ax = plt.subplots(figsize=[6, 4])
    ax.grid(False)
    ax.plot(losses, lw=2, color='#fe86a4')
    ax.set_xlim(0, epochs)
    vmin, vmax = min(min(losses), 0), max(losses)
    vstep = (vmax - vmin) * 0.01
    ax.set_ylim(vmin-vstep, vmax+vstep)
    ax.set(xlabel='epochs', ylabel='loss')
    
    output_gex = simmodel.gex.detach().cpu().numpy()
    
    for idx, tmp_gex in enumerate(tmp_gexs):
        avis_sub.layers[f'input{idx}'] = tmp_gex
    avis_sub.layers['output'] = output_gex
    
    subset = avis_sub.obs['t_near_control'] & (~avis_sub.obs['t_near_ko'])
    cmp_adata = sc.AnnData(np.vstack([avis_sub1[subset].layers['output'], 
                                      avis_sub0[subset].layers['output']]), 
                           var=avis_sub.var.copy())
    
    cmp_adata.obs['grp'] = ['KO'] * subset.sum() + ['WT'] * subset.sum()
    sc.tl.rank_genes_groups(cmp_adata, groupby='grp', method='wilcoxon')
    cmp_df_sf = sc.get.rank_genes_groups_df(cmp_adata, group="KO")
    
    cmp_df_gt = pd.read_csv(f"output/{test_var}_gt.csv",
                           index_col=0)
    
    cmp_df_merge = pd.merge(cmp_df_gt, cmp_df_sf, left_on='names', right_on='names', suffixes=['_gt', '_sf'])
        # cmp_df_merge
        
    cmp_df_merge.to_csv(f"output/celcomen_{test_var}.csv")
    
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    rho = cmp_df_merge[['logfoldchanges_gt', 'logfoldchanges_sf']].corr(method='spearman').iloc[0, 1]
    cmp_df_merge.plot(x='logfoldchanges_gt', y='logfoldchanges_sf', kind='scatter', s=1, ax=ax)
    ax.set_title(f"{test_var}: {rho: .2f}")
    ax.set_xlabel('Ground-truth change')
    ax.set_ylabel('Predicted change')
    
    print(f"{test_var}: {rho: .2f}")
    res[test_var] = rho
    fig.savefig(f"output/celcomen_{test_var}.pdf")


# %%
pd.Series(res).sort_values(ascending=False).plot(kind='bar', figsize=(2, 1))
plt.ylabel('correlation with\nground truth')
plt.axhline(0.3, ls='--', c='r')
fig.savefig(f"output/celcomen_summary.pdf")

# %%
pd.Series(res).to_csv("output/celcomen_summary.csv")