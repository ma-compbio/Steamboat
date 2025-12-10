import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .model import Steamboat
# from .integrated_model # import IntegratedSteamboat
from typing import List, Literal
from torch import nn
import scanpy as sc
import numpy as np
import torch
from .dataset import SteamboatDataset
import scipy as sp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import explained_variance_score
from tqdm.auto import tqdm
import squidpy as sq

palettes = {
    'ncr10': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', 
              '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85'],
    'npg3': ['#952522', '#0a4b84', '#98752b']
}

def rank(x, axis=1):
    """Rank number

    :param x: numpy array of numbers
    :param axis: perform over which axis, defaults to 1
    :return: ranks
    """
    return np.argsort(np.argsort(x, axis=axis), axis=axis)


def calc_v_weights(model: Steamboat, normalize: bool = True):
    """Calculate weight of reconstruction (w_v) metagene

    :param model: Steamboat model
    :param normalize: whether normalize the sum to 1, defaults to True
    :return: weights
    """
    v_weights = model.spatial_gather.v.weight.detach().cpu().numpy().sum(axis=0)
    if normalize:
        v_weights = v_weights / sum(v_weights)
    return v_weights


def calc_head_weights(adatas, model: Steamboat):
    """Calculate weights of heads and scales within each head

    :param adatas: all adatas
    :param model: the trained Steamboat model
    :return: weights
    """
    ego = 0
    local = 0
    regional = 0

    for i in range(len(adatas)):
        ego += np.mean(adatas[i].obsm['ego_attn'], axis=0)
        local += np.mean(adatas[i].obsm['local_attn'], axis=0)
        if 'global_attn_0' in adatas[i].obsm:
            regional += np.mean(adatas[i].obsm['global_attn_0'], axis=0)
    if 'global_attn_0' in adatas[i].obsm:
        matrix = np.vstack([ego, local, regional]) * calc_v_weights(model)
    else:
        matrix = np.vstack([ego, local]) * calc_v_weights(model)
    return matrix

def calc_head_weights_quantile(adatas, model: Steamboat, quantile: float = 0.9):
    """Calculate weights of heads and scales within each head

    :param adatas: all adatas
    :param model: the trained Steamboat model
    :return: weights
    """
    ego = 0
    local = 0
    regional = 0

    for i in range(len(adatas)):
        ego += np.quantile(adatas[i].obsm['ego_attn'], quantile, axis=0)
        local += np.quantile(adatas[i].obsm['local_attn'], quantile, axis=0)
        if 'global_attn_0' in adatas[i].obsm:
            regional += np.quantile(adatas[i].obsm['global_attn_0'], quantile, axis=0)
    if 'global_attn_0' in adatas[i].obsm:
        matrix = np.vstack([ego, local, regional])# * calc_v_weights(model)
    else:
        matrix = np.vstack([ego, local])# * calc_v_weights(model)
    return matrix


def plot_head_weights(head_weights, multiplier: float = 100, order=None, figsize=(7, 0.8), heatmap_kwargs=None, save: str = None):
    """Plot head weights calculated by calc_head_weights

    :param head_weights: head weights calculated by calc_head_weights
    :param multiplier: 100 for percentage, 1000 for mills, etc., defaults to 100
    :param order: ordering of heads, defaults to None
    :param figsize: (width, height), defaults to (7, 0.8)
    :param heatmap_kwargs: additional arguments for heatmap plotting, defaults to None
    :param save: save to file, defaults to None
    """
    matrix = head_weights.copy()
    matrix /= matrix.sum()
    fig, ax = plt.subplots(figsize=figsize)

    heatmap_kwargs0 = dict(vmax=10, linewidths=0.2, linecolor='grey', cmap='Reds', annot=True, fmt='.0f', square=True)
    if heatmap_kwargs is not None:
        for key, value in heatmap_kwargs.items():
            heatmap_kwargs0[key] = value
    
    if order is None:
        sns.heatmap(matrix * multiplier, ax=ax, **heatmap_kwargs0)
    else:
        sns.heatmap(matrix[:, order] * multiplier, vmax=10, ax=ax, linewidths=0.2, linecolor='grey', cmap='Reds', annot=True, fmt='.0f', square=True)
        ax.set_xticklabels(order, rotation=0)
    
    if matrix.shape[0] == 3:
        ax.set_yticklabels(['ego', 'local', 'regional'], rotation=0)
    elif matrix.shape[0] == 2:
        ax.set_yticklabels(['ego', 'local'], rotation=0)

    if save is not None and save != False:
        assert isinstance(save, str), "save must be a string."
        fig.savefig(save, bbox_inches='tight', transparent=True)


def calc_interaction(adatas, model: Steamboat, sample_key: str, cell_type_key: str, pseudocount: float = 20.):
    """Calculate interaction matrix

    :param adatas: all adatas
    :param model: Steamboat model
    :param sample_key: obs key for sample names
    :param cell_type_key: obs key for cell types
    :param pseudocount: pseudocount in denominator when averaging scores in cell type pairs, defaults to 20.
    :return: interaction matrices (one per sample) in a dictionary
    """
    v_weights = calc_v_weights(model)
    celltype_attnp_df_dict = {}
    for i in range(len(adatas)):    
        total_attnp = None
        for j in range(model.spatial_gather.n_heads):
            if total_attnp is None:
                total_attnp = adatas[i].obsp[f'local_attn_{j}'] * v_weights[j] * model.spatial_gather.n_heads
            else:
                total_attnp += adatas[i].obsp[f'local_attn_{j}'] * v_weights[j] * model.spatial_gather.n_heads

        celltypes = sorted(adatas[i].obs[cell_type_key].unique())
        celltype_attnp_df = pd.DataFrame(-1., index=celltypes, columns=celltypes)
        
        actual_min = float("inf")
        for celltype0 in celltype_attnp_df.index:
            mask0 = (adatas[i].obs[cell_type_key] == celltype0)
            for celltype1 in celltype_attnp_df.columns:
                mask1 = (adatas[i].obs[cell_type_key] == celltype1)
                sub_attnp = total_attnp[mask0, :][:, mask1]
                normalization_factor = sub_attnp.nnz + pseudocount
                # normalization_factor = np.prod(sub_attnp.shape)
                if normalization_factor >= 1:
                    celltype_attnp_df.loc[celltype0, celltype1] = sub_attnp.sum() / normalization_factor
                    actual_min = min(actual_min, celltype_attnp_df.loc[celltype0, celltype1])
                else:
                    celltype_attnp_df.loc[celltype0, celltype1] = 0.

        celltype_attnp_df_dict[adatas[i].obs[sample_key].unique().astype(str).item()] = celltype_attnp_df + celltype_attnp_df.T
    return celltype_attnp_df_dict


def calc_adjacency_freq(adatas, sample_key: str, cell_type_key: str, pseudocount: float = 20.):
    """Calculate baseline interaction matrix determined by adjacency frequency

    :param adatas: all adatas
    :param sample_key: obs key for sample names
    :param cell_type_key: obs key for cell types
    :return: adjacency frequency matrices (one per sample) in a dictionary
    """
    adjacency_freq = {}
    for i in range(len(adatas)):
        k = adatas[i].obs[sample_key].unique().astype(str).item()
        adatas[i].obs[cell_type_key] = adatas[i].obs[cell_type_key].astype("category")
        sq.gr.interaction_matrix(adatas[i], cell_type_key, normalized=False)
        temp = pd.DataFrame(adatas[i].uns[f'{cell_type_key}_interactions'], 
                                            index=adatas[i].obs[cell_type_key].cat.categories, 
                                            columns=adatas[i].obs[cell_type_key].cat.categories)
        normalization_factor = adatas[i].obs[cell_type_key].value_counts().sort_index() + pseudocount
        adjacency_freq[k] = temp.div(normalization_factor, axis=0).div(normalization_factor, axis=1)
    return adjacency_freq


def calc_var(model: Steamboat):
    """Write metagenes into a DataFrame

    :param model: Steamboat model
    :return: DataFrame of metagenes
    """
    n_heads = model.spatial_gather.n_heads
    q = model.spatial_gather.q.weight.detach().cpu().numpy()
    k_local = model.spatial_gather.k_local.weight.detach().cpu().numpy()
    k_global = model.spatial_gather.k_regionals[0].weight.detach().cpu().numpy()
    v = model.spatial_gather.v.weight.detach().cpu().numpy().T

    index = ([f'q_{i}' for i in range(n_heads)] + 
            [f'k_local_{i}' for i in range(n_heads)] + 
            [f'k_global_{i}' for i in range(n_heads)] + 
            [f'v_{i}' for i in range(n_heads)])
    
    return pd.DataFrame(np.vstack([q, k_local, k_global, v]), 
                        index=index, columns=model.features).T


def find_top_celltypes(adata, obs_key: str, top: int = 3, return_raw: bool = True):
    """Find top contributing cell types for each metagene by mean scores

    :param adata: AnnData object with cell type information
    :param obs_key: obs key for cell types
    :param top: top cell types per metagene to find, defaults to 3
    :return: list of lists of top cell types per metagene
    """
    celltypes = sorted(adata.obs[obs_key].unique().astype(str))
    celltype_masks = {}
    for celltype in celltypes:
        celltype_masks[celltype] = (adata.obs[obs_key] == celltype).values

    heads = adata.obsm['q'].shape[1]
    q_celltypes = [[] for _ in range(heads)]

    for head in range(heads):
        scores = {}
        for celltype in celltypes:
            mask = celltype_masks[celltype]
            scores[celltype] = adata.obsm['q'][mask, head].mean()
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_celltypes = [item[0] for item in sorted_scores[:top]]
        q_celltypes[head] = top_celltypes
    
    k_local_celltypes = [[] for _ in range(heads)]
    for head in range(heads):
        scores = {}
        for celltype in celltypes:
            mask = celltype_masks[celltype]
            scores[celltype] = adata.obsm['local_k'][mask, head].mean()
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_celltypes = [item[0] for item in sorted_scores[:top]]
        k_local_celltypes[head] = top_celltypes
    
    if return_raw:
        return q_celltypes, k_local_celltypes
    else:
        return pd.DataFrame({'q': q_celltypes, 'local_k': k_local_celltypes})


def find_top_lrs(var: pd.DataFrame, lr: pd.DataFrame, top: int = 3, return_raw: bool = True):
    """Find top contributing ligand-receptor pairs for each metagene

    :param var: var from `calc_var(model)`
    :param top: top genes per metagene to find, defaults to 3
    :return: list of lists of top genes per metagene
    """
    l_mask = lr['ligand'].isin(var.index)
    r_mask = lr['receptor'].isin(var.index)
    lr_mask = l_mask & r_mask
    mylr = lr.loc[lr_mask, :]
    lr_genes = set(mylr['ligand']).union(set(mylr['receptor']))
    myvar = var.loc[var.index.isin(lr_genes), :]
    
    heads = len(var.columns) / 4
    lrs = [[] for _ in range(int(heads))]
    for head in range(int(heads)):
        scores = {}
        for _, row in mylr.iterrows():
            ligand = row['ligand']
            receptor = row['receptor']
            score = myvar.loc[ligand, f'q_{head}'] * myvar.loc[receptor, f'k_local_{head}']
            scores[f'{ligand}-{receptor}'] = score
            score = myvar.loc[receptor, f'q_{head}'] * myvar.loc[ligand, f'k_local_{head}']
            scores[f'{receptor}-{ligand}'] = score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_lrs = [item[0] for item in sorted_scores[:top]]
        lrs[head] = top_lrs

    if return_raw:
        return lrs
    else:
        return pd.DataFrame({'lr_pairs': lrs})


def calc_geneset_auroc(metagenes, genesets):
    """Gene set enrichment analysis by AUROC

    :param metagenes: metagenes from `calc_var(model)`
    :param genesets: genesets in a dictionary
    :return: AUROC in a DataFrame
    """
    gene_df = metagenes.copy()
    df = pd.DataFrame(index=gene_df.columns)
    for k, v in genesets.items():
        aurocs = []
        pvals = []
        for i in gene_df.columns:
            aurocs.append(roc_auc_score(gene_df.index.isin(v), gene_df[i]))
            pvals.append(sp.stats.mannwhitneyu(gene_df.loc[gene_df.index.isin(v), i],
                                                           gene_df.loc[~gene_df.index.isin(v), i]).pvalue)
        df[k + '_auroc'] = aurocs
        # df[k + '_p'] = pvals
    return df


def calc_geneset_auroc_order(sig_df, by='q'):
    """Order the metagenes by AUROC

    :param sig_df: Analysis results
    :param by: by which metagene, defaults to 'q'
    :return: ordering of the metagenes
    """
    plt_df = sig_df[sig_df.index.str.contains(by + '_')]
    order = np.argsort(np.argmax(plt_df, axis=1) - np.max(plt_df, axis=1) / (np.max(plt_df) + 1)).tolist()
    return order


def plot_geneset_auroc(sig_df, order, figsize=(8, 5)):
    """Plot gene set enrichment by AUROC

    :param sig_df: Analysis results
    :param order: order of heads
    :param figsize: (width, height), defaults to (8, 5)
    :return: `fig, ax`
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    ax = axes[0]
    plt_df = sig_df[sig_df.index.str.contains('q_')]
    sns.heatmap(plt_df.T.iloc[:, order], vmin=.2, vmax=.8, cmap='vlag', 
            linewidths=.5, linecolor='grey', ax=ax, square=True)
    ax.set_xticklabels(order, rotation=0)
    ax.set_xlabel('Center cell metagenes')

    ax = axes[1]
    plt_df = sig_df[sig_df.index.str.contains('k_local')]
    sns.heatmap(plt_df.T.iloc[:, order], vmin=.2, vmax=.8, cmap='vlag', 
            linewidths=.5, linecolor='grey', ax=ax, square=True)
    ax.set_xticklabels(order, rotation=0)
    ax.set_xlabel('Local environment metagenes')

    ax = axes[2]
    plt_df = sig_df[sig_df.index.str.contains('k_global')]
    sns.heatmap(plt_df.T.iloc[:, order], vmin=.2, vmax=.8, cmap='vlag', 
            linewidths=.5, linecolor='grey', ax=ax, square=True)
    ax.set_xticklabels(order, rotation=0)
    ax.set_xlabel('Global environment metagenes')

    fig.tight_layout()
    return fig, ax

def explained_variance_by_scale(model: Steamboat, dataset: SteamboatDataset, adatas: list[sc.AnnData], device='cuda'):
    """Calculate explained variance for each scale

    :param model: Steamboat model
    :param dataset: SteamboatDataset object to be processed
    :param adatas: list of AnnData objects corresponding to the dataset
    :param device: Device to run the model, defaults to 'cuda'
    :return: explained variance scores in a dictionary
    """
    # Safeguards
    assert len(adatas) == len(dataset), "mismatch in lenghths of adatas and dataset"
    for i, (adata, data) in enumerate(zip(adatas, dataset)):
        assert adata.shape[0] == data[0].shape[0], f"adata[{i}] has {adata.shape[0]} cells but dataset[{i}] has {data[0].shape[0]}."

    # Calculate embeddings and attention scores for each slide
    total_mean = 0
    total_variance = 0
    total_n_cells = 0
    for x, _, _, _ in dataset:
        total_mean += x.sum(dim=0).cpu().numpy()
        total_n_cells += x.shape[0]
    total_mean /= total_n_cells
    
    for x, _, _, _ in dataset:
        total_variance += ((x - torch.tensor(total_mean).to(x.device)) ** 2).sum(dim=0).cpu().numpy()
    total_variance /= total_n_cells

    total_evs = {'ego': 0, 'local': 0, 'global': 0, 'full': 0}
    full_evs = 0.
    total_marginal_evs = {'ego': 0, 'local': 0, 'global': 0}
    for i, (x, adj_list, regional_xs, regional_adj_lists) in tqdm(enumerate(dataset), total=len(dataset)):
        adj_list = adj_list.squeeze(0).to(device)
        x = x.squeeze(0).to(device)
        regional_adj_lists = [regional_adj_list.to(device) for regional_adj_list in regional_adj_lists]
        regional_xs = [regional_x.to(device) for regional_x in regional_xs]
        
        with torch.no_grad():
            res = model(adj_list, x, x, regional_adj_lists, regional_xs, get_details=False)
            full_evs += ((x - res) ** 2).sum(axis=0).cpu().numpy()
            for scale in ['ego', 'local', 'global']:
                res = model(adj_list, x, x, regional_adj_lists, regional_xs, 
                            get_details=False, explained_variance_mask=scale)
                total_evs[scale] += ((x - res) ** 2).sum(axis=0).cpu().numpy()
                res = model(adj_list, x, x, regional_adj_lists, regional_xs, 
                            get_details=False, explained_variance_mask='no ' + scale)
                total_marginal_evs[scale] += ((x - res) ** 2).sum(axis=0).cpu().numpy()
        
    avg_evs = {}
    avg_evs['full'] = 1 - (full_evs / total_n_cells).sum() / total_variance.sum()
    for scale in ['ego', 'local', 'global']:
        # variance_weighted_evs[scale] = 1 - (total_evs[scale] / total_n_cells).sum() / total_variance.sum()
        avg_evs[scale] = 1 - (total_marginal_evs[scale] / total_n_cells).sum() / total_variance.sum()
        avg_evs[scale] = avg_evs['full'] - avg_evs[scale]
    evs = {}
    evs['full'] = 1 - full_evs / total_n_cells / total_variance
    for scale in ['ego', 'local', 'global']:
        evs[scale] = 1 - total_evs[scale] / total_n_cells / total_variance
        evs[scale] = 1 - total_marginal_evs[scale] / total_n_cells / total_variance
        evs[scale] = evs['full'] - evs[scale]
    
    for i in avg_evs:
        avg_evs[i] = float(avg_evs[i])

    return evs, avg_evs

def sequential_explained_variance_by_scale(model: Steamboat, dataset: SteamboatDataset, adatas: list[sc.AnnData], device='cuda'):
    """Calculate explained variance for each scale

    :param model: Steamboat model
    :param dataset: SteamboatDataset object to be processed
    :param adatas: list of AnnData objects corresponding to the dataset
    :param device: Device to run the model, defaults to 'cuda'
    :return: explained variance scores in a dictionary
    """
    # Safeguards
    assert len(adatas) == len(dataset), "mismatch in lenghths of adatas and dataset"
    for i, (adata, data) in enumerate(zip(adatas, dataset)):
        assert adata.shape[0] == data[0].shape[0], f"adata[{i}] has {adata.shape[0]} cells but dataset[{i}] has {data[0].shape[0]}."

    # Calculate embeddings and attention scores for each slide
    total_mean = 0
    total_variance = 0
    total_n_cells = 0
    for x, _, _, _ in dataset:
        total_mean += x.sum(dim=0).cpu().numpy()
        total_n_cells += x.shape[0]
    total_mean /= total_n_cells
    
    for x, _, _, _ in dataset:
        total_variance += ((x - torch.tensor(total_mean).to(x.device)) ** 2).sum(dim=0).cpu().numpy()
    total_variance /= total_n_cells

    total_evs = {'ego': 0, 'ego+local': 0, 'full': 0}
    for i, (x, adj_list, regional_xs, regional_adj_lists) in tqdm(enumerate(dataset), total=len(dataset)):
        adj_list = adj_list.squeeze(0).to(device)
        x = x.squeeze(0).to(device)
        regional_adj_lists = [regional_adj_list.to(device) for regional_adj_list in regional_adj_lists]
        regional_xs = [regional_x.to(device) for regional_x in regional_xs]
        
        with torch.no_grad():
            for scale in ['ego', 'ego+local', 'full']:
                res = model(adj_list, x, x, regional_adj_lists, regional_xs, 
                            get_details=False, explained_variance_mask=scale)
                total_evs[scale] += ((x - res) ** 2).sum(axis=0).cpu().numpy()
        
    avg_evs = {}
    avg_evs['full'] = 1 - total_evs['full'].sum() / total_n_cells / total_variance.sum()
    avg_evs['ego'] = 1 - (total_evs['ego']).sum() / total_n_cells / total_variance.sum()
    avg_evs['ego+local'] = 1 - (total_evs['ego+local']).sum() / total_n_cells / total_variance.sum()
    avg_evs['local'] = avg_evs['ego+local'] - avg_evs['ego']
    avg_evs['global'] = avg_evs['full'] - avg_evs['ego+local']
    del avg_evs['ego+local']  # remove redundant entry

    evs = {}
    evs['full'] = 1 - total_evs['full'] / total_n_cells / total_variance
    evs['ego'] = 1 - total_evs['ego'] / total_n_cells / total_variance
    evs['ego+local'] = 1 - total_evs['ego+local'] / total_n_cells / total_variance
    evs['local'] = evs['ego+local'] - evs['ego']
    evs['global'] = evs['full'] - evs['ego+local']
    del evs['ego+local']  # remove redundant entry
    
    for i in avg_evs:
        avg_evs[i] = float(avg_evs[i])
    return evs, avg_evs

def calc_obs(adatas: list[sc.AnnData], dataset: SteamboatDataset, model: Steamboat, 
                    device='cuda', get_recon: bool = False):
    """Calculate and store the embeddings and attention scores in the AnnData objects
    
    :param adatas: List of AnnData objects to store the embeddings and attention scores
    :param dataset: SteamboatDataset object to be processed
    :param model: Steamboat model
    :param device: Device to run the model, defaults to 'cuda'
    :param get_recon: Whether to store the reconstructed data, defaults to False
    """
    # Safeguards
    assert len(adatas) == len(dataset), "mismatch in lenghths of adatas and dataset"
    for i, (adata, data) in enumerate(zip(adatas, dataset)):
        assert adata.shape[0] == data[0].shape[0], f"adata[{i}] has {adata.shape[0]} cells but dataset[{i}] has {data[0].shape[0]}."

    # Calculate embeddings and attention scores for each slide
    for i, (x, adj_list, regional_xs, regional_adj_lists) in tqdm(enumerate(dataset), total=len(dataset)):
        adj_list = adj_list.squeeze(0).to(device)
        x = x.squeeze(0).to(device)
        regional_adj_lists = [regional_adj_list.to(device) for regional_adj_list in regional_adj_lists]
        regional_xs = [regional_x.to(device) for regional_x in regional_xs]
        
        with torch.no_grad():
            res, details = model(adj_list, x, x, regional_adj_lists, regional_xs, get_details=True)
            
            if get_recon:
                adatas[i].obsm['X_recon'] = res.cpu().numpy()

            adatas[i].obsm['q'] = details['embq'].cpu().numpy()
            adatas[i].obsm['local_k'] = details['embk'][0].cpu().numpy()

            for j in range(model.spatial_gather.n_scales - 2):
                adatas[i].obsm[f'global_k_{j}'] = model.spatial_gather.k_regionals[j](x).cpu().numpy()

            for j, emb in enumerate(details['embk'][1]):
                adatas[i].uns[f'global_k_{j}'] = emb.cpu().numpy()
                
            adatas[i].obsm['attn'] = details['attn'].cpu().numpy()
            adatas[i].obsm['ego_attn'] = details['attnm'][0].cpu().numpy()
            adatas[i].obsm['local_attn'] = details['attnm'][1].cpu().numpy()

            for j, matrix in enumerate(details['attnm'][2]):
                adatas[i].obsm[f'global_attn_{j}'] = matrix.cpu().numpy()

            # local attention (as graph)
            for j in range(model.spatial_gather.n_heads):
                w = details['attnp'][1].cpu().numpy()[:, j, :].flatten()
                uv = adj_list.cpu().numpy()
                u = uv[0]
                v = uv[1]
                if uv.shape[0] == 3: # masked for unequal neighbors
                    m = (uv[2] > 0)
                    w, u, v = w[m], u[m], v[m]
                adatas[i].obsp[f'local_attn_{j}'] = sp.sparse.csr_matrix((w, (u, v)), 
                                                                            shape=(adatas[i].shape[0], 
                                                                                adatas[i].shape[0]))


def gather_obs(adata: sc.AnnData, adatas: list[sc.AnnData]):
    """Gather obs/obsm/uns from a list of AnnData objects to a single AnnData object

    :param adata: AnnData object to store the gathered obs/obsm/uns
    :param adatas: List of AnnData objects to be gathered
    """
    all_embq = []
    all_embk = []
    all_embk_glb = []
    all_ego_attn = []
    all_local_attn = []
    all_global_attn = []
    all_attn = []
    
    for i in range(len(adatas)):
        all_embq.append(adatas[i].obsm['q'])
        all_embk.append(adatas[i].obsm['local_k'])
        all_ego_attn.append(adatas[i].obsm['ego_attn'])
        all_local_attn.append(adatas[i].obsm['local_attn'])
        all_global_attn.append(adatas[i].obsm['global_attn_0'])
        all_attn.append(adatas[i].obsm['attn'])
        all_embk_glb.append(adatas[i].obsm['global_k_0'])

    adata.obsm['q'] = np.vstack(all_embq)
    adata.obsm['local_k'] = np.vstack(all_embk)
    adata.obsm['ego_attn'] = np.vstack(all_ego_attn)
    adata.obsm['local_attn'] = np.vstack(all_local_attn)
    adata.obsm['global_attn'] = np.vstack(all_global_attn)
    
    adata.obsm['attn'] = np.vstack(all_attn)
    adata.obsm['global_k_0'] = np.vstack(all_embk_glb)

    if 'X_recon' in adatas[0].obsm:
        all_recon = []
        for i in range(len(adatas)):
            all_recon.append(adatas[i].obsm['X_recon'])
        adata.obsm['X_recon'] = np.vstack(all_recon)

    return adata


def neighbors(adata: sc.AnnData,
              use_rep: str = 'attn', 
              key_added: str = 'steamboat_emb',
              metric='cosine', 
              neighbors_kwargs: dict = None):
    """A thin wrapper for scanpy.pp.neighbors for Steamboat functionalities

    :param adata: AnnData object to be processed
    :param use_rep: embedding to be used, defaults to 'attn'
    :param key_added: key in obsp to store the resulting similarity graph, defaults to 'steamboat_emb'
    :param metric: metric for similarity graph, defaults to 'cosine'
    :param neighbors_kwargs: Other parameters for scanpy.pp.neighbors if desired, defaults to None
    :return: hands over what scanpy.pp.neighbors returns
    """
    if neighbors_kwargs is None:
        neighbors_kwargs = {}
    return sc.pp.neighbors(adata, use_rep=use_rep, key_added=key_added, metric=metric, **neighbors_kwargs)


def leiden(adata: sc.AnnData, resolution: float = 1., *,
            obsp='steamboat_emb_connectivities',
            key_added='steamboat_clusters',
            leiden_kwargs: dict = None):
    """A thin wrapper for scanpy.tl.leiden to cluster for cell types (for spatial domain segmentation, use `segment`).

    :param adata: AnnData object to be processed
    :param resolution: resolution for Leiden clustering, defaults to 1.
    :param obsp: obsp key to be used, defaults to 'steamboat_emb_connectivities'
    :param key_added: obs key to be added for resulting clusters, defaults to 'steamboat_clusters'
    :param leiden_kwargs: Other parameters for scanpy.tl.leiden if desired, defaults to None
    :return: hands over what scanpy.tl.leiden returns
    """
    if leiden_kwargs is None:
        leiden_kwargs = {}
    return sc.tl.leiden(adata, obsp=obsp, key_added=key_added, resolution=resolution, **leiden_kwargs)
    

def segment(adata: sc.AnnData, resolution: float = 1., *,
            key_added: str = 'steamboat_spatial_domain',
            key_added_pairwise: str = 'pairwise',
            key_added_similarity: str = 'similarity', 
            key_added_combined: str = 'combined', 
            n_prop: int = 3,
            spatial_graph_threshold: float = 0.0,
            leiden_kwargs: dict = None):
    """Spatial domain segmentation using Steamboat embeddings and graphs

    :param adata: AnnData object to be processed
    :param resolution: resolution for Leiden clustering, defaults to 1.
    :param key_added: obs key for semgentaiton result, defaults to 'steamboat_spatial_domain'
    :param key_added_pairwise: obsp key for pairwise cell-cell attention graph, defaults to 'pairwise'
    :param key_added_similarity: obsp key for per-cell attention k-NN similarity graph, defaults to 'similarity'
    :param key_added_combined: obsp key for combined pairwise and similarity graphs, defaults to 'combined'
    :param n_prop: power (numbers of propagation) for the pairwise graph, defaults to 3
    :param spatial_graph_threshold: threshold to include/exclude an edge, a larger number will make the program run faster but potentially less accurate, defaults to 0.0
    :param leiden_kwargs: Other parameters for scanpy.tl.leiden if desired, defaults to None
    :return: _descripthands over what scanpy.tl.leiden returnsion_
    """
    if leiden_kwargs is None:
        leiden_kwargs = {}

    adata.obsm['local_attn_std'] = adata.obsm['local_attn'] / adata.obsm['local_attn'].std(axis=0, keepdims=True)
    sc.pp.neighbors(adata, use_rep='local_attn_std', key_added=key_added_similarity, metric='euclidean')

    temp = 0
    j = 0
    while f'local_attn_{j}' in adata.obsp:
        temp += adata.obsp[f'local_attn_{j}']
        j += 1

    temp = temp ** n_prop
    temp = temp.power(1/n_prop)

    temp.data /= temp.data.max()
    temp.data[temp.data < spatial_graph_threshold] = 0
    temp.eliminate_zeros()

    adata.obsp[key_added_pairwise + '_connectivities'] = temp
    adata.obsp[key_added_combined + '_connectivities'] = (adata.obsp[key_added_pairwise + '_connectivities'] + 
                                                          adata.obsp[key_added_similarity + '_connectivities'])
    adata.obsp[key_added_combined + '_connectivities'].eliminate_zeros() 
    return sc.tl.leiden(adata, obsp=key_added_combined + '_connectivities', 
                        key_added=key_added, resolution=resolution, **leiden_kwargs)


def plot_wq(model: Steamboat, chosen_features: List[str], figsize=(3, 3)):
    """Plot the reconstruction metagenes (w_q) only

    :param model: Steamboat model
    :param chosen_features: chosen genes to plot
    :param figsize: (width, height), defaults to (3, 3)
    :return: `fig`, `ax`
    """
    features_mask = [model.features.index(i) for i in chosen_features]

    q = model.spatial_gather.q.weight.detach().cpu().numpy().T
    q = q[features_mask, :]
    q = q / q.max(axis=0)
    head_order = np.argsort(np.argmax(q.T, axis=1) - np.max(q.T, axis=1) / (np.max(q.T) + 1)).tolist()

    common_params = {'linewidths': .05, 'linecolor': 'gray', 'cmap': 'Reds'}
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(q[:, head_order], yticklabels=chosen_features, xticklabels=head_order, square=True, **common_params, ax=ax)
    return fig, ax

plot_vq = plot_wq # quick fix for a typo...


def plot_all_transforms2(model, top: int = 3, reorder: bool = False, 
                    figsize: str | tuple[float, float] = 'auto', 
                    vmin: float = 0., vmax: float = 1.,
                    xticklabels: tuple[str, str, str] = ("environment", "center cell", 'reconstruction')):
    """Plot all metagenes

    :param model: Steamboat model
    :param top: Number of top genes per metagene to plot, defaults to 3
    :param reorder: Reorder the genes by metagene, or keep the orginal ordering, defaults to False
    :param figsize: Size of the figure, defaults to 'auto'
    :param vmin: minimum value in the color bar, defaults to 0.
    :param vmax: maximum value in the color bar, defaults to 1.
    """
    assert model.spatial_gather.n_scales == 2, "This function is only for `scale == 2`. For `scale == 3`, use `plot_all_transforms`."

    n_heads = model.spatial_gather.n_heads

    q = model.spatial_gather.q.weight.detach().cpu()
    k = model.spatial_gather.k_local.weight.detach().cpu()
    v = model.spatial_gather.v.weight.detach().cpu().T
    # switch = model.spatial_gather.switch().detach().cpu()

    if top > 0:
        if reorder:
            rank_v = np.argsort(-v, axis=1)[:, :top]
            rank_q = np.argsort(-q, axis=1)[:, :top]
            rank_k = np.argsort(-k, axis=1)[:, :top]
            feature_mask = {}
            for i in range(n_heads):
                for j in rank_k[i, :]:
                    feature_mask[j] = None
                for j in rank_q[i, :]:
                    feature_mask[j] = None
                for j in rank_v[i, :]:
                    feature_mask[j] = None
            feature_mask = list(feature_mask.keys())
        else:
            rank_v = np.rank(v)
            rank_q = np.rank(q)
            rank_k = np.rank(k)
            max_rank = np.max(np.vstack([rank_v, rank_q, rank_k]), axis=0)
            feature_mask = (max_rank > (max_rank.max() - 3))
            
        chosen_features = np.array(model.features)[feature_mask]
    else:
        feature_mask = list(range(len(model.features)))
        chosen_features = np.array(model.features)

    if figsize == 'auto':
        figsize = (n_heads * 0.49 + 1 + .5, len(chosen_features) * 0.15 + 1.)
    # print(figsize)
    fig, axes = plt.subplots(1, n_heads + 1, sharey='row', sharex='col', figsize=figsize)
    plot_axes = axes
    # bar_axes = axes[0]
    cbar_ax = plot_axes[-1].inset_axes([0.0, 0.1, 1.0, .8])
    common_params = {'linewidths': .05, 'linecolor': 'gray', 'yticklabels': chosen_features, 
                     'cmap': 'Reds', 'cbar_kws': {"orientation": "vertical"}, 'square': True,
                     'vmax': vmax, 'vmin': vmin}

    for i in range(0, n_heads):
        title = ''
        what = f'{i}'
        
        to_plot = np.vstack((k[i, feature_mask],
                             q[i, feature_mask],
                             v[i, feature_mask])).T
        
        true_vmax = to_plot.max(axis=0)
        # print(true_vmax)
        to_plot /= true_vmax
 
        # bar_axes[i].bar(np.arange(len(true_vmax)) + .5, true_vmax)
        # bar_axes[i].set_xticks(np.arange(len(true_vmax)) + .5, [''] * len(true_vmax), rotation=90)
        # bar_axes[i].set_yscale('log')
        # bar_axes[i].set_title(title, size=10, fontweight='bold')
        # if i != 0:
            # bar_axes[i].get_yaxis().set_visible(False)
        # for pos in ['right', 'top', 'left']:
        #     if pos == 'left' and i == 0:
        #         continue
        #     else:
                # bar_axes[i].spines[pos].set_visible(False)
        sns.heatmap(to_plot, xticklabels=xticklabels, ax=plot_axes[i], 
                    **common_params, cbar_ax=cbar_ax)
        plot_axes[i].set_xlabel(f"{what}")
        
    # All text straight up
    for i in range(n_heads):
        plot_axes[i].set_xticklabels(plot_axes[i].get_xticklabels(), rotation=90)

    for i in range(1, n_heads):
        plot_axes[i].get_yaxis().set_visible(False)

    # Remove duplicate cbars
    # bar_axes[-1].set_visible(False)

    plot_axes[-1].get_yaxis().set_visible(False)
    plot_axes[-1].get_xaxis().set_visible(False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plot_axes[-1].spines[pos].set_visible(False)
    # axes[-1].set_visible(False)

    fig.align_xlabels()
    plt.tight_layout()


def plot_all_transforms(model: Steamboat, 
                   top: int = 3, head_order=None,
                   figsize: str | tuple[float, float] = 'auto',
                   chosen_features: List[str] = None):
    """Plot all metagenes for `scale == 3`

    :param model: Steamboat model
    :param top: top genes per metagene to plot, defaults to 3
    :param head_order: order of heads in a list, which can be some or all the heads, defaults to None
    :param figsize: (width, height), defaults to 'auto'
    :param chosen_features: selected features to plot, defaults to None
    """
    assert model.spatial_gather.n_scales == 3, "This function is only for `scale == 3`. For `scale == 2`, use `plot_all_transforms2`."

    if chosen_features is None:
        feature_mask = {}
        for d in head_order if head_order is not None else range(model.spatial_gather.n_heads):
            k1 = model.spatial_gather.k_local.weight[d, :].detach().cpu().numpy()
            k2 = model.spatial_gather.k_regionals[0].weight[d, :].detach().cpu().numpy()
            q = model.spatial_gather.q.weight[d, :].detach().cpu().numpy()
            v = model.spatial_gather.v.weight[:, d].detach().cpu().numpy()
        
            rank_q = np.argsort(-q)[:top]
            rank_k1 = np.argsort(-k1)[:top]
            rank_k2 = np.argsort(-k2)[:top]
            rank_v = np.argsort(-v)[:top]
    
            for j in rank_k1:
                feature_mask[j] = None
            for j in rank_k2:
                feature_mask[j] = None
            for j in rank_q:
                feature_mask[j] = None
            for j in rank_v:
                feature_mask[j] = None
        feature_mask = list(feature_mask.keys())
        chosen_features = np.array(model.features)[feature_mask]
    else:
        feature_mask = []
        for i in chosen_features:
            feature_mask.append(model.features.index(i))
    print(chosen_features)
    
    n_plot_heads = model.spatial_gather.n_heads
    if head_order is not None:
        n_plot_heads = len(head_order)

    if figsize == 'auto':
        figsize = (.7 * (1 + n_plot_heads), len(chosen_features) * 0.15 + .75)
    fig, axes = plt.subplots(1, n_plot_heads + 1, figsize=figsize, sharey='row')

    cbar_ax = axes[-1].inset_axes([0.0, 0.1, .2, .8])
    axes[-1].get_xaxis().set_visible(False)
    for ax in axes[1:]:
        ax.get_yaxis().set_visible(False)
    for pos in ['right', 'top', 'bottom', 'left']:
        axes[-1].spines[pos].set_visible(False)
        
    for i_ax, d in enumerate(head_order if head_order is not None else range(model.spatial_gather.n_heads)):
        k1 = model.spatial_gather.k_local.weight[d, :].detach().cpu().numpy()
        k2 = model.spatial_gather.k_regionals[0].weight[d, :].detach().cpu().numpy()
        q = model.spatial_gather.q.weight[d, :].detach().cpu().numpy()
        v = model.spatial_gather.v.weight[:, d].detach().cpu().numpy()
        
        common_params = {'linewidths': .05, 'linecolor': 'gray', 'yticklabels': chosen_features, 
                         'cmap': 'Reds'}

        to_plot = np.vstack((k2[feature_mask],
                                 k1[feature_mask],
                                 q[feature_mask],
                                 v[feature_mask])).T
        true_vmax = to_plot.max(axis=0)
        # print(true_vmax)
        to_plot /= true_vmax
        
        sns.heatmap(to_plot, xticklabels=['global env', 'local env', 'ego env / center', 'reconstruction'], square=True, 
                    ax=axes[i_ax], **common_params, cbar_ax=cbar_ax)
        axes[i_ax].set_title(d)
        # axes[i_ax].set_xticklabels(['global env', 'local env', 'ego env / center', 'reconstruction'], rotation=45, ha='right', va='center', rotation_mode='anchor')
        # ax.set_xticklabels(plot_axes[i].get_xticklabels(), rotation=0)
        # ax.get_yaxis().set_visible(False)
    
    plt.tight_layout()


def plot_cell_type_enrichment(all_adata, adatas, score_dim, label_key, select_labels=None,
                              figsize=(.75, 4)):
    all_adata.obsm[f'q_{score_dim}'] = np.vstack([i.obsm['q'][:, None, score_dim] for i in adatas])
    all_adata.obsm[f'global_attn_{score_dim}'] = np.vstack([i.obsm['global_attn_0'][:, None, score_dim] for i in adatas])
    all_adata.obsm[f'global_k_0_{score_dim}'] = np.vstack([i.obsm['global_k_0'][:, None, score_dim] for i in adatas])

    cols = [f'q_{score_dim}']
    global_attn_df = pd.DataFrame(all_adata.obsm[f'q_{score_dim}'], 
                                index=all_adata.obs_names, 
                                columns=cols)
    global_attn_df[label_key] = all_adata.obs[label_key]

    cell_median_df = global_attn_df[[label_key] + cols].groupby(label_key).median().astype('float')
    cell_p_df = cell_median_df.copy()
    cell_p_df[:] = 0.
    cell_f_df = cell_p_df.copy()

    for i in cell_p_df.columns:
        for j in cell_p_df.index:
            x = global_attn_df.loc[global_attn_df[label_key] == j, i]
            y = global_attn_df.loc[global_attn_df[label_key] != j, i]
            test_res = sp.stats.mannwhitneyu(x, y)
            cell_p_df.loc[j, i] = test_res.pvalue
            cell_f_df.loc[j, i] = test_res.statistic / len(x) / len(y)

    selected_celltypes = {}
    for i in cell_f_df.columns:
        for j in cell_f_df.sort_values(i, ascending=False).index[:len(cell_f_df)]:
            if j not in ['dirt', 'undefined']:
                selected_celltypes[j] = None
    selected_celltypes = list(selected_celltypes.keys())
    if select_labels is not None:
        selected_celltypes = [i for i in selected_celltypes if i in select_labels]
    
    cols = [f'global_attn_{score_dim}']
    global_attn_df = pd.DataFrame(all_adata.obsm[f'global_attn_{score_dim}'], 
                                index=all_adata.obs_names, 
                                columns=cols)
    global_attn_df[label_key] = all_adata.obs[label_key]

    for i in cols:
        for j in cell_p_df.index:
            x = global_attn_df.loc[global_attn_df[label_key] == j, i]
            y = global_attn_df.loc[global_attn_df[label_key] != j, i]
            test_res = sp.stats.mannwhitneyu(x, y)
            cell_p_df.loc[j, i] = test_res.pvalue
            cell_f_df.loc[j, i] = test_res.statistic / len(x) / len(y)


    cols = [f'global_k_0_{score_dim}']
    global_attn_df = pd.DataFrame(all_adata.obsm[f'global_k_0_{score_dim}'], 
                                index=all_adata.obs_names, 
                                columns=cols)
    global_attn_df[label_key] = all_adata.obs[label_key]

    for i in cols:
        for j in cell_p_df.index:
            x = global_attn_df.loc[global_attn_df[label_key] == j, i]
            y = global_attn_df.loc[global_attn_df[label_key] != j, i]
            test_res = sp.stats.mannwhitneyu(x, y)
            cell_p_df.loc[j, i] = test_res.pvalue
            cell_f_df.loc[j, i] = test_res.statistic / len(x) / len(y)
            
    cell_p_df *= cell_p_df.shape[0]

    common_params = {'linewidths': .05, 'linecolor': 'gray', 'cmap': 'vlag', 'center': .5, 'square': True}

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cell_f_df.loc[selected_celltypes], ax=ax, **common_params)
    # selected_cell_p_df = cell_p_df.loc[selected_celltypes]
    # for i, iv in enumerate(selected_cell_p_df.index):
    #     for j, jv in enumerate(selected_cell_p_df.columns):
    #         text = p2stars(selected_cell_p_df.loc[iv, jv])
    #         ax.text(j + .5, i + .5, text,
    #                 horizontalalignment='center',
    #                 verticalalignment='center',
    #                 c='white', size=8)
    ax.set_xticks([])

    return fig, ax