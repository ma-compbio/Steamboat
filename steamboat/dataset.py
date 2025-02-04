from torch.utils.data import Dataset
import numpy as np
import torch
import squidpy as sq
import scanpy as sc
from tqdm.auto import tqdm
import scipy as sp
from typing import Union
import scipy.sparse
import warnings

class SteamboatDataset(Dataset):
    def __init__(self, data_list, sparse_graph):
        """Steamboat Dataset class

        :param data_list: a list of dictionaries containing 'X' and 'adj' keys
        :param sparse_graph: Whether to use adjacency list or adjacency matrix
        """
        super().__init__()
        self.data = data_list
        self.sparse_graph = sparse_graph

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample['X'], sample['adj'], sample['regional_Xs'], sample['regional_adjs']

    def to(self, device):
        """Send everything to a device. Always copy (even if it's on the device already).
        
        """
        new_data = []
        for sample in self.data:
            new_sample = {}
            new_sample['X'] = sample['X'].to(device)
            new_sample['adj'] = sample['adj'].to(device)
            new_sample['regional_Xs'] = [j.to(device) for j in sample['regional_Xs']]
            new_sample['regional_adjs'] = [j.to(device) for j in sample['regional_adjs']]
            new_data.append(new_sample)
        return SteamboatDataset(new_data, sparse_graph=self.sparse_graph)
    

def prep_adatas(adatas: list[sc.AnnData], n_neighs: int = 8, norm=True, log1p=True, scale=False, renorm=False) -> list[sc.AnnData]:
    """Preprocess a list of AnnData objects

    :param adatas: A list of `SCANPY AnnData`
    :param n_neighs: number of neighbors for kNN spatial graph, defaults to 8
    :param log_norm: Whether or not to normalize and log-transform the data, defaults to True
    :return: A list of preprocessed `SCANPY AnnData`
    """
    with warnings.catch_warnings(action="ignore"):
        warnings.simplefilter("ignore")
        for i in tqdm(range(len(adatas))):
            adata = adatas[i]
            if norm:
                sc.pp.normalize_total(adata)
            if log1p:
                sc.pp.log1p(adata)
            if scale:
                sc.pp.scale(adata, max_value=10)
            if renorm:
                sc.pp.normalize_total(adata, target_sum=100, zero_center=False)
            
            sq.gr.spatial_neighbors(adata, n_neighs=n_neighs)
    return adatas

def make_dataset(adatas: list[sc.AnnData], sparse_graph=True, mask_var: str = None, obsm_key = None,
                 regional_obs: str | list[str] = None) -> SteamboatDataset:
    """Create a PyTorch Dataset from a list of adata
    The input data should be a list of AnnData that contains 1. raw counts or normalized counts
    :param adatas: A list of `SCANPY AnnData`
    :param sparse_graph: Use adjacency list. 
    :param mask_var: Column in `var` to select variables. Default: `obs.highly_variable` if available, otherwise no filtering. Specify `False` to use all genes.
    :return: A `torch.Dataset` including all data.
    """
    # Sanity checks
    if mask_var is None:
        if 'highly_variable' in adatas[0].var.columns:
            print(f"Using {mask_var} to mask variables. Explicitly specify `mask_var=False` to use all genes.")
            mask_var = 'highly_variable'
        else:
            mask_var = False

    if mask_var:
        for i in range(len(adatas)):
            assert mask_var in adatas[i].var.columns, f"Not all adatas have {mask_var} in var"
        temp = adatas[0].var[mask_var]
        for i in range(1, len(adatas)):
            assert (adatas[i].var[mask_var] == temp).all(), f"Not all adatas have {mask_var} in var"

    if regional_obs is None:
        print('No regional annotation, so the dataset will only support ego and local attention. To enalbe regional attention, provide a region_obs.')
    elif isinstance(regional_obs, str):
        print(f'Using {regional_obs} as regional annotation.')
    elif isinstance(regional_obs, list):
        assert all(isinstance(i, str) for i in regional_obs), 'region_obs should be a string or a list of strings.'
        print(f'Using {regional_obs} as regional annotations.')
    else:
        raise ValueError('region_obs should be a string or a list of strings.')

    datasets = []
    unequal_nbs = []

    for i in tqdm(range(len(adatas))):
        adata = adatas[i]
        data_dict = {}

        # Gather expression profile
        if obsm_key is None:
            X = adata.X
        else:
            X = adata.obsm[obsm_key]
        if mask_var:
            X = X[:, adata.var[mask_var]]
        if isinstance(X, sp.sparse.spmatrix):
            data_dict['X'] = torch.from_numpy(X.astype(np.float32).toarray())
        else:
            data_dict['X'] = torch.from_numpy(X.astype(np.float32))
        
        # Gather spatial graph
        if sparse_graph:
            # have_equal_deg = True
            v, u = adata.obsp['spatial_connectivities'].nonzero()
            k0 = u.shape[0] / adata.shape[0]
            k = int(np.round(k0))

            order = np.argsort(v)
            u = u[order]
            v = v[order]

            if np.abs(k - k0) < 1e-6 and (v.reshape([-1, k]) == np.arange(adata.shape[0])[:, None]).all():
                # Case 1: All cells have a equal number of neighbors.
                # E.g., direct result of k-NN graph
                data_dict['adj'] = torch.from_numpy(np.vstack([u, v]))
            else:
                # Case 2: Not all cells have the same number of neighbors.
                # E.g., result of a radius graph, delaunay triangulation, a subgraph of a k-NN graph, etc.
                # We find the cell with highest degree and pad the adjacency matrix with the cell itself.
                # A separate mask is used to indicate the valid neighbors, so that the padding does not affect the computation.
                ks = np.array(adata.obsp['spatial_connectivities'].sum(axis=0)).squeeze().astype(int)
                max_k = int(ks.max())
                unequal_nbs.append(i)
                aligned_u = np.zeros((adata.shape[0], max_k), dtype=int)
                aligned_v = np.zeros((adata.shape[0], max_k), dtype=int)
                align_mask = np.zeros((adata.shape[0], max_k), dtype=int)

                pt = 0
                for i in range(adata.shape[0]):
                    pt2 = pt + ks[i]
                    aligned_u[i, :] = v[pt]
                    aligned_v[i, :] = v[pt]

                    aligned_u[i, :ks[i]] = u[pt:pt2]
                    assert (v[pt:pt2] == i).all()
                    align_mask[i, :ks[i]] = 1
                    pt = pt2
                        
                data_dict['adj'] = torch.from_numpy(np.vstack([aligned_u.flatten(), 
                                                               aligned_v.flatten(), 
                                                               align_mask.flatten()]))

        else:
            data_dict['adj'] = torch.from_numpy((adata.obsp['spatial_connectivities'] == 1).toarray())

        # Placeholders for regional data
        data_dict['regional_Xs'] = []
        data_dict['regional_adjs'] = []
        
        for key in regional_obs:
            unique_values = adata.obs[key].unique()
            if unique_values.shape[0] != 1:
                raise NotImplementedError('Only support one unique value for regional observation.')
            else:
                temp_X = data_dict['X'].mean(axis=0, keepdim=True)
                v = np.arange(adata.shape[0], dtype=int)
                u = np.zeros(adata.shape[0], dtype=int)
                temp_adj = torch.from_numpy(np.vstack([u, v]))

            data_dict['regional_Xs'].append(temp_X)
            data_dict['regional_adjs'].append(temp_adj)

        datasets.append(data_dict)
    
    # Let the user know if not all cells have the same number of neighbors in case something is wrong
    if unequal_nbs:
        print("Not all cells in the following samples have the same number of neighbors:")
        print(*unequal_nbs, sep=', ', end='.\n')
        print("Steamboat can handle this. You can safely ignore this warning if this is expected.")

    return SteamboatDataset(datasets, sparse_graph)


# def make_dataset(adatas: list[sc.AnnData], sparse_graph=True, mask_var=None) -> Dataset:
#     """Create a PyTorch Dataset from a list of adata
#     The input data should be a list of AnnData that contains 1. raw counts or normalized counts
#     :param adatas: A list of `SCANPY AnnData`
#     :param sparse_graph: Use adjacency list. 
#     :param mask_var: Column in `var` to select variables. Default: `obs.highly_variable` if available, otherwise no filtering. Specify `False` to use all genes.
#     :return: A `torch.Dataset` including all data.
#     """
#     # Sanity checks
#     if mask_var is None:
#         if 'highly_variable' in adatas[0].var.columns:
#             print(f"Using {mask_var} to mask variables. Explicitly specify `mask_var=False` to use all genes.")
#             mask_var = 'highly_variable'
#         else:
#             mask_var = False
# 
#     if mask_var:
#         for i in range(len(adatas)):
#             assert mask_var in adatas[i].var.columns, f"Not all adatas have {mask_var} in var"
#         temp = adatas[0].var[mask_var]
#         for i in range(1, len(adatas)):
#             assert (adatas[i].var[mask_var] == temp).all(), f"Not all adatas have {mask_var} in var"
# 
#     datasets = []
#     unequal_nbs = []
# 
#     for i in tqdm(range(len(adatas))):
#         adata = adatas[i]
#         data_dict = {}
# 
#         # Gather expression profile
#         X = adata.X
#         if mask_var:
#             X = X[:, adata.var[mask_var]]
#         if isinstance(adata.X, sp.sparse.spmatrix):
#             data_dict['X'] = torch.from_numpy(X.astype(np.float32).toarray())
#         else:
#             data_dict['X'] = torch.from_numpy(X.astype(np.float32))
#         
#         # Gather spatial graph
#         if sparse_graph:
#             # have_equal_deg = True
#             v, u = adata.obsp['spatial_connectivities'].nonzero()
#             k0 = u.shape[0] / adata.shape[0]
#             k = int(np.round(k0))
# 
#             order = np.argsort(v)
#             u = u[order]
#             v = v[order]
# 
#             if np.abs(k - k0) < 1e-6 and (v.reshape([-1, k]) == np.arange(adata.shape[0])[:, None]).all():
#                 # Case 1: All cells have a equal number of neighbors.
#                 # E.g., direct result of k-NN graph
#                 data_dict['adj'] = torch.from_numpy(np.vstack([u, v]))
#             else:
#                 # Case 2: Not all cells have the same number of neighbors.
#                 # E.g., result of a radius graph, delaunay triangulation, a subgraph of a k-NN graph, etc.
#                 # We find the cell with highest degree and pad the adjacency matrix with the cell itself.
#                 # A separate mask is used to indicate the valid neighbors, so that the padding does not affect the computation.
#                 ks = np.array(adata.obsp['spatial_connectivities'].sum(axis=0)).squeeze().astype(int)
#                 max_k = int(ks.max())
#                 unequal_nbs.append(i)
#                 aligned_u = np.zeros((adata.shape[0], max_k), dtype=int)
#                 aligned_v = np.zeros((adata.shape[0], max_k), dtype=int)
#                 align_mask = np.zeros((adata.shape[0], max_k), dtype=int)
# 
#                 pt = 0
#                 for i in range(adata.shape[0]):
#                     pt2 = pt + ks[i]
#                     aligned_u[i, :] = v[pt]
#                     aligned_v[i, :] = v[pt]
# 
#                     aligned_u[i, :ks[i]] = u[pt:pt2]
#                     assert (v[pt:pt2] == i).all()
#                     align_mask[i, :ks[i]] = 1
#                     pt = pt2
#                         
#                 data_dict['adj'] = torch.from_numpy(np.vstack([aligned_u.flatten(), 
#                                                                aligned_v.flatten(), 
#                                                                align_mask.flatten()]))
# 
#         else:
#             data_dict['adj'] = torch.from_numpy((adata.obsp['spatial_connectivities'] == 1).toarray())
# 
#         datasets.append(data_dict)
#     
#     # Let the user know if not all cells have the same number of neighbors in case something is wrong
#     if unequal_nbs:
#         print("Not all cells in the following samples have the same number of neighbors:")
#         print(*unequal_nbs, sep=', ', end='.\n')
#         print("Steamboat can handle this. You can safely ignore this warning if this is expected.")
# 
#     return SteamboatDataset(datasets, sparse_graph)