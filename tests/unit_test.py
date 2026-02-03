# import sys
# sys.path.append('C:/Files/projects/Steamboat/')

import pandas as pd
import numpy as np
import scipy as sp
import scanpy as sc
import squidpy as sq
import steamboat as sf # Steamboat Factorization -> sf
import steamboat.tools
from pathlib import Path

def test_preproc(sim_adata):
    adata = sim_adata.copy()
    adata.obs['global'] = 0
    adatas = [adata.copy() for _ in range(2)]
    adatas = sf.prep_adatas(adatas, n_neighs=8)
    dataset = sf.make_dataset(adatas, regional_obs=['global'])

    assert len(dataset) == 2, "The length of the dataset should be 2."
    assert (adatas[0].obsp['spatial_connectivities'].sum(axis=1) == 8).all(), "Each cell should have 8 neighbors in the connectivity matrix."


def test_prediction(sim_answer):
    """Given the saved model and result, test whether the prediction is correct."""
    model, adata = sim_answer
    adata_answer = adata.copy()
    adata = adata.copy()

    adatas = [adata.copy() for _ in range(2)]
    adatas = sf.prep_adatas(adatas, n_neighs=8, norm=False, log1p=False)
    dataset = sf.make_dataset(adatas, regional_obs=[])

    sf.tools.calc_obs(adatas, dataset, model, get_recon=True, device='cpu')
    for k in adata.obsm:
        assert np.allclose(adata.obsm[k], adata_answer.obsm[k]), f"The values in obsm['{k}'] is wrong."


def test_clustering_and_segmentation(sim_answer):
    """Clustering and segmentation have randomness, so we only test whether the functions run and add the expected keys."""
    _, adata = sim_answer
    adata = adata.copy()

    sf.tools.neighbors(adata)
    assert 'steamboat_emb_connectivities' in adata.obsp, "Neighbors calculation failed to add 'steamboat_emb_connectivities' to obsp."

    sf.tools.leiden(adata)
    assert 'steamboat_clusters' in adata.obs, "Leiden clustering failed to add 'steamboat_clusters' to obs."

    sf.tools.segment(adata)
    assert 'steamboat_spatial_domain' in adata.obs, "Segmentation failed to add 'steamboat_spatial_domain' to obs."
    

def test_read_lrdb():
    """Test reading ligand-receptor database for both human and mouse."""
    lrdb_human = sf.tools.read_lrdb('human')
    lrdb_mouse = sf.tools.read_lrdb('mouse')

    assert not lrdb_human.empty, "Human LRDB should not be empty."
    assert not lrdb_mouse.empty, "Mouse LRDB should not be empty."


def test_contribution(tonsil_answer):
    model, adata = tonsil_answer
    adata = adata.copy()
    adata.obs['global'] = 0
    adatas = sf.prep_adatas([adata], norm=False, log1p=False, n_neighs=8)
    dataset = sf.make_dataset(adatas, sparse_graph=True, regional_obs=['global'])
    sf.tools.calc_obs(adatas, dataset, model, get_recon=True, device='cpu')
    sf.tools.gather_obs(adata, adatas)

    try:
        gene_scale, scale = sf.tools.contribution_by_scale(model, dataset, adatas, 'cpu')
    except Exception as e:
        raise AssertionError("Calculating gene contribution by scale failed.") from e
    
    try: 
        gene_scale, scale = sf.tools.contribution_by_scale_and_head(model, dataset, adatas, 'cpu')
    except Exception as e:
        raise AssertionError("Calculating gene contribution by scale and head failed.") from e
    

def test_cci(tonsil_answer):
    model, adata = tonsil_answer
    adata = adata.copy()
    adata.obs['global'] = 0
    adatas = sf.prep_adatas([adata], norm=False, log1p=False, n_neighs=8)
    dataset = sf.make_dataset(adatas, sparse_graph=True, regional_obs=['global'])
    sf.tools.calc_obs(adatas, dataset, model, get_recon=True, device='cpu')
    sf.tools.gather_obs(adata, adatas)

    try:
        cci_matrix = sf.tools.calc_interaction(adatas, model, 'biosample_id', 'cluster')['tonsil1']
    except Exception as e:
        raise AssertionError("Calculating cell-cell interaction failed.") from e
    

def test_lr(tonsil_answer):
    model, adata = tonsil_answer
    adata = adata.copy()
    adata.obs['global'] = 0
    adatas = sf.prep_adatas([adata], norm=False, log1p=False, n_neighs=8)
    dataset = sf.make_dataset(adatas, sparse_graph=True, regional_obs=['global'])
    sf.tools.calc_obs(adatas, dataset, model, get_recon=True, device='cpu')
    sf.tools.gather_obs(adata, adatas)
    
    try:
        lrp_dfs = sf.tools.score_lrs(adata, model, None, gene_names='index')
    except Exception as e:
        raise AssertionError("Scoring ligand-receptor pairs failed.") from e
    