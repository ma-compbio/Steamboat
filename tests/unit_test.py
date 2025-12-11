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
    model, adata = sim_answer
    adata_answer = adata.copy()
    adata = adata.copy()

    adatas = [adata.copy() for _ in range(2)]
    adatas = sf.prep_adatas(adatas, n_neighs=8, norm=False, log1p=False)
    dataset = sf.make_dataset(adatas, regional_obs=[])

    sf.tools.calc_obs(adatas, dataset, model, get_recon=True, device='cpu')
    for k in adata.obsm:
        assert np.allclose(adata.obsm[k], adata_answer.obsm[k]), f"The values in obsm['{k}'] is wrong."
    