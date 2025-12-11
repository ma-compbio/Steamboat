import sys
sys.path.append('C:/Files/projects/Steamboat/')
import pandas as pd
import numpy as np
import scipy as sp
import scanpy as sc
import squidpy as sq
import steamboat as sf # Steamboat Factorization -> sf
import steamboat.tools
from pathlib import Path

def test_preproc():
    test_dir = Path(__file__).parent
    adata = sc.read_h5ad(test_dir / "data/simulation.h5ad")
    adata.obs['global'] = 0
    adatas = [adata.copy() for _ in range(2)]
    adatas = sf.prep_adatas(adatas, n_neighs=8)
    dataset = sf.make_dataset(adatas, regional_obs=['global'])

    assert len(dataset) == 2, "The length of the dataset should be 2."
    assert (adatas[0].obsp['spatial_connectivities'].sum(axis=1) == 8).all(), "Each cell should have 8 neighbors in the connectivity matrix."

