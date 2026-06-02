# %%
from dance.transforms import Compose
from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.spagcn import SpaGCN
from dance.utils import set_seed
from dance.transforms import CellPCA, Compose, FilterGenesMatch, SetConfig
from dance.transforms.graph import SpaGCNGraph, SpaGCNGraph2D

# %%
import scanpy as sc
import numpy as np
import gc
import pandas as pd

# %%
from dance.modules.spatial.spatial_domain.stagate import Stagate

# %%
import sklearn.metrics

# %%
from tqdm import tqdm

# %%
from dance import logger
logger.setLevel("ERROR")

# %%
from dance.transforms import AnnDataTransform, Compose, SetConfig
from dance.transforms.graph import StagateGraph
from dance.typing import Any, LogLevel, Optional, Tuple

def preprocessing_pipeline(model_name: str = "knn", radius: float = 150, n_neighbors: int = 8, 
                           log_level: LogLevel = "ERROR"):
        return Compose(
            AnnDataTransform(sc.pp.normalize_total, target_sum=1e4),
            AnnDataTransform(sc.pp.log1p),
            StagateGraph(model_name, radius=radius, n_neighbors=n_neighbors),
            SetConfig({
                "feature_channel": "StagateGraph",
                "feature_channel_type": "obsp",
                "label_channel": "label",
                "label_channel_type": "obs"
            }),
            log_level=log_level,
        )

preprocessing_pipeline = preprocessing_pipeline()

# %%


# %%
import time


# %%
from dance.datasets.base import Data
aris = []
nmis = []
t0 = time.time()
for i in range(82, 129):
    adata = sc.read_h5ad(f"../../SpaceFormer/experiments/backup/mmbrain/mmbrain_raw_{i}.h5ad")
    pixel_factor = 80 / ((adata.obsm['spatial'][:, 0].max() - adata.obsm['spatial'][:, 0].min()) * 
           (adata.obsm['spatial'][:, 1].max() - adata.obsm['spatial'][:, 1].min()) 
           / adata.shape[0]) ** .5
    adata.obsm['spatial_pixel'] = adata.obsm['spatial'] * pixel_factor 
    adata.obs['label'] = adata.obs['parcellation_division']
    
    data = Data(adata)
    preprocessing_pipeline(data)

    adj, y = data.get_data(return_type="default")
    x = data.data.X
    edge_list_array = np.vstack(np.nonzero(adj))

    model = Stagate(hidden_dims=[x.shape[1], 512, 32])
    pred = model.fit_predict((x, edge_list_array), 1000, num_cluster=adata.obs['parcellation_division'].nunique(), random_state=0)
    adata.obs['stagate'] = pred
    adata.obs[['stagate']].to_csv(f"../../SpaceFormer/experiments/backup/mmbrain/stagate_spatial_domain_{i}.csv")

    ari = sklearn.metrics.adjusted_rand_score(adata.obs['parcellation_division'], adata.obs['stagate'])
    nmi = sklearn.metrics.normalized_mutual_info_score(adata.obs['parcellation_division'], adata.obs['stagate'])

    print(i, adata.obs['stagate'].nunique(), ari, nmi, f"{(time.time() - t0) / 60:.2f} minutes")
    aris.append(ari)
    nmis.append(nmi)

    del adata
    gc.collect()

tt = time.time() - t0

# %%
aris = []
nmis = []
for i in tqdm(range(129)):
    y = pd.read_csv(f"../../SpaceFormer/experiments/backup/mmbrain/stagate_spatial_domain_{i}.csv", index_col=0)
    x = pd.read_csv(f"../../SpaceFormer/experiments/backup/mmbrain/groundtruth_spatial_domain_{i}.csv", index_col=0)
    ari = sklearn.metrics.adjusted_rand_score(x['parcellation_division'], y['stagate'])
    nmi = sklearn.metrics.normalized_mutual_info_score(x['parcellation_division'], y['stagate'])
    aris.append(ari)
    nmis.append(nmi)

# %%
df = pd.DataFrame({'ARI': aris, 'NMI': nmis})
df.to_csv("../../SpaceFormer/experiments/backup/mmbrain/stagate_spatial_domain.csv")

df

# %%
## 0 - 81 151.33 minutes
## 82 - 128 73.53 minutes


