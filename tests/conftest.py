# import sys
# sys.path.append('C:/Files/projects/Steamboat/')

import pytest
import scanpy as sc
from pathlib import Path
import torch
import steamboat as sf

@pytest.fixture(scope="session")
def sim_adata():
    test_dir = Path(__file__).parent
    data_path = test_dir / "data" / "simulation.h5ad"
    print(f"Loading data from {data_path}...")
    data = sc.read_h5ad(data_path)
    return data

@pytest.fixture(scope="session")
def sim_answer():
    test_dir = Path(__file__).parent
    data_path = test_dir / "data" / "simulation_result.h5ad"
    print(f"Loading data from {data_path}...")
    data = sc.read_h5ad(data_path)

    model_path = test_dir / "data" / "simulation_model.pth"
    print(f"Loading saved model from {model_path}...")
    model = sf.Steamboat(data.var_names.tolist(), n_heads=5, n_scales=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return model, data

@pytest.fixture(scope="session")
def tonsil_answer():
    test_dir = Path(__file__).parent
    data_path = test_dir / "data" / "tonsil_tiny.h5ad"
    print(f"Loading data from {data_path}...")
    data = sc.read_h5ad(data_path)

    test_dir = Path(__file__).parent
    model_path = test_dir / "data" / "tonsil_model.pth"
    print(f"Loading saved model from {model_path}...")
    model = sf.Steamboat(data.var_names.tolist(), n_heads=60, n_scales=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return model, data