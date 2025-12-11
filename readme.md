# Steamboat

[![PyPI - Version](https://img.shields.io/pypi/v/steamboat-bio)](https://pypi.org/project/steamboat-bio/) [![Documentation Status](https://readthedocs.org/projects/steamboat/badge/?version=latest)](https://steamboat.readthedocs.io/en/latest/?badge=latest) [![Unit and integration tests](https://github.com/ma-compbio/Steamboat/actions/workflows/tests.yml/badge.svg)](https://github.com/ma-compbio/Steamboat/actions/workflows/tests.yml)

Steamboat is an interpretable machine learning framework leveraging a self-supervised, multi-head attention model that uniquely decomposes the gene expression of a cell into multiple key factors:
- intrinsic cell programs,
- neighboring cell communication, and 
- long-range interactions.

These pieces of information are used to generate cell embedding, cell network, and reconstructed gene expression.

![fig1-v3-abstract](https://github.com/user-attachments/assets/0fc4cbe1-b43c-48dc-9397-81881d2ecda7)

## System requirements

### Hardware
Steamboat can run on a laptop, desktop, or server. 
The experiments were done on a desktop computer with an RTX 3080 GPU (10GB VRAM). 
A GPU can significantly reduce the time needed to train the models (more than 5x on the most demanding examples).

### Operating system
Steamboat is python-based and run on all mainsteam operating systems. It has been tested on Windows 10 and Springdale Linux.

### Software dependencies

<details>

<summary>Lastest tested working dependency combination</summary>

| Package      | Tested in 06/2025    | Tested in 12/2025    |
|--------------|----------------------|----------------------|
| Python       | 3.11.5               | 3.19.9               |
| Torch        | 2.1.2 (w/ cuda 12.1) | 2.9.1 (w/ cuda 13.0) |
| Scanpy       | 1.9.6                | 1.11.5               |
| Squidpy      | 1.5.0                | 1.6.6                |
| Scipy        | 1.11.4               | 1.16.3               |
| Numpy        | 1.26.2               | 2.3.1                |
| Networkx     | 3.1                  | 3.5                  |
| Matplotlib   | 3.8.0                | 3.10.6               |
| Seaborn      | 0.13.2               | 0.13.2               |
| Scikit-learn | 1.2.2                | 1.7.2                |

</details>

## Installation
We recommend using [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) to create a virtual environment.
```bash
conda create -n steamboat python=3.13
conda activate steamboat
pip install steamboat-bio
```
Installation usually takes about 2 minutes, but can vary depending on computer and network conditions.

<details>

<summary>Tips for GPU support</summary>

Before running `pip install steamboat-bio`, follow the [official guide](https://pytorch.org/get-started/locally/) to install the appropriate Pytorch version for your system and hardware.
The exact commands will depend on your hardware and system. In general, they look like this.

```bash
conda create -n steamboat python=3.13
conda activate steamboat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 # DO NOT RUN. ADJUST IT FOR YOUR SYSTEM.
pip install steamboat-bio
```

</details>

<details>

<summary>Run without installation</summary>

If you are interested in modifying the package, or just don't feel like installing it, Steamboat can be imported directly after adding its directory to the path.
```bash
git clone https://github.com/ma-compbio/Steamboat
```
```python
import sys
sys.path.append("/path/of/the/cloned/repository")
```

You may need to install the dependencies listed in `requirements.txt` manually.

</details>

## Basic workflow
```python
import steamboat as sf # "sf" = "Steamboat Factorization"
import steamboat.tools
```

First, make a list (`adatas`) of one or more `AnnData` objects, and preprocess them.
```python
adatas = sf.prep_adatas(adatas, log_norm=True)
dataset = sf.make_dataset(adatas)
```

Create a `Steamboat` model and fit it to the data.
```python
model = sf.Steamboat(short_features, n_heads=10, n_scales=3)
model = model.to("cuda") # if GPU acceleration is supported.
model.fit(dataset)
```

After training, you can check the trained metagenes.
```python
sf.tools.plot_all_transforms(model, top=1)
```

For clustering and segmentation, run the following lines. Change the resolution to your liking.
```python
sf.tools.neighbors(adata)
sf.tools.leiden(adata, resolution=0.1)
sf.tools.segment(adata, resolution=0.5)
```

## Demos
A few examples in Jupyter notebook are included in the examples folder: 
1. [Illustration (simulated)](https://github.com/ma-compbio/Steamboat/blob/main/examples/Ex0_tiny_simulation.ipynb)
2. [Ovarian cancer data](https://github.com/ma-compbio/Steamboat/blob/main/examples/Ex1_hgsc.ipynb)
3. Mouse brain
   - [Training](https://github.com/ma-compbio/Steamboat/blob/main/examples/Ex2_mouse_brain_train.ipynb)
   - [Interpretation of results, such as clustering, segmentation, global attention explanation, and ligand-receptor analysis](https://github.com/ma-compbio/Steamboat/blob/main/examples/Ex2_mouse_brain_interpretation.ipynb)
   - [Spatial perturbation, including cell transplant and environmental knock-out](https://github.com/ma-compbio/Steamboat/blob/main/examples/Ex2_mouse_brain_spatial_perturbation.ipynb)
4. [Colorectal cancer data](https://github.com/ma-compbio/Steamboat/blob/main/examples/Ex3_crc.ipynb)

The simulation demo takes about five minutes to run. The mouse brain data takes one hour to train. Other demos take about ten minutes each.

Data used in these examples are available in [Google Drive](https://drive.google.com/drive/folders/1PbLOhYRXp1TKVfPNPWiO4-F3ucsc4u8T?usp=sharing). Trained models are also uploaded.

## Documentation
For the full API and real data examples, please visit our [documentation](https://steamboat.readthedocs.io/en/latest/).
