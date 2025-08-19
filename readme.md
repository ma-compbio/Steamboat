# Steamboat

[![Documentation Status](https://readthedocs.org/projects/steamboat/badge/?version=latest)](https://steamboat.readthedocs.io/en/latest/?badge=latest)

Steamboat is an interpretable machine learning framework leveraging a self-supervised, multi-head attention model that uniquely decomposes the gene expression of a cell into multiple key factors:
- intrinsic cell programs,
- neighboring cell communication, and 
- long-range interactions.

These pieces of information are used to generate cell embedding, cell network, and reconstructed gene expression.

![fig1-v3-abstract](https://github.com/user-attachments/assets/0fc4cbe1-b43c-48dc-9397-81881d2ecda7)

## System requirements

### Hardware
Steamboat can run on a laptop, desktop, or server. 
The experiments were done on a desktop computer with a 6-core Ryzen 5 3600 CPU and an RTX 3080 GPU. 
A GPU can significantly reduce the time needed to train the models.

### Operating system
Steamboat is python-based and run on all mainsteam operating systems. It has been tested on Windows 10 and Springdale Linux.

### Software dependencies
| Package      | Tested with          |
|--------------|----------------------|
| Python       | 3.11.5               |
| Torch        | 2.1.2 (w/ cuda 12.1) |
| Scanpy       | 1.9.6                |
| Squidpy      | 1.5.0                |
| Scipy        | 1.11.4               |
| Numpy        | 1.26.2               |
| Networkx     | 3.1                  |
| Matplotlib   | 3.8.0                |
| Seaborn      | 0.13.2               |
| Scikit-learn | 1.2.2                |

### Installation
We recommend using [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) to create an virtual environment.
```bash
conda create -n steamboat
conda activate steamboat
```
Please follow the [official guide](https://pytorch.org/get-started/locally/) to install the appropriate Pytorch version for you system and hardware.
Then, please install the required packages with `pip install -r requirements.txt`. 

Steamboat can be imported directly after adding its directory to the path.
```bash
git clone https://github.com/ma-compbio/Steamboat
```
```python
import sys
sys.path.append("/path/of/the/cloned/repository")
```

The installation takes about a half hour.

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
model = model.to("cuda") # if you GPU acceleration is supported.
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

Data used in these examples are available in [Google Drive](https://drive.google.com/drive/folders/1PbLOhYRXp1TKVfPNPWiO4-F3ucsc4u8T?usp=sharing).

## Documentation
For the full API and real data examples, please visit our [documentation](https://steamboat.readthedocs.io/en/latest/).
