# Steamboat

[![Documentation Status](https://readthedocs.org/projects/steamboat/badge/?version=latest)](https://steamboat.readthedocs.io/en/latest/?badge=latest)

Steamboat is a computational model to decipher short- and long-distance cellular interactions based on the multi-head attention mechanism. 
![fig1-v3-abstract](https://github.com/user-attachments/assets/0fc4cbe1-b43c-48dc-9397-81881d2ecda7)

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

## Examples
A few examples in Jupyter notebook are included in the examples folder: 
- [Illustration (simulated)](https://github.com/ma-compbio/Steamboat/blob/main/examples/Ex0_tiny_simulation.ipynb)
- [Ovarian cancer data](https://github.com/ma-compbio/Steamboat/blob/main/examples/Ex1_hgsc.ipynb)
- [Colorectal cancer data](https://github.com/ma-compbio/Steamboat/blob/main/examples/Ex3_crc.ipynb)

Data used in these examples are available in [Google Drive](https://drive.google.com/drive/folders/1PbLOhYRXp1TKVfPNPWiO4-F3ucsc4u8T?usp=sharing).

## Documentation
For the full API and real data examples, please visit our [documentation](https://steamboat.readthedocs.io/en/latest/).
