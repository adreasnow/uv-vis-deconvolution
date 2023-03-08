# Deconvolution toolkit

This is a jupyter notebook and some associated functions to deconvolute UV-Vis and Fluorescence spectra into component Gaussians, either in derivative or non-derivative space.

I would advise against using this notebook in VSCode, it's a bit too hefty to play nicely with their ipython implementation

## Requirements

* python >= 3.10 (it uses native dataclasses)
* numpy
* scipy
* matplotlib
* jupyter lab
* ipywidgets
* ipyfilechooser
* tqdm
* ipympl

These can all be installed via conda or pip

```bash
conda install python numpy scipy matplotlib ipywidgets ipyfilechooser tqdm jupyterlab ipympl
```

or 

```bash
pip install python numpy scipy matplotlib ipywidgets ipyfilechooser tqdm jupyterlab ipympl
```

## Running

Once you have the depemndencies, you should be able to clone it down and run it as you would any jupyter notebook.

```bash
git clone https://github.com/adreasnow/uv-vis-deconvolution.git Downloads/deconvolution
jupyter lab --notebook-dir=Downloads/deconvolution
```

## Example of the interface
(Don't worry, the process for how it works and what each slider does is in the notebook itself)
![](https://raw.githubusercontent.com/adreasnow/uv-vis-deconvolution/main/example.png)
