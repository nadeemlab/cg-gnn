# `cg-gnn`

`cg-gnn` (short for "Cell Graph - Graph Neural Networks'') is a library to create cell graphs from pathology slide data and train a graph neural network model using them to predict patient outcomes. This library is designed to be used with and as part of the [SPT framework](https://github.com/nadeemlab/SPT), although independent functionality is also possible provided you can provide formatted, cell level slide data.

This library is a heavily modified version of [histocartography](https://github.com/BiomedSciAI/histocartography) and two of its applications, [hact-net](https://github.com/histocartography/hact-net) and [patho-quant-explainer](https://github.com/histocartography/patho-quant-explainer).

## Getting started

First, use [`spt cggnn extract`](https://github.com/nadeemlab/SPT/tree/main/spatialprofilingtoolbox/cggnn) to fetch pandas HDF files and JSONs that `cg-gnn` can use. Then, install `cg-gnn` using one of the methods below and run `main.py` from the command line or `run_all` from `cggnn/run_all.py`, providing it the paths to the files output by `spt cggnn extract` and your choice of parameters.

### Installation

#### Using pip

In addition to installing via pip,
```
pip install cg-gnn
```
you must also install using the instructions on their websites,
* [pytorch](https://pytorch.org/get-started/locally/)
* [DGL](https://www.dgl.ai/pages/start.html)
* [CUDA](https://anaconda.org/nvidia/cudatoolkit) (optional but highly recommended if your machine supports it)

#### From source

1. Clone this repository
2. Create a conda environment using
```
conda env create -f environment.yml
```
3. Run this module from the command line using `main.py`. Alternatively, scripts in the main directory running from `a` to `d4` allow you to step through each individual section of the `cg-gnn` pipeline, saving files along the way.


## Credits

As mentioned above, this repository is a heavily modified version of [the histocartography project](https://github.com/BiomedSciAI/histocartography) and two of its applications: [hact-net](https://github.com/histocartography/hact-net) and [patho-quant-explainer](https://github.com/histocartography/patho-quant-explainer). Specifically,

* Cell graph formatting, saving, and loading using DGL is patterned on how they were implemented in hact-net
* The neural network training and inference module is modified from the hact-net implementation for cell graphs
* Importance score and separability calculations are sourced from patho-quant-explainer
* The dependence on histocartography is indirect, through the functionality used by the above features

Due to dependency issues that arose when using the version of histocartography published on PyPI, we've chosen to copy and make slight updates to only the modules of histocartography used by the features supported in this library.
