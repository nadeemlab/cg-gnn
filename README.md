# `cg-gnn`

`cg-gnn` (short for "Cell Graph - Graph Neural Networks") is a library to train a graph neural network model on graphs built out of cell spatial data to predict patient outcomes or any other y-variable you choose. This library is designed to be used with the [Spatial Profiling Toolbox (SPT)](https://github.com/nadeemlab/SPT), although independent functionality is also possible provided you can provide cell graphs in the same format as SPT [(as implemented in the `graphs` submodule)](https://github.com/nadeemlab/SPT/tree/main/spatialprofilingtoolbox/graphs).

In addition to standalone use, `cg-gnn` also serves as an example implementation of an SPT-compatible graph neural network pipeline, for open source developers to reference when building their own deep learning tools that use cell graphs created by SPT. The key features that have to be implemented are
1. model training and inference
2. cell-level importance score calculation
If the input and output schema is followed, your tool will be compatible with the SPT ecosystem, allowing users to easily integrate your tool into their SPT workflows and upload your model's results to an SPT database.

This library is a heavily modified version of [histocartography](https://github.com/BiomedSciAI/histocartography) and two of its applications, [hact-net](https://github.com/histocartography/hact-net) and [patho-quant-explainer](https://github.com/histocartography/patho-quant-explainer).

## Installation

### Using pip

In addition to installing via pip,
```
pip install cg-gnn
```
you must also install using the instructions on their websites,
* [pytorch](https://pytorch.org/get-started/locally/)
* [DGL](https://www.dgl.ai/pages/start.html)
* [CUDA](https://anaconda.org/nvidia/cudatoolkit) (optional but highly recommended if your machine supports it)

### From source

1. Clone this repository
2. Create a conda environment that can run this software using
```
conda env create -f environment.yml
```

### Docker

For convenience, Dockerized versions of this package are provided at [nadeemlab/spt-cg-gnn](https://hub.docker.com/repository/docker/nadeemlab/spt-cg-gnn/general). We recommend using the CUDA-enabled version, provided it will run on your machine.

## Quickstart

Use [`spt graphs extract` and `spt graphs generate-graphs`](https://github.com/nadeemlab/SPT/tree/main/spatialprofilingtoolbox/graphs) to create cell graphs from a SPT database instance that this python package can use.

This module includes two scripts that you can call from the command line, or you can use the modules directly in Python.
1. `cg-gnn-train` trains a graph neural network model on a set of cell graphs, saves the model to file, and updates the cell graphs it was trained on with cell-level importance-to-classification scores if an explainer model type is provided.
2. `cg-gnn-separability` calculates class separability metrics given a trained model and other metadata.

## Credits

As mentioned above, this repository is a heavily modified version of [the histocartography project](https://github.com/BiomedSciAI/histocartography) and two of its applications: [hact-net](https://github.com/histocartography/hact-net) and [patho-quant-explainer](https://github.com/histocartography/patho-quant-explainer). Specifically,

* Cell graph formatting, saving, and loading using DGL is patterned on how they were implemented in hact-net
* The neural network training and inference module is modified from the hact-net implementation for cell graphs
* Importance score and separability calculations are sourced from patho-quant-explainer
* The dependence on histocartography is indirect, through the functionality used by the above features

Due to dependency issues that arose when using the version of histocartography published on PyPI, we've chosen to copy and make slight updates to only the modules of histocartography used by the features supported in this library.
