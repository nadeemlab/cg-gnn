<p align="center">
  <img src="https://raw.githubusercontent.com/histocartography/histocartography/main/docs/_static/logo_large.png" height="200">
</p>

[![Build Status](https://travis-ci.com/histocartography/histocartography.svg?branch=main)](https://travis-ci.com/histocartography/histocartography)
[![codecov](https://codecov.io/gh/histocartography/histocartography/branch/main/graph/badge.svg?token=OILOGEBP0Q)](https://codecov.io/gh/histocartography/histocartography)
[![PyPI version](https://badge.fury.io/py/histocartography.svg)](https://badge.fury.io/py/histocartography)
![GitHub](https://img.shields.io/github/license/histocartography/histocartography)
[![Downloads](https://pepy.tech/badge/histocartography)](https://pepy.tech/project/histocartography)

**[Documentation](https://histocartography.github.io/histocartography/)**
| **[Paper](https://arxiv.org/pdf/2107.10073.pdf)** 

**Welcome to the `histocartography` repository!** `histocartography` is a python-based library designed to facilitate the development of graph-based computational pathology pipelines. The library includes plug-and-play modules to perform,
- standard histology image pre-processing (e.g., *stain normalization*, *nuclei detection*, *tissue detection*)
- entity-graph representation building (e.g. *cell graph*, *tissue graph*, *hierarchical graph*)
- modeling Graph Neural Networks (e.g. *GIN*, *PNA*)
- feature attribution based graph interpretability techniques (e.g. *GraphGradCAM*, *GraphGradCAM++*, *GNNExplainer*)
- visualization tools 

All the functionalities are grouped under a user-friendly API. 

If you encounter any issue or have questions regarding the library, feel free to [open a GitHub issue](add_link). We'll do our best to address it. 

# Installation 

## PyPI installer (recommended)

`pip install histocartography`

## Development setup 

- Clone the repo:

```
git clone https://github.com/histocartography/histocartography.git && cd histocartography
```

- Create a conda environment:

```
conda env create -f environment.yml
```

- Activate it:

```
conda activate histocartography
```

- Add `histocartography` to your python path:

```
export PYTHONPATH="<PATH>/histocartography:$PYTHONPATH"
```

## Tests

To ensure proper installation, run unit tests as:

```sh 
python -m unittest discover -s test -p "test_*" -v
```

Running tests on cpu can take up to 20mn. 

# Using histocartography 

The `histocartography` library provides a set of helpers grouped in different modules, namely `preprocessing`, `ml`, `visualization` and `interpretability`.  

For instance, in `histocartography.preprocessing`, building a cell-graph from an H&E image is as simple as:

```
>> from histocartography.preprocessing import NucleiExtractor, DeepFeatureExtractor, KNNGraphBuilder
>> 
>> nuclei_detector = NucleiExtractor()
>> feature_extractor = DeepFeatureExtractor(architecture='resnet34', patch_size=72)
>> knn_graph_builder = KNNGraphBuilder(k=5, thresh=50, add_loc_feats=True)
>>
>> image = np.array(Image.open('docs/_static/283_dcis_4.png'))
>> nuclei_map, _ = nuclei_detector.process(image)
>> features = feature_extractor.process(image, nuclei_map)
>> cell_graph = knn_graph_builder.process(nuclei_map, features)
```

The output can be then visualized with:

```
>> from histocartography.visualization import OverlayGraphVisualization, InstanceImageVisualization

>> visualizer = OverlayGraphVisualization(
...     instance_visualizer=InstanceImageVisualization(
...         instance_style="filled+outline"
...     )
... )
>> viz_cg = visualizer.process(
...     canvas=image,
...     graph=cell_graph,
...     instance_map=nuclei_map
... )
>> viz_cg.show()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/histocartography/histocartography/main/docs/_static/283_dcis_4_cg.png"  height="400">
</p>

A list of examples to discover the capabilities of the `histocartography` library is provided in `examples`. The examples will show you how to perform:

- **stain normalization** with Vahadane or Macenko algorithm
- **cell graph generation** to transform an H&E image into a graph-based representation where nodes encode nuclei and edges nuclei-nuclei interactions. It includes: nuclei detection based on HoverNet pretrained on PanNuke dataset, deep feature extraction and kNN graph building. 
- **tissue graph generation** to transform an H&E image into a graph-based representation where nodes encode tissue regions and edges tissue-to-tissue interactions. It includes: tissue detection based on superpixels, deep feature extraction and RAG graph building. 
- **feature cube extraction** to extract deep representations of individual patches depicting the image
- **cell graph explainer** to generate an explanation to highlight salient nodes. It includes inference on a pretrained CG-GNN model followed by GraphGradCAM explainer. 

A tutorial with detailed descriptions and visualizations of some of the main functionalities is provided [here](https://github.com/maragraziani/interpretAI_DigiPath/blob/feature/handson2%2Fpus/hands-on-session-2/hands-on-session-2.ipynb) as a notebook. 

# External Ressources 

## Learn more about GNNs 

- We have prepared a gentle introduction to Graph Neural Networks. In this tutorial, you can find [slides](https://github.com/guillaumejaume/tuto-dl-on-graphs/blob/main/slides/ml-on-graphs-tutorial.pptx), [notebooks](https://github.com/guillaumejaume/tuto-dl-on-graphs/tree/main/notebooks) and a set of [reference papers](https://github.com/guillaumejaume/tuto-dl-on-graphs).
- For those of you interested in exploring Graph Neural Networks in depth, please refer to [this content](https://github.com/guillaumejaume/graph-neural-networks-roadmap) or [this one](https://github.com/thunlp/GNNPapers).


## Papers already using this library

- Hierarchical Graph Representations for Digital Pathology, Pati et al., Medical Image Analysis, 2021. [[pdf]](https://arxiv.org/abs/2102.11057) [[code]](https://github.com/histocartography/hact-net) 
- Quantifying Explainers of Graph Neural Networks in Computational Pathology,  Jaume et al., CVPR, 2021. [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Jaume_Quantifying_Explainers_of_Graph_Neural_Networks_in_Computational_Pathology_CVPR_2021_paper.pdf) [[code]](https://github.com/histocartography/patho-quant-explainer) 
- Learning Whole-Slide Segmentation from Inexact and Incomplete Labels using Tissue Graphs, Anklin et al., MICCAI, 2021. [[pdf]](https://arxiv.org/abs/2103.03129) [[code]](https://github.com/histocartography/seg-gini) 

If you use this library, please consider citing:

```
@inproceedings{jaume2021,
    title = {HistoCartography: A Toolkit for Graph Analytics in Digital Pathology},
    author = {Guillaume Jaume, Pushpak Pati, Valentin Anklin, Antonio Foncubierta, Maria Gabrani},
    booktitle={MICCAI Workshop on Computational Pathology},
    pages={117--128},
    year = {2021}
} 

@inproceedings{pati2021,
    title = {Hierarchical Graph Representations for Digital Pathology},
    author = {Pushpak Pati, Guillaume Jaume, Antonio Foncubierta, Florinda Feroce, Anna Maria Anniciello, Giosuè Scognamiglio, Nadia Brancati, Maryse Fiche, Estelle Dubruc, Daniel Riccio, Maurizio Di Bonito, Giuseppe De Pietro, Gerardo Botti, Jean-Philippe Thiran, Maria Frucci, Orcun Goksel, Maria Gabrani},
    booktitle = {Medical Image Analysis (MedIA)},
    volume={75},
    pages={102264},
    year = {2021}
} 
```



# Hierarchical Graph Representations in Digital Pathology

This repository contains the code to reproduce results of the [Hierarchical Graph Representations in Digital Pathology](https://arxiv.org/pdf/2102.11057.pdf) paper. 

The code mostly relies on the [`histocartography`](https://github.com/histocartography/histocartography) library, a python-based package for modeling and learning with graphs of pathology images. 

All the experiments are based on the BRACS dataset. The data needs to be downloaded separately (see Installation steps). 

![Overview of the proposed approach.](figs/readme_fig1.png)

## Installation 

### Cloning and handling dependencies 

Clone the repo:

```
git clone https://github.com/histocartography/hact-net.git && cd hact-net
```

Create a conda environment and activate it:

```
conda env create -f environment.yml
conda activate hactnet
```

### Downloading the BRACS dataset 

BRACS is a dataset of Hematoxylin and Eosin (H&E) histopathological images for automated detection/classification of breast tumors. BRACS includes >4k tumor regions-of-interest labeled in 7 categories (Normal, Benign, UDH, ADH, FEA, DCIS, Invasive). 

In order to download the BRACS dataset, you need to create an account [there](https://www.bracs.icar.cnr.it/). Then, go to `Data Collection`, `Download`, and hit the `Regions of Interest Set` button to access the data. Download the `previous_version` data. The data are stored on an FTP server. 

## Running the code 

The proposed HACT-Net architecture operates on a HieArchical Cell-to-Tissue representation that is further processed by a Graph Neural Network. Running HACT-Net requires 2 steps:

### Step 1: HieArchical Cell-to-Tissue (HACT) generation 

The HACT representation can be generated for the `train` set by running: 

```
cd core
python generate_hact_graphs.py --image_path <PATH-TO-BRACS>/BRACS/train/ --save_path <SOME-SAVE-PATH>/hact-net-data
```

For generating HACT on the `test` and `val` set, simply replace the `image_path` by `<PATH-TO-BRACS>/BRACS/val/` or `<PATH-TO-BRACS>/BRACS/test/`. 

The script will automatically create three directories containing for each image:
- a cell graph as a `.bin` file
- a tissue graph as a `.bin` file
- an assignment matrix as an `.h5` file

After the generation of HACT graphs on the whole BRACS set, the `hact-net-data` dir should look like:

```
hact-net-data
|
|__ cell_graphs 
    |
    |__ train
    |
    |__ test
    |
    |__ val
|
|__ tissue_graphs
    |
    |__ train
    |
    |__ test
    |
    |__ val
|
|__ assignment_matrices 
    |
    |__ train
    |
    |__ test
    |
    |__ val
```

### Step 2: Training HACTNet 

We provide the option to train 3 types of models, namely a Cell Graph model, Tissue Graph model and HACTNet model. 


Training HACTNet as:

```
python train.py --cg_path <SOME-SAVE-PATH>/hact-net-data/cell_graphs/ --tg_path <SOME-SAVE-PATH>/hact-net-data/tissue_graphs/ --assign_mat_path <SOME-SAVE-PATH>/hact-net-data/assignment_matrices/  --config_fpath ../data/config/hact_bracs_hactnet_7_classes_pna.yml -b 8 --in_ram --epochs 60 -l 0.0005 
```


Training a Cell Graph model as:

```
python train.py --cg_path <SOME-SAVE-PATH>/hact-net-data/cell_graphs/ --config_fpath ../data/config/cg_bracs_cggnn_7_classes_pna.yml -b 8 --in_ram --epochs 60 -l 0.0005 

```

Training a Tissue Graph model as:

```
python train.py --tg_path <SOME-SAVE-PATH>/hact-net-data/tissue_graphs/ --config_fpath ../data/config/tg_bracs_tggnn_7_classes_pna.yml -b 8 --in_ram --epochs 60 -l 0.0005 

```

Usage is:

```
usage: train.py [-h] [--cg_path CG_PATH] [--tg_path TG_PATH]
                [--assign_mat_path ASSIGN_MAT_PATH] [-conf CONFIG_FPATH]
                [--model_path MODEL_PATH] [--in_ram] [-b BATCH_SIZE]
                [--epochs EPOCHS] [-l LEARNING_RATE] [--out_path OUT_PATH]
                [--logger LOGGER]

optional arguments:
  -h, --help            show this help message and exit
  --cg_path CG_PATH     path to the cell graphs.
  --tg_path TG_PATH     path to tissue graphs.
  --assign_mat_path ASSIGN_MAT_PATH
                        path to the assignment matrices.
  -conf CONFIG_FPATH, --config_fpath CONFIG_FPATH
                        path to the config file.
  --model_path MODEL_PATH
                        path to where the model is saved.
  --in_ram              if the data should be stored in RAM.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size.
  --epochs EPOCHS       epochs.
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate.
  --out_path OUT_PATH   path to where the output data are saved (currently
                        only for the interpretability).
  --logger LOGGER       Logger type. Options are "mlflow" or "none"
```

The output of this script will be a directory containing three models corresponding to the best validation loss, validation accuracy and weighted F1-score. 

### (Step 3: Inference on HACTNet)

We also provide a script for running inference with the option to use a pretrained model.

For instance, running inference with a pretrained HACTNet model: 

```
python inference.py --cg_path <SOME-SAVE-PATH>/hact-net-data/cell_graphs/ --tg_path <SOME-SAVE-PATH>/hact-net-data/tissue_graphs/ --assign_mat_path <SOME-SAVE-PATH>/hact-net-data/assignment_matrices/  --config_fpath ../data/config/hact_bracs_hactnet_7_classes_pna.yml --pretrained
```

We provide 3 pretrained checkpoints performing as:

| Model | Accuracy | Weighted F1-score |
| ----- |:--------:|:-----------------:|
| Cell Graph Model   | 58.1 | 56.7 |
| Tissue Graph Model | 58.6 | 57.8 |
| HACTNet Model      | 61.7   | 61.5 |


If you use this code, please consider citing our work:

```
@inproceedings{pati2021,
    title = "Hierarchical Graph Representations in Digital Pathology",
    author = "Pushpak Pati, Guillaume Jaume, Antonio Foncubierta, Florinda Feroce, Anna Maria Anniciello, Giosuè Scognamiglio, Nadia Brancati, Maryse Fiche, Estelle Dubruc, Daniel Riccio, Maurizio Di Bonito, Giuseppe De Pietro, Gerardo Botti, Jean-Philippe Thiran, Maria Frucci, Orcun Goksel, Maria Gabrani",
    booktitle = "arXiv",
    url = "https://arxiv.org/abs/2102.11057",
    year = "2021"
} 
```


# Quantifying Explainers of Graph Neural Networks in Computational Pathology

This repository includes the code for replicating results presented in the paper [Quantifying Explainers of Graph Neural Networks in Computational Pathology](https://arxiv.org/pdf/2011.12646.pdf) presented at CVPR 2021.  

The code mostly relies on the [`histocartography`](https://github.com/histocartography/histocartography) library, a python-based package for modeling and learning with graph-based representations of pathology images. 

All the experiments are based on the [`BRACS`](https://www.bracs.icar.cnr.it/) dataset. The data needs to be downloaded separately (see Installation steps). 

![Overview of the proposed approach.](figs/readme_fig1.png)


## Installation 

### Cloning and handling dependencies 

Clone the repo:

```
git clone https://github.com/histocartography/patho-quant-explainer.git && cd patho-quant-explainer
```

Create a conda environment and activate it:

```
conda env create -f environment.yml
conda activate pathoexplainer
```

Make sure that you have the latest version of histocartography, `histocartography==0.2`.

### Downloading the BRACS dataset 

BRACS is a dataset of Hematoxylin and Eosin (H&E) histopathological images for automated detection/classification of breast tumors. BRACS includes >4k tumor regions-of-interest labeled in 7 categories (Normal, Benign, UDH, ADH, FEA, DCIS, Invasive). 

In order to download the BRACS dataset, you need to create an account [here](https://www.bracs.icar.cnr.it/). Then, go to `Data Collection`, `Download`, and hit the `Regions of Interest Set` button to access the data. Download the `v1` data. The data are stored on an FTP server. 

## Running the code 

The approach for explainability of histology images is based on 3 steps: cell graph generation, post-hoc graph explaination to identify salient nodes and the proposed quantitative analysis. 

### Step 1: Cell graph generation 

Each image needs to be transformed into a cell graph where nodes represent nuclei and edges nuclei-nuclei interactions. The cell graph for the BRACS test set can be generated with: 

```
cd core
python generate_cell_graphs.py --data_path <PATH-TO-BRACS>/BRACS/test/ --save_path <SOME-SAVE-PATH>/quant-gnn-explainers-data
```

The script will automatically create a directory containing cell graphs as a `.bin` file for each image. There should be 626 files created. 

### Step 2: Explaining the cell graphs

We benchmark 4 different explainers: GraphLRP, GNNExplainer, GraphGradCAM and GraphGradCAM++, that returns a different explanation, i.e., nuclei-level importance scores, for each cell graph. The system will automatically download a pre-trained checkpoint trained on 3-class BRACS using the `histocartography` library. This model reaches 74% accuracy on the test set. 

Generating explanation with:

```
python generate_explanations --cell_graphs <SOME-SAVE-PATH>/quant-gnn-explainers-data/cell_graphs --explainer graphgradcam --save_path <SOME-SAVE-PATH>/quant-gnn-explainers-data/graphgradcam
```

The explainer type can be set to either: `graphgradcam`, `graphgradcampp`, `gnnexplainer` or `graphlrp`. 

### (Step 2-bis: Visualizing the explanation)

We provide a script to visualize the explanation by overlaying the node-level importance scores on the original image. For instance by running:

```
python visualize_explanations --cell_graphs <SOME-SAVE-PATH>/quant-gnn-explainers-data/cell_graphs --images <SOME-SAVE-PATH>/quant-gnn-explainers-data/images --importance_scores <SOME-SAVE-PATH>/quant-gnn-explainers-data/graphgradcam
 --save_path <SOME-SAVE-PATH>/quant-gnn-explainers-data/visualization
 ```

### Step 3: Quantifying explainers

If you don't want to re-generate all the cell graphs and explanations for the whole set, we provide a zip file that you can directly download [here](https://ibm.box.com/shared/static/412lfz992djt8u6bgu13y9cj9qsurwui.zip). This file contains pre-processed cell graphs and explanations. 

To run the quantitative analysis, you can simply run:

```
python run_qualitative_analysis.py --cell_graphs <SOME-SAVE-PATH>/quant-gnn-explainers-data/cell_graphs --importance_scores <SOME-SAVE-PATH>quant-gnn-explainers-data/graphgradcam --save_path <SOME-SAVE-PATH>/visualization/graphgradcam
```

The code will save a number of histograms that represent the distribution of nuclei-level attributes per tumor type. The code also print the separability scores: average, maximum and correlation. 

Note: The code will by default remove misclassified samples, it is also possible to run on the entire test set by adding the flag `misclassification` (see implementation for details). 

An example of attribute histograms would be:

![Attribute-level histograms.](figs/readme_fig2.png)


### Disclaimer: 

A number of elements changed in the pipeline explaining different numbers than the ones reported in the original paper. However, the conclusion remains. The differences include a different nuclei detector, an optimized stain normalization, a different list of nuclei-level attributes (changed to reduce computational requirements), a more robust attribute normalization.  

If you use this code, please consider citing our work:

```
@inproceedings{jaume2021,
    title = "Quantifying Explainers of Graph Neural Networks in Computational Pathology",
    author = "Guillaume Jaume, Pushpak Pati, Behzad Bozorgtabar, Antonio Foncubierta-Rodríguez, Florinda Feroce, Anna Maria Anniciello, Tilman Rau, Jean-Philippe Thiran, Maria Gabrani, Orcun Goksel",
    booktitle = "IEEE CVPR",
    url = "https://arxiv.org/abs/2011.12646",
    year = "2021"
} 
```
