#!/bin/bash

cd /nadeem_lab/Eliram/repos/hact-net/core
source /nadeem_lab/miniconda3/etc/profile.d/conda.sh 
conda activate hactnet_hpc4

which python

python train.py \
--cg_path /nadeem_lab/Eliram/repos/hact-net/data/hact-net-data2/cell_graphs/ \
--config_fpath ./config/bracs_cggnn_7_classes_pna.yml -b 8 --in_ram --epochs 60 -l 0.0005

