#!/bin/bash

cd /nadeem_lab/Eliram/repos/hact-net/core
source /nadeem_lab/miniconda3/bin/activate 
conda activate hactnet_hpc4

python inference.py \
--cg_path /nadeem_lab/Eliram/repos/hact-net/data/hact-net-data-v1/cell_graphs/ \
--tg_path /nadeem_lab/Eliram/repos/hact-net/data/hact-net-data-v1/tissue_graphs/ \
--assign_mat_path /nadeem_lab/Eliram/repos/hact-net/data/hact-net-data-v1/assignment_matrices/ \
--config_fpath ./config/bracs_hact_7_classes_pna.yml --pretrained

