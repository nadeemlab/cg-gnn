#!/bin/bash

cd /nadeem_lab/Eliram/repos/hact-net/core/
source /nadeem_lab/miniconda3/bin/activate 
conda activate histocartography_hpc_clone

python generate_hact_graphs_no_normalization.py \
--image_path /nadeem_lab/datasets/BRACS/BRACS_RoI/latest_version/test/ \
--save_path /nadeem_lab/Eliram/repos/hact-net/data/hact-net-data-v5/
