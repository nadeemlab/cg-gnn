#!/bin/bash

cd /nadeem_lab/Carlin/hact-net/core/
source ~/miniconda3/bin/activate
conda activate hactnet_hpc5

python generate_hact_graphs_no_normalization.py \
--image_path /nadeem_lab/datasets/BRACS_RoI/latest_version/val/ \
--save_path /nadeem_lab/Carlin/hact-net/data/hact-net-data-check-script/