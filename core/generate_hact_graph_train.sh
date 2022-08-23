#!/bin/bash

cd /nadeem_lab/Carlin/hact-net/core/
source ~/miniconda3/bin/activate
conda activate hactnet_hpc5

python generate_hact_graphs.py \
--image_path /nadeem_lab/datasets/BRACS_RoI/latest_version/train/ \
--save_path /nadeem_lab/Carlin/hact-net/data/hact-net-data-v6/ \
--disable_stain_norm
