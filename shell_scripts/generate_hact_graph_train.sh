#!/bin/bash

conda activate hactnet_hpc

python /nadeem_lab/Carlin/hact-net/core/generate_hact_graphs.py \
--image_path /nadeem_lab/datasets/BRACS_RoI/latest_version/train/ \
--save_path /nadeem_lab/Carlin/hact-net/data/hact-net-data-v6/ \
--disable_stain_norm
