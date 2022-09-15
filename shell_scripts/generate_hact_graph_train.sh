#!/bin/bash

conda activate hact

python /nadeem_lab/Carlin/hact-net/hact-net/generate_hact_graphs.py \
--image_path /nadeem_lab/data/external_downloads/BRACS_RoI/latest_version/train/ \
--save_path /nadeem_lab/Carlin/hact-net/data/hact-net-data-v6/ \
--disable_stain_norm
