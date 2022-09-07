#!/bin/bash

conda activate hactnet_hpc

python /nadeem_lab/Carlin/hact-net/core/train.py \
--cg_path /nadeem_lab/Carlin/hact-net/data/melanoma-v0.2/ \
--model_path /nadeem_lab/Carlin/hact-net/models/ \
--config_fpath /nadeem_lab/Carlin/hact-net/core/config/melanoma_cggnn.yml \
-b 8 --in_ram --epochs 5 -l 0.0005

