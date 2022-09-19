#!/bin/bash

conda activate hact

python /nadeem_lab/Carlin/hact-net/hact-net/train.py \
--cg_path /nadeem_lab/Carlin/hact-net/data/melanoma-limited-600-py39/ \
--model_path /nadeem_lab/Carlin/hact-net/models/ \
--model_name melanoma_600-limited \
--config_fpath /nadeem_lab/Carlin/hact-net/config/melanoma_cggnn.yml \
-b 8 --in_ram --epochs 100 -l 0.0005

