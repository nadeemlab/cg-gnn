#!/bin/bash

conda activate hactnet_hpc

python /nadeem_lab/Carlin/hact-net/core/generate_graph_from_spt.py \
--spt_csv_path /nadeem_lab/Carlin/hact-net/data/cell_features.csv \
--save_path /nadeem_lab/Carlin/hact-net/data/melanoma/