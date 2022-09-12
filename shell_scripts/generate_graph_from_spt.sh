#!/bin/bash

conda activate hactnet_hpc

python /nadeem_lab/Carlin/hact-net/core/generate_graph_from_spt.py \
--spt_csv_feat_filename /nadeem_lab/Carlin/hact-net/data/melanoma_features.csv \
--spt_csv_label_filename /nadeem_lab/Carlin/hact-net/data/melanoma_labels.csv \
--save_path /nadeem_lab/Carlin/hact-net/data/melanoma-v1/