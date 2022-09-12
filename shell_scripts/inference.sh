#!/bin/bash

conda activate hactnet_hpc

python /nadeem_lab/Carlin/hact-net/core/inference.py \
--cg_path /nadeem_lab/Carlin/hact-net/data/hact-net-data-v1/cell_graphs/ \
--tg_path /nadeem_lab/Carlin/hact-net/data/hact-net-data-v1/tissue_graphs/ \
--assign_mat_path /nadeem_lab/Carlin/hact-net/data/hact-net-data-v1/assignment_matrices/ \
--config_fpath ./config/bracs_hact_7_classes_pna.yml \
# Choose either the pretrained model or a model of your specification.
--pretrained
# --model_path /nadeem_lab/Carlin/hact-net/models/5a9c88a9-4e70-4657-b7ab-9cde591cdf95/model_best_val_weighted_f1_score.pt

