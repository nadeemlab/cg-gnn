#!/bin/bash

onda activate hactnet_hpc

python /nadeem_lab/Carlin/hact-net/core/train.py \
--cg_path /nadeem_lab/Carlin/hact-net/data/hact-net-data-v5/cell_graphs/ \
--tg_path /nadeem_lab/Carlin/hact-net/data/hact-net-data-v5/tissue_graphs/ \
--assign_mat_path /nadeem_lab/Carlin/hact-net/data/hact-net-data-v5/assignment_matrices/ \
--model_path /nadeem_lab/Carlin/hact-net/models/ \
--config_fpath /nadeem_lab/Carlin/hact-net/core/config/bracs_hact_7_classes_pna.yml -b 8 --in_ram --epochs 100 -l 0.0005
