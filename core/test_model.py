import torch
import mlflow
import os
import uuid
import yaml
from tqdm import tqdm
import mlflow.pytorch
import numpy as np
import pandas as pd
import shutil
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report

from histocartography.ml import CellGraphModel, TissueGraphModel, HACTModel

from dataloader import make_data_loader

# cuda support
IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
NODE_DIM = 514

model_path='/nadeem_lab/Eliram/repos/hact-net/core/5a9c88a9-4e70-4657-b7ab-9cde591cdf95'
config_fpath='./config/bracs_hact_7_classes_pna.yml'
cg_path='/nadeem_lab/Eliram/repos/hact-net/data/hact-net-data-v4/cell_graphs/'
tg_path='/nadeem_lab/Eliram/repos/hact-net/data/hact-net-data-v4/tissue_graphs/'
assign_mat_path='/nadeem_lab/Eliram/repos/hact-net/data/hact-net-data-v4/assignment_matrices/'

# load config file
with open(config_fpath, 'r') as f:
    config = yaml.safe_load(f)

# testing loop

# define loss function
loss_fn = torch.nn.CrossEntropyLoss()

model = HACTModel(
    cg_gnn_params=config['cg_gnn_params'],
    tg_gnn_params=config['tg_gnn_params'],
    classification_params=config['classification_params'],
    cg_node_dim=NODE_DIM,
    tg_node_dim=NODE_DIM,
    num_classes=7
).to(DEVICE)

test_dataloader = make_data_loader(
        cg_path=os.path.join(cg_path, 'test') if cg_path is not None else None,
        tg_path=os.path.join(tg_path, 'test') if tg_path is not None else None,
        assign_mat_path=os.path.join(assign_mat_path, 'test') if assign_mat_path is not None else None,
        batch_size=8,
        load_in_ram='store_true',
    )

model.eval()
for metric in ['best_val_loss', 'best_val_accuracy', 'best_val_weighted_f1_score']:
    
    print('\n*** Start testing w/ {} model ***'.format(metric))

    model_name = [f for f in os.listdir(model_path) if f.endswith(".pt") and metric in f][0]
    model.load_state_dict(torch.load(os.path.join(model_path, model_name)))

    all_test_logits = []
    all_test_labels = []
    for batch in tqdm(test_dataloader, desc='Testing: {}'.format(metric), unit='batch'):
        labels = batch[-1]
        data = batch[:-1]
        with torch.no_grad():
            logits = model(*data)
        all_test_logits.append(logits)
        all_test_labels.append(labels)

    all_test_logits = torch.cat(all_test_logits).cpu()
    all_test_preds = torch.argmax(all_test_logits, dim=1)
    all_test_labels = torch.cat(all_test_labels).cpu()

    # compute & store loss
    with torch.no_grad():
        loss = loss_fn(all_test_logits, all_test_labels).item()

    # compute & store accuracy
    all_test_preds = all_test_preds.detach().numpy()
    all_test_labels = all_test_labels.detach().numpy()
    accuracy = accuracy_score(all_test_labels, all_test_preds)
    
    # compute & store weighted f1-score
    weighted_f1_score = f1_score(all_test_labels, all_test_preds, average='weighted')

    # compute and store classification report 
    report = classification_report(all_test_labels, all_test_preds,digits=4)
    out_path = os.path.join(model_path, 'classification_report.txt')
    with open(out_path, "w") as f:
        f.write(report)

    # log MLflow models
    mlflow.pytorch.log_model(model, 'model_' + metric)

    print('Test loss {}'.format(loss))
    print('Test weighted F1 score {}'.format(weighted_f1_score))
    print('Test accuracy {}'.format(accuracy))