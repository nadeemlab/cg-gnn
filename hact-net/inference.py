#!/usr/bin/env python3
"""
Script for testing CG-GNN, TG-GNN and HACT models
"""

from os.path import join
from argparse import ArgumentParser

from yaml import safe_load
from torch import load, no_grad, argmax, cat
from torch.cuda import is_available
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from histocartography.ml import CellGraphModel, TissueGraphModel, HACTModel

from util import CGTGDataset, collate

# cuda support
IS_CUDA = is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'

def parse_arguments():
    "Parse command line arguments."
    parser = ArgumentParser()
    parser.add_argument(
        '--cg_path',
        type=str,
        help='path to the cell graphs.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--tg_path',
        type=str,
        help='path to tissue graphs.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--assign_mat_path',
        type=str,
        help='path to the assignment matrices.',
        default=None,
        required=False
    )
    parser.add_argument(
        '-conf',
        '--config_fpath',
        type=str,
        help='path to the config file.',
        default='',
        required=False
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='path to model to test.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--in_ram',
        help='if the data should be stored in RAM.',
        action='store_true',
    )
    parser.add_argument(
        '--pretrained',
        help='if the data should be stored in RAM.',
        action='store_true',
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        help='batch size.',
        default=1,
        required=False
    )
    return parser.parse_args()

def main(args):
    """
    Test HACTNet, CG-GNN or TG-GNN.
    Args:
        args (Namespace): parsed arguments.
    """

    assert not(args.pretrained and args.model_path is not None), "Provide a model path or set pretrained. Not both."
    assert (args.pretrained or args.model_path is not None), "Provide either a model path or set pretrained."

    # load config file
    with open(args.config_fpath, 'r', encoding='utf-8') as f:
        config = safe_load(f)

    # make test data loader
    dataloader = DataLoader(
        CGTGDataset(
            cg_path=join(
                args.cg_path, 'train') if args.cg_path is not None else None,
            tg_path=join(
                args.tg_path, 'train') if args.tg_path is not None else None,
            assign_mat_path=join(
                args.assign_mat_path, 'train') if args.assign_mat_path is not None else None,
            load_in_ram=args.in_ram
        ),
        batch_size=args.batch_size,
        collate_fn=collate
    )

    # declare model 
    if 'bracs_cggnn' in args.config_fpath:
        model = CellGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=config['node_feat_dim'],
            num_classes=7,
            pretrained=args.pretrained
        ).to(DEVICE)

    elif 'bracs_tggnn' in args.config_fpath:
        model = TissueGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=config['node_feat_dim'],
            num_classes=7,
            pretrained=args.pretrained
        ).to(DEVICE)

    elif 'bracs_hact' in args.config_fpath:
        model = HACTModel(
            cg_gnn_params=config['cg_gnn_params'],
            tg_gnn_params=config['tg_gnn_params'],
            classification_params=config['classification_params'],
            cg_node_dim=config['node_feat_dim'],
            tg_node_dim=config['node_feat_dim'],
            num_classes=7,
            pretrained=args.pretrained
        ).to(DEVICE)
    else:
        raise ValueError('Model type not recognized. Options are: TG, CG or HACT.')

    # load weights if model path is provided. 
    if not args.pretrained:
        model.load_state_dict(load(args.model_path))
    model.eval()

    # print # of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    # start testing
    all_test_logits = []
    all_test_labels = []
    for batch in tqdm(dataloader, desc='Testing', unit='batch'):
        labels = batch[-1]
        data = batch[:-1]
        with no_grad():
            logits = model(*data)
        all_test_logits.append(logits)
        all_test_labels.append(labels)

    all_test_logits = cat(all_test_logits).cpu()
    all_test_preds = argmax(all_test_logits, dim=1)
    all_test_labels = cat(all_test_labels).cpu()

    all_test_preds = all_test_preds.detach().numpy()
    all_test_labels = all_test_labels.detach().numpy()

    accuracy = accuracy_score(all_test_labels, all_test_preds)
    weighted_f1_score = f1_score(all_test_labels, all_test_preds, average='weighted')
    report = classification_report(all_test_labels, all_test_preds)

    print(f'Test weighted F1 score {weighted_f1_score}')
    print(f'Test accuracy {accuracy}')
    print(f'Test classification report {report}')


if __name__ == "__main__":
    main(args=parse_arguments())
