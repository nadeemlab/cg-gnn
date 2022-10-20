#!/usr/bin/env python3
"""
Script for training CG-GNN, TG-GNN and HACT models
"""
from os.path import join, isdir
from argparse import ArgumentParser

from hactnet.train import train
from hactnet.util import load_cell_graphs


def parse_arguments():
    "Parse command line arguments."
    parser = ArgumentParser()
    parser.add_argument(
        '--cg_path',
        type=str,
        help='path to the cell graphs.'
    )
    parser.add_argument(
        '--model_save_path',
        type=str,
        help='path to where the model is saved.',
        default='',
        required=False
    )
    parser.add_argument(
        '--in_ram',
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
    parser.add_argument(
        '--epochs', type=int, help='epochs.', default=10, required=False
    )
    parser.add_argument(
        '-l',
        '--learning_rate',
        type=float,
        help='learning rate.',
        default=10e-3,
        required=False
    )
    parser.add_argument(
        '--out_path',
        type=str,
        help='path to where the output data are saved (currently only for the interpretability).',
        default='../../data/graphs',
        required=False
    )
    parser.add_argument(
        '--logger',
        type=str,
        help='Logger type. Options are "mlflow" or "none"',
        required=False,
        default='none'
    )
    parser.add_argument(
        '--k',
        type=int,
        help='Folds to use in k-fold cross validation. 0 means don\'t use k-fold cross validation '
        'unless no validation dataset is provided, in which case k defaults to 3.',
        required=False,
        default=0
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    train_dir = join(args.cg_path, 'train')
    if isdir(train_dir):
        cg_train = load_cell_graphs(train_dir)
        val_dir = join(args.cg_path, 'val')
        cg_val = load_cell_graphs(val_dir) if isdir(val_dir) else None
        test_dir = join(args.cg_path, 'test')
        cg_test = load_cell_graphs(test_dir) if isdir(test_dir) else None
    else:
        cg_train = load_cell_graphs(args.cg_path)
        cg_val = None
        cg_test = None

    train((cg_train, cg_val, cg_test),
          args.model_save_path,
          args.logger,
          args.in_ram,
          args.epochs,
          args.learning_rate,
          args.batch_size,
          args.k)
