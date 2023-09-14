"""Script for training CG-GNN, TG-GNN, and HACT models."""

from argparse import ArgumentParser
from typing import Tuple, List

from dgl import DGLGraph

from cggnn import train
from cggnn.util import load_cell_graphs


def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        '--cg_path',
        type=str,
        help='Path to the cell graphs.',
        required=True
    )
    parser.add_argument(
        '--output_directory',
        type=str,
        help='Directory to save the model subdirectory into.',
        required=True
    )
    parser.add_argument(
        '--in_ram',
        help='If the data should be stored in RAM.',
        action='store_true',
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        help='Batch size to use during training.',
        default=1,
        required=False
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs to do.',
        default=10,
        required=False
    )
    parser.add_argument(
        '-l',
        '--learning_rate',
        type=float,
        help='Learning rate to use during training.',
        default=1e-3,
        required=False
    )
    parser.add_argument(
        '-k',
        '--k_folds',
        type=int,
        help='Folds to use in k-fold cross validation. 0 means don\'t use k-fold cross validation '
        'unless no validation dataset is provided, in which case k defaults to 3.',
        required=False,
        default=0
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    graphs = load_cell_graphs(args.cg_path)
    cg_train: Tuple[List[DGLGraph], List[int]] = ([], [])
    cg_val: Tuple[List[DGLGraph], List[int]] = ([], [])
    cg_test: Tuple[List[DGLGraph], List[int]] = ([], [])

    for gd in graphs:
        which_set: Tuple[List[DGLGraph], List[int]] = cg_train
        if gd.train_validation_test == 'validation':
            which_set = cg_val
        elif gd.train_validation_test == 'test':
            which_set = cg_test
        which_set[0].append(gd.graph)
        which_set[1].append(gd.label)

    train((cg_train, cg_val, cg_test),
          args.output_directory,
          args.in_ram,
          args.epochs,
          args.learning_rate,
          args.batch_size,
          args.k_folds)
