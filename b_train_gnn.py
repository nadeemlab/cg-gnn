"""Script for training CG-GNN, TG-GNN, and HACT models."""

from argparse import ArgumentParser

from cggnn import train
from cggnn.util import load_cell_graphs


def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        '--cg_path',
        type=str,
        help='Directory with the cell graphs, metadata, and feature names.',
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
    parser.add_argument(
        '--random_seed',
        type=int,
        help='Random seed to use for reproducibility.',
        default=None,
        required=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    train(load_cell_graphs(args.cg_path)[0],
          args.output_directory,
          in_ram=args.in_ram,
          epochs=args.epochs,
          learning_rate=args.learning_rate,
          batch_size=args.batch_size,
          k_folds=args.k_folds,
          random_seed=args.random_seed)
