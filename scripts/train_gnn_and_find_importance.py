"""Script for training CG-GNN, TG-GNN, and HACT models."""

from argparse import ArgumentParser
from os.path import join
from typing import Dict, List, DefaultDict

from dgl import DGLGraph  # type: ignore
from spatialprofilingtoolbox.cggnn.util import load_cell_graphs, save_cell_graphs

from cggnn import train, calculate_importance, unify_importance_across, save_importances
from cggnn.util import instantiate_model


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
        '--explainer',
        type=str,
        help='Which explainer type to use.',
        default='pp',
        required=False
    )
    parser.add_argument(
        '--merge_rois',
        help='Merge ROIs together by specimen.',
        action='store_true'
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
    model = train(load_cell_graphs(args.cg_path)[0],
                  args.output_directory,
                  in_ram=args.in_ram,
                  epochs=args.epochs,
                  learning_rate=args.learning_rate,
                  batch_size=args.batch_size,
                  k_folds=args.k_folds,
                  random_seed=args.random_seed)
    graphs_data = load_cell_graphs(args.cg_path)[0]
    cell_graphs = calculate_importance([d.graph for d in graphs_data],
                                       model,
                                       args.explainer,
                                       random_seed=args.random_seed)
    save_cell_graphs(graphs_data, args.cg_path)
    if args.merge_rois:
        cell_graphs_by_specimen: Dict[str, List[DGLGraph]] = DefaultDict(list)
        for cg in graphs_data:
            cell_graphs_by_specimen[cg.specimen].append(cg.graph)
        hs_id_to_importance = unify_importance_across(
            list(cell_graphs_by_specimen.values()),
            instantiate_model(graphs_data, model_checkpoint_path=args.model_checkpoint_path),
            random_seed=args.random_seed)
        save_importances(hs_id_to_importance, join(args.cg_path, 'importances.csv'))
