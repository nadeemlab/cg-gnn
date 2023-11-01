"""Merge cell importance scores multiple ROIs into a single score."""

from os.path import join
from argparse import ArgumentParser
from typing import Dict, List, DefaultDict

from dgl import DGLGraph

from cggnn.explain import unify_importance_across, save_importances
from cggnn.util import load_cell_graphs, instantiate_model


def parse_arguments():
    """Process command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        '--cg_path',
        type=str,
        help='Path to the cell graphs.',
        required=True
    )
    parser.add_argument(
        '--model_checkpoint_path',
        type=str,
        help='Path to the model checkpoint.',
        required=True
    )
    parser.add_argument(
        '--output_directory',
        type=str,
        help='Where to save the output reporting.',
        default=None,
        required=False
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
    graphs_data = load_cell_graphs(args.cg_path)[0]
    cell_graphs_by_specimen: Dict[str, List[DGLGraph]] = DefaultDict(list)
    for cg in graphs_data:
        cell_graphs_by_specimen[cg.specimen].append(cg.graph)
    hs_id_to_importance = unify_importance_across(
        list(cell_graphs_by_specimen.values()),
        instantiate_model(graphs_data, model_checkpoint_path=args.model_checkpoint_path),
        random_seed=args.random_seed)
    if args.output_directory is not None:
        save_importances(hs_id_to_importance, join(args.output_directory, 'importances.csv'))
