"""Update the cell graphs with each cell's importance value."""

from argparse import ArgumentParser
from os.path import join
from typing import Dict, List, DefaultDict

from dgl import DGLGraph  # type: ignore

from cggnn.explain import calculate_importance, unify_importance_across, save_importances
from cggnn.util import load_cell_graphs, instantiate_model, save_cell_graphs


def parse_arguments():
    """Process command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        '--cg_path',
        type=str,
        help='Directory with the cell graphs, metadata, and feature names.',
        required=True
    )
    parser.add_argument(
        '--explainer',
        type=str,
        help='Which explainer type to use.',
        default='pp',
        required=False
    )
    parser.add_argument(
        '--model_checkpoint_path',
        type=str,
        help='Path to the model checkpoint.',
        required=True
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
    graphs_data = load_cell_graphs(args.cg_path)[0]
    cell_graphs = calculate_importance([d.graph for d in graphs_data],
                                       instantiate_model(
                                           graphs_data,
                                           model_checkpoint_path=args.model_checkpoint_path),
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
