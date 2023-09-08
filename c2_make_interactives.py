"""Generate interactive cell graph visualizations."""

from argparse import ArgumentParser
from typing import Dict, List, DefaultDict

from numpy import genfromtxt
from dgl import DGLGraph

from cggnn.explain import generate_interactives
from cggnn.util import load_cell_graphs


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
        '--feature_names_path',
        type=str,
        help='Path to the list of feature names.',
        required=True
    )
    parser.add_argument(
        '--spt_hdf_cell_filename',
        type=str,
        help='Where to find the data for cells to lookup channel and phenotype names.',
        required=True
    )
    parser.add_argument(
        '--merge_rois',
        help='Merge ROIs together by specimen.',
        action='store_true'
    )
    parser.add_argument(
        '--output_directory',
        type=str,
        help='Where to save the output graph visualizations.',
        default=None,
        required=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    cell_graphs_data = load_cell_graphs(args.cg_path)
    graph_groups: Dict[str, List[DGLGraph]] = DefaultDict(list)
    for g in cell_graphs_data:
        if args.merge_rois:
            graph_groups[g.specimen].append(g.graph)
        else:
            graph_groups[g.name].append(g.graph)
    feature_names = genfromtxt(args.feature_names_path, dtype=str).tolist()
    generate_interactives(graph_groups, feature_names, args.output_directory)
