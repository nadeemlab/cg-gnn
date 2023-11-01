"""Generate interactive cell graph visualizations."""

from argparse import ArgumentParser

from cggnn.explain import generate_interactives
from cggnn.util import load_cell_graphs


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
        '--feature_names_path',
        type=str,
        help='Path to the list of feature names.',
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
    graphs_data, feature_names = load_cell_graphs(args.cg_path)
    generate_interactives(graphs_data, feature_names, args.output_directory)
