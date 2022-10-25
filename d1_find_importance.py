"Update the cell graphs with each cell's importance value."

from os.path import join
from argparse import ArgumentParser

from pandas import read_hdf
from torch import FloatTensor
from dgl import save_graphs

from hactnet.explain import calculate_importance
from hactnet.util import load_cell_graphs, load_cell_graph_names, instantiate_model


def parse_arguments():
    "Process command line arguments."
    parser = ArgumentParser()
    parser.add_argument(
        '--cg_path',
        type=str,
        help='Path to the cell graphs.',
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    cell_graphs_and_labels = load_cell_graphs(args.cg_path)
    cell_graphs = calculate_importance(cell_graphs_and_labels[0], instantiate_model(
        cell_graphs_and_labels, model_checkpoint_path=args.model_checkpoint_path), args.explainer)
    for g, l, n in zip(cell_graphs, cell_graphs_and_labels[1], load_cell_graph_names(args.cg_path)):
        save_graphs(join(args.cg_path, n + '.bin'),
                    [g],
                    {'label': FloatTensor([l])})
