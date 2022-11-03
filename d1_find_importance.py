"Update the cell graphs with each cell's importance value."

from os.path import join
from argparse import ArgumentParser

from torch import FloatTensor
from dgl import save_graphs

from cggnns.explain import calculate_importance
from cggnns.util import load_cell_graphs, instantiate_model


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
    cell_graphs_data = load_cell_graphs(args.cg_path)
    cell_graphs = [d.graph for d in cell_graphs_data]
    cell_graph_labels = [d.label for d in cell_graphs_data]
    cell_graphs = calculate_importance(cell_graphs, instantiate_model(
        (cell_graphs, cell_graph_labels), model_checkpoint_path=args.model_checkpoint_path
    ), args.explainer)
    for g, l, tvt, s, n in zip(cell_graphs, cell_graph_labels,
                               [d.train_validation_test for d in cell_graphs_data],
                               [d.specimen for d in cell_graphs_data],
                               [d.name for d in cell_graphs_data]):
        save_graphs(join(args.cg_path, tvt, s, n + '.bin'),
                    [g],
                    {'label': FloatTensor([l])})
