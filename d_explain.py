"Explain a cell graph (CG) prediction using a pretrained CG-GNN and a graph explainer: GraphGradCAM."
from argparse import ArgumentParser

from pandas import read_hdf

from hactnet.explain import explain_cell_graphs
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
        default=None,
        required=False
    )
    parser.add_argument(
        '--out_directory',
        type=str,
        help='Where to save the output graph visualizations.',
        default=None,
        required=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    cell_graphs = load_cell_graphs(args.cg_path)
    explain_cell_graphs(cell_graphs[0],
                        instantiate_model(
                            cell_graphs, model_checkpoint_path=args.model_checkpoint_path),
                        args.explainer,
                        [col[3:] for col in read_hdf(
                            "data/melanoma_cells.h5").columns.values if col.startswith('FT_')],
                        load_cell_graph_names(args.cg_path),
                        args.out_directory)
