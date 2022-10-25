"Generate interactive cell graph visualizations."

from argparse import ArgumentParser

from pandas import read_hdf

from hactnet.explain import generate_interactives
from hactnet.util import load_cell_graphs, load_cell_graph_names


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
        '--cell_data_hdf_path',
        type=str,
        help='Where to find the data for cells to lookup feature and phenotype names.',
        required=True
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
    generate_interactives(
        load_cell_graphs(args.cg_path)[0],
        [col[3:] for col in read_hdf(
            args.cell_data_hdf_path).columns.values if col.startswith('FT_')],
        load_cell_graph_names(args.cg_path),
        args.out_directory)
