"Generate interactive cell graph visualizations."

from argparse import ArgumentParser
from typing import Dict, List, DefaultDict

from pandas import read_hdf
from dgl import DGLGraph

from cggnns.explain import generate_interactives
from cggnns.util import load_cell_graphs


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
        '--merge_rois',
        help='Merge ROIs together by specimen.',
        action='store_true'
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
    cell_graphs_data = load_cell_graphs(args.cg_path)
    graph_groups: Dict[str, List[DGLGraph]] = DefaultDict(list)
    for g in cell_graphs_data:
        if args.merge_rois:
            graph_groups[g.specimen].append(g.g)
        else:
            graph_groups[g.name].append(g.g)
    columns = read_hdf(args.cell_data_hdf_path).columns.values
    generate_interactives(
        graph_groups,
        [col[3:] for col in columns if col.startswith('FT_')],
        [col[3:] for col in columns if col.startswith('PH_')],
        args.out_directory)
