"Explain a cell graph (CG) prediction using a pretrained CG-GNN and a graph explainer."

from os.path import join
from argparse import ArgumentParser
from typing import Dict, List, DefaultDict

from dgl import DGLGraph

from hactnet.explain import unify_importance_across, save_importances
from hactnet.util import load_cell_graphs, instantiate_model


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
        '--model_checkpoint_path',
        type=str,
        help='Path to the model checkpoint.',
        required=True
    )
    parser.add_argument(
        '--out_directory',
        type=str,
        help='Where to save the output reporting.',
        default=None,
        required=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    cell_graphs_data = load_cell_graphs(args.cg_path)
    cell_graphs_by_specimen: Dict[str, List[DGLGraph]] = DefaultDict(list)
    for cg in cell_graphs_data:
        cell_graphs_by_specimen[cg.specimen].append(cg.g)
    hs_id_to_importance = unify_importance_across(
        list(cell_graphs_by_specimen.values()),
        instantiate_model(([d.g for d in cell_graphs_data], [d.label for d in cell_graphs_data]),
                          model_checkpoint_path=args.model_checkpoint_path))
    if args.out_directory is not None:
        save_importances(hs_id_to_importance, join(
            args.out_directory, 'importances.csv'))
