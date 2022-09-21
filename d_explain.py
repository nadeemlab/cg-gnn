"Explain a cell graph (CG) prediction using a pretrained CG-GNN and a graph explainer: GraphGradCAM."
from argparse import ArgumentParser

from hactnet.explain import explain_cell_graphs


def parse_arguments():
    "Process command line arguments."
    parser = ArgumentParser()
    parser.add_argument(
        '--cg_path',
        type=str,
        help='path to the cell graphs.',
        required=True
    )
    parser.add_argument(
        '-conf',
        '--config_fpath',
        type=str,
        help='path to the config file.',
        default='',
        required=True
    )
    parser.add_argument(
        '--model_checkpoint',
        type=str,
        help='path to the model checkpoint.',
        default='',
        required=True
    )
    parser.add_argument(
        '--image_path',
        type=str,
        help='path to the source images.',
        default=None,
        required=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    explain_cell_graphs(args.cg_path, args.config_fpath, args.model_checkpoint, args.image_path)
