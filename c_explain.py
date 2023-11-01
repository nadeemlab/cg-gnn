"""Explain a cell graph (CG) prediction using a pretrained CG-GNN and a graph explainer."""

from argparse import ArgumentParser

from cggnn.explain import explain_cell_graphs
from cggnn.util import load_cell_graphs, instantiate_model, load_label_to_result


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
        '--prune_misclassified',
        help='Remove entries for misclassified cell graphs when calculating separability scores.',
        action='store_true'
    )
    parser.add_argument(
        '--label_to_result_path',
        type=str,
        help='Where to find the data mapping label ints to their string results.',
        required=False
    )
    parser.add_argument(
        '--output_directory',
        type=str,
        help='Where to save the output reporting.',
        default=None,
        required=False
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
    graphs_data, feature_names = load_cell_graphs(args.cg_path)
    df_concept, df_aggregated, dfs_k_dist, importances = explain_cell_graphs(
        graphs_data,
        instantiate_model(graphs_data, model_checkpoint_path=args.model_checkpoint_path),
        args.explainer,
        feature_names,
        merge_rois=args.merge_rois,
        prune_misclassified=args.prune_misclassified,
        label_to_result=load_label_to_result(args.label_to_result_path),
        output_directory=args.output_directory,
        random_seed=args.random_seed
    )

    print('')
    print(df_concept)
    print('')
    print(df_aggregated)
    print('')
    for cg_pair, df_k in dfs_k_dist.items():
        print(cg_pair)
        print(df_k)
        print('')
    print(len(importances))
