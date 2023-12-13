"""Convert SPT graph objects to CG-GNN graph objects and run training and evaluation with them."""

from os import remove

from spatialprofilingtoolbox.cggnn.util import load_hs_graphs, save_hs_graphs

from cggnn.run import train_and_evaluate
from cggnn.scripts.train import parse_arguments
from spt_helper import convert_spt_graphs_data, convert_dgl_graphs_data

if __name__ == '__main__':
    args = parse_arguments()

    spt_graphs_data, feature_names = load_hs_graphs('.')
    graphs_data = convert_spt_graphs_data(spt_graphs_data)

    model, graphs_data, hs_id_to_importances = train_and_evaluate(args.cg_directory,
                                                                  args.in_ram,
                                                                  args.batch_size,
                                                                  args.epochs,
                                                                  args.learning_rate,
                                                                  args.k_folds,
                                                                  args.explainer,
                                                                  args.merge_rois,
                                                                  args.random_seed)

    spt_graphs_data = convert_dgl_graphs_data(graphs_data)
    save_hs_graphs(spt_graphs_data, '.')
    remove('graphs.bin')
    remove('graph_info.pkl')
