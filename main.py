"Run through the entire SPT CGnet pipeline."

from os import makedirs
from shutil import rmtree
from argparse import ArgumentParser

from numpy import savetxt

from hactnet.spt_to_file import spt_to_dataframes
from hactnet.generate_graph_from_spt import generate_graphs
from hactnet.train import train
from hactnet.explain import explain_cell_graphs


def parse_arguments():
    "Process command line arguments."
    parser = ArgumentParser()
    parser.add_argument(
        '--measurement_study',
        type=str,
        help='Name of the measurement study table in SPT.',
        required=True
    )
    parser.add_argument(
        '--analysis_study',
        type=str,
        help='Name of the analysis study table in SPT.',
        required=True
    )
    parser.add_argument(
        '--specimen_study',
        type=str,
        help='Name of the specimen study table in SPT.',
        required=True
    )
    parser.add_argument(
        '--host',
        type=str,
        help='Host SQL server IP.',
        required=True
    )
    parser.add_argument(
        '--dbname',
        type=str,
        help='Database in SQL server to query.',
        required=True
    )
    parser.add_argument(
        '--user',
        type=str,
        help='Server login username.',
        required=True
    )
    parser.add_argument(
        '--password',
        type=str,
        help='Server login password.',
        required=True
    )
    parser.add_argument(
        '--val_data_prc',
        type=int,
        help='Percentage of data to use as validation data. Set to 0 if you want to do k-fold '
        'cross-validation later. (Training percentage is implicit.) Default 15%.',
        default=15,
        required=False
    )
    parser.add_argument(
        '--test_data_prc',
        type=int,
        help='Percentage of data to use as the test set. (Training percentage is implicit.) '
        'Default 15%.',
        default=15,
        required=False
    )
    parser.add_argument(
        '--roi_side_length',
        type=int,
        help='Side length in pixels of the ROI areas we wish to generate.',
        default=600,
        required=False
    )
    parser.add_argument(
        '-conf',
        '--config_fpath',
        type=str,
        help='path to the config file.',
        default='',
        required=False
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        help='batch size.',
        default=1,
        required=False
    )
    parser.add_argument(
        '--epochs', type=int, help='epochs.', default=10, required=False
    )
    parser.add_argument(
        '-l',
        '--learning_rate',
        type=float,
        help='learning rate.',
        default=10e-3,
        required=False
    )
    parser.add_argument(
        '--logger',
        type=str,
        help='Logger type. Options are "mlflow" or "none"',
        required=False,
        default='none'
    )
    parser.add_argument(
        '--k',
        type=int,
        help='Folds to use in k-fold cross validation. 0 means don\'t use k-fold cross validation '
        'unless no validation dataset is provided, in which case k defaults to 3.',
        required=False,
        default=0
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    makedirs('tmp/', exist_ok=True)
    df_feat, df_label = spt_to_dataframes(args.analysis_study, args.measurement_study,
                                          args.specimen_study, args.host, args.dbname, args.user,
                                          args.password)
    sets_data, graph_to_label, graph_names = generate_graphs(df_feat, df_label, args.val_data_prc,
                                                             args.test_data_prc,
                                                             args.roi_side_length)
    training_graphs = [g for g_list in sets_data[0].values() for g in g_list]
    training_labels = [graph_to_label[g] for g in training_graphs]
    model = train(args.config_fpath, 'tmp/', (training_graphs, training_labels), None,
                  None, None, None, True, args.epochs, args.learning_rate, args.batch_size, args.k)
    explanations = explain_cell_graphs(sets_data[0], model, None, None, None)
    for graph, explanation in explanations.items():
        savetxt(f'tmp/{graph_names[graph]}_node_importance.txt', explanation)
    rmtree('tmp/')
