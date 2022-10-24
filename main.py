"Run through the entire SPT CGnet pipeline."

from os import makedirs
from shutil import rmtree
from argparse import ArgumentParser
from typing import Tuple, List

from dgl import DGLGraph

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
    parser.add_argument(
        '--explainer',
        type=str,
        help='Which explainer type to use.',
        default='pp',
        required=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    makedirs('tmp/', exist_ok=True)
    df_cell, df_label = spt_to_dataframes(
        args.analysis_study, args.measurement_study, args.specimen_study, args.host, args.dbname,
        args.user, args.password)
    sets_data, graph_to_label, graph_names = generate_graphs(
        df_cell, df_label, args.val_data_prc, args.test_data_prc, args.roi_side_length)
    tuple_sets: List[Tuple[List[DGLGraph], List[int]]] = []
    name_sets: List[List[str]] = []
    for set_dict in sets_data:
        graphs = [g for g_list in set_dict.values() for g in g_list]
        labels = [graph_to_label[g] for g in graphs]
        name_sets.append([graph_names[g] for g in graphs])
        tuple_sets.append((graphs, labels))
    assert len(tuple_sets) == 3
    assert tuple_sets[0] is not None
    train_val_test = (tuple_sets[0], tuple_sets[1], tuple_sets[2])
    model = train(train_val_test, 'tmp/', logger=args.logger, epochs=args.epochs,
                  learning_rate=args.learning_rate, batch_size=args.batch_size, k=args.k)
    columns = df_cell.columns.values
    i = -1
    while len(train_val_test[i][0]) == 0:
        i -= 1
        if i < -3:
            raise RuntimeError('all sets created are empty')
    eval_set = train_val_test[i]
    assert eval_set is not None
    explanations = explain_cell_graphs(
        eval_set, model, args.explainer,
        [g.ndata['phenotypes'] for g in eval_set[0]],
        [col[3:] for col in columns if col.startswith('PH_')],
        feature_names=[col[3:] for col in columns if col.startswith('FT_')],
        cell_graph_names=name_sets[i],
        out_directory='out/')
    explanations[0].to_csv('out/separability_concept.csv')
    explanations[1].to_csv('out/separability_attribute.csv')
    for class_pair, df in explanations[2].items():
        df.to_csv(f'out/separability_k_best_{class_pair}.csv')
    rmtree('tmp/')
