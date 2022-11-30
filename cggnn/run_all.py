"Run through the entire SPT CG-GNN pipeline."

from os import makedirs
from shutil import rmtree
from typing import Tuple, List, Literal, Optional

from dgl import DGLGraph

from cggnn.util.constants import TRAIN_VALIDATION_TEST
from cggnn.spt_to_df import spt_to_dataframes
from cggnn.generate_graph_from_spt import generate_graphs
from cggnn.train import train
from cggnn.explain import explain_cell_graphs


def run_pipeline(measurement_study: str,
                 analysis_study: str,
                 specimen_study: str,
                 host: str,
                 dbname: str,
                 user: str,
                 password: str,
                 validation_data_percent: int = 15,
                 test_data_percent: int = 15,
                 roi_side_length: int = 600,
                 target_column: Optional[str] = None,
                 batch_size: int = 1,
                 epochs: int = 10,
                 learning_rate: float = 10e-3,
                 k_folds: int = 0,
                 explainer: str = 'pp',
                 merge_rois: bool = True,
                 prune_misclassified: bool = True) -> None:
    "Run the full SPT CG-GNN pipeline."
    makedirs('tmp/', exist_ok=True)
    df_cell, df_label, label_to_result = spt_to_dataframes(
        analysis_study, measurement_study, specimen_study, host, dbname,
        user, password)
    graphs = generate_graphs(df_cell, df_label, validation_data_percent, test_data_percent,
                             roi_side_length, target_column)

    train_validation_test: Tuple[Tuple[List[DGLGraph], List[int]],
                                 Tuple[List[DGLGraph], List[int]],
                                 Tuple[List[DGLGraph], List[int]]] = (([], []), ([], []), ([], []))
    for gd in graphs:
        s: Literal[0, 1, 2] = 0
        if gd.train_validation_test == 'validation':
            s = 1
        elif gd.train_validation_test == 'test':
            s = 2
        train_validation_test[s][0].append(gd.graph)
        train_validation_test[s][1].append(gd.label)
    model = train(train_validation_test, 'tmp/', epochs=epochs,
                  learning_rate=learning_rate, batch_size=batch_size, k_folds=k_folds)

    columns = df_cell.columns.values
    i = -1
    while len(train_validation_test[i][0]) == 0:
        i -= 1
        if i < -3:
            raise RuntimeError('all sets created are empty')
    evaluation_set = train_validation_test[i]
    assert evaluation_set is not None
    explanations = explain_cell_graphs(
        graphs, model, explainer,
        [col[3:] for col in columns if col.startswith('FT_')],
        [col[3:] for col in columns if col.startswith('PH_')],
        merge_rois=merge_rois,
        prune_misclassified=prune_misclassified,
        cell_graph_names=[d.name for d in graphs
                          if d.train_validation_test == TRAIN_VALIDATION_TEST[i]],
        label_to_result=label_to_result,
        out_directory='out/')

    explanations[0].to_csv('out/separability_concept.csv')
    explanations[1].to_csv('out/separability_attribute.csv')
    for class_pair, df in explanations[2].items():
        df.to_csv(f'out/separability_k_best_{class_pair}.csv')
    rmtree('tmp/')
