"""Run through the entire SPT CG-GNN pipeline."""

from os import makedirs
from shutil import rmtree
from typing import Dict, Tuple, Optional

from pandas import DataFrame

from cggnn.util import CellGraphModel
from cggnn.generate_graphs import generate_graphs
from cggnn.train import train
from cggnn.explain import explain_cell_graphs


def run(df_cell: DataFrame,
        df_label: DataFrame,
        label_to_result: Dict[int, str],
        validation_data_percent: int = 0,
        test_data_percent: int = 15,
        use_channels: bool = True,
        use_phenotypes: bool = True,
        roi_side_length: Optional[int] = None,
        cells_per_slide_target: int = 5_000,
        target_name: Optional[str] = None,
        in_ram: bool = True,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 1,
        k_folds: int = 0,
        explainer_model: str = 'pp',
        merge_rois: bool = True,
        prune_misclassified: bool = True,
        random_seed: Optional[int] = None
        ) -> Tuple[CellGraphModel, Dict[int, float]]:
    """Run the SPT CG-GNN pipeline on the given DataFrames and identifier-to-name dictionaries."""
    makedirs('tmp/', exist_ok=True)

    graphs_data, feature_names = generate_graphs(df_cell,
                                                 df_label,
                                                 validation_data_percent,
                                                 test_data_percent,
                                                 use_channels=use_channels,
                                                 use_phenotypes=use_phenotypes,
                                                 roi_side_length=roi_side_length,
                                                 cells_per_slide_target=cells_per_slide_target,
                                                 target_name=target_name,
                                                 random_seed=random_seed)

    model = train(graphs_data,
                  'tmp/',
                  in_ram=in_ram,
                  epochs=epochs,
                  learning_rate=learning_rate,
                  batch_size=batch_size,
                  k_folds=k_folds,
                  random_seed=random_seed)

    explanations = explain_cell_graphs(
        graphs_data,
        model,
        explainer_model,
        feature_names,
        merge_rois=merge_rois,
        prune_misclassified=prune_misclassified,
        label_to_result=label_to_result,
        output_directory='out/',
        random_seed=random_seed)

    explanations[0].to_csv('out/separability_concept.csv')
    explanations[1].to_csv('out/separability_attribute.csv')
    for class_pair, df in explanations[2].items():
        df.to_csv(f'out/separability_k_best_{class_pair}.csv')
    rmtree('tmp/')

    return model, explanations[3]
