"""
Explain a cell graph (CG) prediction using a pretrained CG-GNN and a graph explainer.

As used in:
"Quantifying Explainers of Graph Neural Networks in Computational Pathology",
    Jaume et al, CVPR, 2021.
"""

from os.path import join
from typing import List, Optional, Tuple, Dict, DefaultDict, Any, Union

from dgl import DGLGraph
from numpy.typing import NDArray
from pandas import DataFrame

from cggnn.util import CellGraphModel
from cggnn.util.util import GraphData
from .importance import calculate_importance
from .plot_interactives import generate_interactives
from .separability import calculate_separability
from .specimen_importance import save_importances, unify_importance_across


def explain_cell_graphs(graphs_data: List[GraphData],
                        model: CellGraphModel,
                        explainer_model: str,
                        feature_names: List[str],
                        merge_rois: bool = True,
                        prune_misclassified: bool = True,
                        concept_grouping: Optional[Dict[str, List[str]]] = None,
                        risk: Optional[NDArray[Any]] = None,
                        pathological_prior: Optional[NDArray[Any]] = None,
                        label_to_result: Optional[Dict[int, str]] = None,
                        output_directory: Optional[str] = None,
                        random_seed: Optional[int] = None
                        ) -> Tuple[DataFrame, DataFrame,
                                   Dict[Union[Tuple[int, int], Tuple[str, str]], DataFrame],
                                   Dict[int, float]]:
    """Generate explanations for all the cell graphs."""
    calculate_importance([d.graph for d in graphs_data], model,
                         explainer_model, random_seed=random_seed)

    if output_directory is not None:
        generate_interactives(graphs_data, feature_names, output_directory, merge_rois)

    df_seperability_by_concept, df_seperability_aggregated, dfs_k_max_distance = \
        calculate_separability(graphs_data,
                               model,
                               feature_names,
                               label_to_result=label_to_result,
                               prune_misclassified=prune_misclassified,
                               concept_grouping=concept_grouping,
                               risk=risk,
                               pathological_prior=pathological_prior,
                               out_directory=output_directory)

    cell_graphs_by_specimen: Dict[str, List[DGLGraph]] = DefaultDict(list)
    for cell_graph_data in graphs_data:
        cell_graphs_by_specimen[cell_graph_data.specimen].append(cell_graph_data.graph)
    importances = unify_importance_across(list(cell_graphs_by_specimen.values()),
                                          model,
                                          random_seed=random_seed)
    if output_directory is not None:
        save_importances(importances, join(output_directory, 'importances.csv'))

    return df_seperability_by_concept, df_seperability_aggregated, dfs_k_max_distance, importances
