"""
Explain a cell graph (CG) prediction using a pretrained CG-GNN and a graph explainer.

As used in:
"Quantifying Explainers of Graph Neural Networks in Computational Pathology", Jaume et al, CVPR, 2021.
"""

from os.path import join
from typing import List, Optional, Tuple, Dict, DefaultDict

from dgl import DGLGraph
from numpy import ndarray
from pandas import DataFrame, Index

from cggnn.util import CellGraphModel
from cggnn.util.util import GraphData
from .importance import calculate_importance
from .plot_interactives import generate_interactives
from .separability import calculate_separability
from .specimen_importance import save_importances, unify_importance_across


def _class_pair_rephrase(class_pair: Tuple[int, int],
                         label_to_result: Dict[int, str]) -> Tuple[str, str]:
    'Convert an int class pair to a tuple class pair.'
    return tuple(label_to_result[label] for label in class_pair)


def explain_cell_graphs(cell_graphs_data: List[GraphData],
                        model: CellGraphModel,
                        explainer_model: str,
                        feature_names: List[str],
                        phenotype_names: List[str],
                        prune_misclassified: bool = True,
                        concept_grouping: Optional[Dict[str,
                                                        List[str]]] = None,
                        risk: Optional[ndarray] = None,
                        pathological_prior: Optional[ndarray] = None,
                        merge_rois: bool = True,
                        cell_graph_names: Optional[List[str]] = None,
                        label_to_result: Optional[Dict[int, str]] = None,
                        out_directory: Optional[str] = None
                        ) -> Tuple[DataFrame, DataFrame, Dict[Tuple[int, int], DataFrame],
                                   Dict[int, float]]:
    "Generate explanations for all the cell graphs."

    cell_graphs_and_labels = ([d.graph for d in cell_graphs_data], [
                              d.label for d in cell_graphs_data])
    calculate_importance(cell_graphs_and_labels[0], model, explainer_model)
    if (out_directory is not None) and (cell_graph_names is not None):
        graph_groups: Dict[str, List[DGLGraph]] = DefaultDict(list)
        for graph in cell_graphs_data:
            if merge_rois:
                graph_groups[graph.specimen].append(graph.graph)
            else:
                graph_groups[graph.name].append(graph.graph)
        generate_interactives(graph_groups, feature_names,
                              phenotype_names, out_directory)

    df_seperability_by_concept, df_seperability_aggregated, dfs_k_max_distance = \
        calculate_separability(cell_graphs_and_labels, model, feature_names, phenotype_names,
                               prune_misclassified=prune_misclassified,
                               concept_grouping=concept_grouping, risk=risk,
                               pathological_prior=pathological_prior, out_directory=out_directory)

    if label_to_result is not None:
        df_seperability_by_concept.columns = [
            _class_pair_rephrase(class_pair, label_to_result) for class_pair in
            df_seperability_by_concept.columns.values]
        df_seperability_aggregated.set_index(Index(
            (_class_pair_rephrase(class_pair, label_to_result)
             if isinstance(class_pair, tuple) else class_pair
             ) for class_pair in df_seperability_aggregated.index.values), inplace=True)
        dfs_k_max_distance = {_class_pair_rephrase(
            class_pair, label_to_result): df for class_pair, df in dfs_k_max_distance.items()}

    cell_graphs_by_specimen: Dict[str, List[DGLGraph]] = DefaultDict(list)
    for cell_graph_data in cell_graphs_data:
        cell_graphs_by_specimen[cell_graph_data.specimen].append(
            cell_graph_data.graph)
    importances = unify_importance_across(
        list(cell_graphs_by_specimen.values()), model)
    if out_directory is not None:
        save_importances(importances, join(out_directory, 'importances.csv'))

    return df_seperability_by_concept, df_seperability_aggregated, dfs_k_max_distance, importances
