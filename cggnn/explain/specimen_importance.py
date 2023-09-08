"""Unify importance scores for cells from different ROIs into a single score."""

from typing import Dict, List, Tuple, DefaultDict

from dgl import DGLGraph
from numpy import average
from pandas import Series

from cggnn.util import CellGraphModel
from cggnn.util.constants import INDICES, IMPORTANCES
from cggnn.train import infer_with_model


def unify_importance(graphs: List[DGLGraph], model: CellGraphModel) -> Dict[int, float]:
    """Merge the importance values for each cell in a specimen."""
    probs = infer_with_model(model, graphs, return_probability=True)
    hs_id_to_importances: Dict[int, List[Tuple[float, float]]] = DefaultDict(list)
    for i_graph, graph in enumerate(graphs):
        for i in range(graph.num_nodes()):
            hs_id_to_importances[graph.ndata[INDICES][i].item()].append(
                (graph.ndata[IMPORTANCES][i], max(probs[i_graph, ])))
    hs_id_to_importance: Dict[int, float] = {}
    for hs_id, importance_confidences in hs_id_to_importances.items():
        hs_id_to_importance[hs_id] = average([ic[0] for ic in importance_confidences],
                                             weights=[ic[1] for ic in importance_confidences])
    return hs_id_to_importance


def unify_importance_across(graphs_by_specimen: List[List[DGLGraph]],
                            model: CellGraphModel) -> Dict[int, float]:
    """Merge importance values for all cells in all ROIs in all specimens."""
    hs_id_to_importance: Dict[int, float] = {}
    for graphs in graphs_by_specimen:
        for hs_id, importance in unify_importance(graphs, model).items():
            if hs_id in hs_id_to_importance:
                raise RuntimeError(
                    'The same histological structure ID appears in multiple specimens.')
            hs_id_to_importance[hs_id] = importance
    return hs_id_to_importance


def save_importances(hs_id_to_importance: Dict[int, float], out_directory: str) -> None:
    """Save importance scores per histological structure to CSV."""
    s = Series(hs_id_to_importance).sort_index()
    s.name = 'importance'
    s.to_csv(out_directory)
