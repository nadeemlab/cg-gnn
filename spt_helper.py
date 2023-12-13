"""Helper functions to translate SPT HSGraphs and prepare them for CG-GNN training."""

from numpy import nonzero  # type: ignore
from networkx import to_scipy_sparse_array  # type: ignore
from torch import (
    FloatTensor,
    IntTensor,  # type: ignore
)
from dgl import DGLGraph, graph
from spatialprofilingtoolbox.cggnn.util import HSGraph, GraphData as SPTGraphData

from cggnn.util import GraphData
from cggnn.util.constants import INDICES, CENTROIDS, FEATURES, IMPORTANCES


def convert_spt_graph(g_spt: HSGraph) -> DGLGraph:
    """Convert a SPT HSGraph to a CG-GNN cell graph."""
    num_nodes = g_spt.node_features.shape[0]
    g_dgl = graph([])
    g_dgl.add_nodes(num_nodes)
    g_dgl.ndata[INDICES] = IntTensor(g_spt.histological_structure_ids)
    g_dgl.ndata[CENTROIDS] = FloatTensor(g_spt.centroids)
    g_dgl.ndata[FEATURES] = FloatTensor(g_spt.node_features)
    # Note: channels and phenotypes are binary variables, but DGL only supports FloatTensors
    edge_list = nonzero(g_spt.adj.toarray())
    g_dgl.add_edges(list(edge_list[0]), list(edge_list[1]))
    return g_dgl


def convert_spt_graph_data(g_spt: SPTGraphData) -> GraphData:
    """Convert a SPT GraphData object to a CG-GNN/DGL GraphData object."""
    return GraphData(
        graph=convert_spt_graph(g_spt.graph),
        label=g_spt.label,
        name=g_spt.name,
        specimen=g_spt.specimen,
        set=g_spt.set,
    )


def convert_spt_graphs_data(graphs_data: list[SPTGraphData]) -> list[GraphData]:
    """Convert a list of SPT HSGraphs to CG-GNN cell graphs."""
    return [convert_spt_graph_data(g_spt) for g_spt in graphs_data]


def convert_dgl_graph(g_dgl: DGLGraph) -> HSGraph:
    """Convert a DGLGraph to a CG-GNN cell graph."""
    return HSGraph(
        adj=to_scipy_sparse_array(g_dgl.to_networkx()),
        node_features=g_dgl.ndata[FEATURES],
        centroids=g_dgl.ndata[CENTROIDS],
        histological_structure_ids=g_dgl.ndata[INDICES],
        importances=g_dgl.ndata[IMPORTANCES] if (IMPORTANCES in g_dgl.ndata) else None,
    )


def convert_dgl_graph_data(g_dgl: GraphData) -> SPTGraphData:
    return SPTGraphData(
        graph=convert_dgl_graph(g_dgl.graph),
        label=g_dgl.label,
        name=g_dgl.name,
        specimen=g_dgl.specimen,
        set=g_dgl.set,
    )


def convert_dgl_graphs_data(graphs_data: list[GraphData]) -> list[SPTGraphData]:
    """Convert a list of DGLGraphs to CG-GNN cell graphs."""
    return [convert_dgl_graph_data(g_dgl) for g_dgl in graphs_data]
