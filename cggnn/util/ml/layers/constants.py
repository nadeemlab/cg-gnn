"""Constants used in machine learning layers."""

import torch
from torch.nn import ReLU, Tanh, Sigmoid, ELU, LeakyReLU, PReLU
import dgl
import numpy as np

from cggnn.util.constants import FEATURES, CENTROIDS


ACTIVATIONS = {
    'relu': ReLU(),
    'tanh': Tanh(),
    'sigmoid': Sigmoid(),
    'elu': ELU(),
    'PReLU': PReLU(),
    'leaky_relu': LeakyReLU()
}


GNN_MSG = 'gnn_msg'
GNN_NODE_FEAT_IN = FEATURES
GNN_NODE_FEAT_OUT = 'gnn_node_feat_out'
GNN_AGG_MSG = 'gnn_agg_msg'
GNN_EDGE_WEIGHT = 'gnn_edge_weight'
GNN_EDGE_FEAT = 'gnn_edge_feat'
CENTROID = CENTROIDS


AVAILABLE_LAYER_TYPES = {
    'gin_layer': 'GINLayer',
    'dense_gin_layer': 'DenseGINLayer',
    'pna_layer': 'PNALayer'
}


GNN_MODULE = 'cggnn.util.ml.layers.{}'


def min_nodes(graph, features):
    """Return min nodes."""
    from dgl.backend import pytorch
    feat = pytorch.pad_packed_tensor(
        graph.ndata[features],
        graph.batch_num_nodes,
        float('inf'))
    return pytorch.min(feat, 1)


READOUT_TYPES = {
    'sum': dgl.sum_nodes,
    'mean': dgl.mean_nodes,
    'max': dgl.max_nodes,
    'min': min_nodes
}


def reduce_min(x, dim):
    """Get mins in the dim-th dimension."""
    return torch.min(x, dim=dim)[0]


def reduce_max(x, dim):
    """Get maxes in the dim-th dimension."""
    return torch.max(x, dim=dim)[0]


REDUCE_TYPES = {
    'sum': torch.sum,
    'mean': torch.mean,
    'max': reduce_max,
    'min': reduce_min,
}

EPS = 1e-5


def aggregate_mean(h):
    """Find means in the first dimension."""
    return torch.mean(h, dim=1)


def aggregate_max(h):
    """Get maxes in the first dimension."""
    return torch.max(h, dim=1)[0]


def aggregate_min(h):
    """Get min in the first dimension."""
    return torch.min(h, dim=1)[0]


def aggregate_std(h):
    """Aggregate standard deviation of h."""
    return torch.sqrt(aggregate_var(h) + EPS)


def aggregate_var(h):
    """Aggregate variance of h."""
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


AGGREGATORS = {
    'mean': aggregate_mean,
    'max': aggregate_max,
    'min': aggregate_min,
    'std': aggregate_std
}


# each scaler is a function that takes as input X (B x N x Din), adj (B x N x N) and
# avg_d (dictionary containing averages over training set) and returns
# X_scaled (B x N x Din) as output

def scale_identity(h, D=None, avg_d=None):
    """Scale by 1."""
    return h


def scale_amplification(h, D, avg_d):
    """Scale h by log(D + 1) / d * h where d is the average of the log(D + 1) in train."""
    # log(D + 1) / d * h     where d is the average of the ``log(D + 1)`` in
    # the training set
    return h * (np.log(D + 1) / avg_d["log"])


def scale_attenuation(h, D, avg_d):
    """Scale by (log(D + 1))^-1 / d * X where d is the average of the log(D + 1))^-1 in train."""
    # (log(D + 1))^-1 / d * X     where d is the average of the ``log(D + 1))^-1`` in training
    return h * (avg_d["log"] / np.log(D + 1))


SCALERS = {
    'identity': scale_identity,
    'amplification': scale_amplification,
    'attenuation': scale_attenuation
}
