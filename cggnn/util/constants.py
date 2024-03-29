"""Consistent names for dict field strings."""

INDICES = 'histological_structure'
FEATURES = 'features'
CENTROIDS = 'centroid'
IMPORTANCES = 'importance'

DEFAULT_GNN_PARAMETERS = {
    'layer_type': "pna_layer",
    'output_dim': 64,
    'num_layers': 3,
    'readout_op': "lstm",
    'readout_type': "mean",
    'aggregators': "mean max min std",
    'scalers': "identity amplification attenuation",
    'avg_d': 4,
    'dropout': 0.,
    'graph_norm': True,
    'batch_norm': True,
    'towers': 1,
    'pretrans_layers': 1,
    'posttrans_layers': 1,
    'divide_input': False,
    'residual': False
}
DEFAULT_CLASSIFICATION_PARAMETERS = {
    'num_layers': 2,
    'hidden_dim': 128
}
