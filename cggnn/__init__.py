"""Train and explain a graph neural network on a dataset of cell graphs."""

from cggnn.generate_graph_from_spt import generate_graphs
from cggnn.run import run
from cggnn.train import train, infer
