"""Train and explain a graph neural network on a dataset of cell graphs."""

from cggnn.generate_graphs import generate_graphs
from cggnn.run import run
from cggnn.train import train, infer, infer_with_model
