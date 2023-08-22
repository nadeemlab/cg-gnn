"""
Calculate importance scores per node in an ROI.

As used in:
"Quantifying Explainers of Graph Neural Networks in Computational Pathology",
    Jaume et al, CVPR, 2021.
"""

from typing import List

from tqdm import tqdm
from torch import FloatTensor
from torch.cuda import is_available
from dgl import DGLGraph

from cggnn.util import CellGraphModel
from cggnn.util.interpretability import (BaseExplainer, GraphLRPExplainer, GraphGradCAMExplainer,
                                         GraphGradCAMPPExplainer, GraphPruningExplainer)
from cggnn.util.constants import IMPORTANCES

IS_CUDA = is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'


def calculate_importance(cell_graphs: List[DGLGraph],
                         model: CellGraphModel,
                         explainer_model: str
                         ) -> List[DGLGraph]:
    """Calculate the importance for all cells in every graph."""
    # Define the explainer
    explainer: BaseExplainer
    explainer_model = explainer_model.lower().strip()
    if explainer_model in {'lrp', 'graphlrpexplainer'}:
        explainer = GraphLRPExplainer(model=model)
    elif explainer_model in {'cam', 'gradcam', 'graphgradcamexplainer'}:
        explainer = GraphGradCAMExplainer(model=model)
    elif explainer_model in {'pp', 'campp', 'gradcampp', 'graphgradcamppexplainer'}:
        explainer = GraphGradCAMPPExplainer(model=model)
    elif explainer_model in {'pruning', 'gnn', 'graphpruningexplainer'}:
        explainer = GraphPruningExplainer(model=model)
    else:
        raise ValueError("explainer_model not recognized.")

    # Set model to train so it'll let us do backpropogation.
    # This shouldn't be necessary since we don't want the model to change at all while running the
    # explainer. In fact, it isn't necessary when running the original histocartography code, but
    # in this version of python and torch, it results in a can't-backprop-in-eval error in torch
    # because calculating the weights requires backprop-ing to get the backward_hook.
    # TODO: Fix this.
    model = model.train()

    # Calculate the importance scores for every graph
    for graph in tqdm(cell_graphs):
        importance_scores, _ = explainer.process(graph.to(DEVICE))
        graph.ndata[IMPORTANCES] = FloatTensor(importance_scores)

    return cell_graphs
