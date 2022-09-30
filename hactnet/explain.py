"""
Explain a cell graph (CG) prediction using a pretrained CG-GNN and a graph explainer: GraphGradCAM.

As used in:
"Quantifying Explainers of Graph Neural Networks in Computational Pathology", Jaume et al, CVPR, 2021.
"""

from os.path import join, split
from glob import glob
from typing import Optional, List, Dict

from PIL import Image
from yaml import safe_load
from tqdm import tqdm
from torch.cuda import is_available
from numpy import array, ndarray, save
from dgl import DGLGraph

from hactnet.histocartography.ml import CellGraphModel
from hactnet.histocartography.interpretability import GraphGradCAMExplainer
from hactnet.histocartography.ml.models.base_model import BaseModel
from hactnet.histocartography.visualization import OverlayGraphVisualization, InstanceImageVisualization


IS_CUDA = is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'


def explain_cell_graphs(cell_graphs_by_specimen: Dict[str, List[DGLGraph]],
                        model: Optional[BaseModel] = None,
                        config_fpath: Optional[str] = None,
                        model_checkpoint_fpath: Optional[str] = None,
                        image_path: Optional[str] = None
                        ) -> Dict[DGLGraph, ndarray]:
    """
    Generate an explanation for all the cell graphs in cell path dir.
    """

    if (model is None) and ((config_fpath is None) or (model_checkpoint_fpath is None)):
        raise ValueError(
            "Must provide either model or config and checkpoint paths.")

    # 1. get cell graph & image paths
    # cg_fnames = glob(join(cell_graph_path, '*.bin'))
    if image_path is not None:
        image_fnames = glob(join(image_path, '*.png'))

    # 2. create model and set device and mode
    if model is None:
        assert (config_fpath is not None) and (
            model_checkpoint_fpath is not None)

        with open(config_fpath, 'r', encoding='utf-8') as file:
            config = safe_load(file)

        model = CellGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=config['node_feat_dim'],
            num_classes=config['num_classes'],
            pretrained=model_checkpoint_fpath)
    model = model.to(DEVICE).eval()

    # 3. define the explainer
    explainer = GraphGradCAMExplainer(model=model)

    # 4. define graph visualizer
    # visualizer = OverlayGraphVisualization(
    #     instance_visualizer=InstanceImageVisualization(),
    #     colormap='jet',
    #     node_style="fill"
    # )

    # 4.5. Set model to train so it'll let us do backpropogation.
    #      This shouldn't be necessary since we don't want the model to change at all while running
    #      the explainer. In fact, it isn't necessary when running the original histocartography
    #      code, but in this version of python and torch, it results in a can't-backprop-in-eval
    #      error in torch because calculating the weights requires backprop-ing to get the
    #      backward_hook. TODO: Fix this.
    model = model.train()

    # 5. process all the images
    importance_scores_by_graph: Dict[DGLGraph, ndarray] = {}
    for specimen, graphs in tqdm(cell_graphs_by_specimen.items()):
        for graph in graphs:

            # b. run explainer
            importance_scores, _ = explainer.process(graph.to(DEVICE))
            assert isinstance(importance_scores, ndarray)

            # if image_path is not None:
            #     # c. load corresponding image
            #     image_path = [
            #         x for x in image_fnames if graph_name in x.replace(
            #             '.png', '.bin')][0]
            #     _, image_name = split(image_path)
            #     image = array(Image.open(image_path))

            #     # d. visualize and save the output
            #     node_attrs = {
            #         "color": importance_scores
            #     }
            #     canvas = visualizer.process(
            #         image, graph, node_attributes=node_attrs)
            #     canvas.save(join('output', 'explainer', image_name))
            # else:
            #     array_path = [x.replace('.bin', '') for x in cg_fnames][0]
            #     _, array_name = split(array_path)
            #     save(join('output', 'explainer', array_name), importance_scores)

            importance_scores_by_graph[graph] = importance_scores

    return importance_scores_by_graph
