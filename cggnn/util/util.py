"""Cell/tissue graph dataset utility functions."""

from os.path import join
from importlib import import_module
from copy import deepcopy
from json import load as json_load
from random import seed
from typing import Tuple, List, Dict, Any, Optional, Iterable, NamedTuple, Literal

from numpy import loadtxt
from numpy.random import seed as np_seed
from torch import LongTensor, IntTensor, load, manual_seed, use_deterministic_algorithms  # type: ignore
from torch.cuda import is_available, manual_seed_all
from torch.cuda import manual_seed as cuda_manual_seed  # type: ignore
from torch.backends import cudnn  # type: ignore
from torch.utils.data import Dataset
from dgl import batch, DGLGraph  # type: ignore
from dgl import seed as dgl_seed  # type: ignore
from dgl.data.utils import save_graphs, save_info, load_graphs, load_info  # type: ignore

from cggnn.util.ml.cell_graph_model import CellGraphModel
from cggnn.util.constants import FEATURES, DEFAULT_GNN_PARAMETERS, DEFAULT_CLASSIFICATION_PARAMETERS


IS_CUDA = is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
COLLATE_USING = {
    'DGLGraph': batch,
    'DGLHeteroGraph': batch,
    'Tensor': lambda x: x,
    'int': lambda x: IntTensor(x).to(DEVICE),
    'int64': lambda x: IntTensor(x).to(DEVICE),
    'float': lambda x: LongTensor(x).to(DEVICE)
}


def instantiate_model(cell_graphs: List[GraphData],
                      gnn_parameters: Dict[str, Any] = DEFAULT_GNN_PARAMETERS,
                      classification_parameters: Dict[str,
                                                      Any] = DEFAULT_CLASSIFICATION_PARAMETERS,
                      model_checkpoint_path: Optional[str] = None
                      ) -> CellGraphModel:
    """Return a model set up as specified."""
    model = CellGraphModel(
        gnn_params=gnn_parameters,
        classification_params=classification_parameters,
        node_dim=cell_graphs[0].graph.ndata[FEATURES].shape[1],
        num_classes=int(max(g.label for g in cell_graphs))+1
    ).to(DEVICE)
    if model_checkpoint_path is not None:
        model.load_state_dict(load(model_checkpoint_path))
    return model


def dynamic_import_from(source_file: str, class_name: str) -> Any:
    """Import class_name from source_file dynamically.

    Args:
        source_file (str): Where to import from
        class_name (str): What to import

    Returns:
        Any: The class to be imported
    """
    module = import_module(source_file)
    return getattr(module, class_name)


def signal_last(input_iterable: Iterable[Any]) -> Iterable[Tuple[bool, Any]]:
    """Signal the last element of an iterable."""
    iterable = iter(input_iterable)
    return_value = next(iterable)
    for value in iterable:
        yield False, return_value
        return_value = value
    yield True, return_value


def copy_graph(x):
    """Copy a graph."""
    return deepcopy(x)


def torch_to_numpy(x):
    """Convert a torch tensor to a numpy array."""
    return x.cpu().detach().numpy()


def set_seeds(random_seed: int) -> None:
    """Set random seeds for all libraries."""
    seed(random_seed)
    np_seed(random_seed)
    manual_seed(random_seed)
    dgl_seed(random_seed)
    cuda_manual_seed(random_seed)
    manual_seed_all(random_seed)  # multi-GPU
    # use_deterministic_algorithms(True)
    # # multi_layer_gnn uses nondeterministic algorithm when on GPU
    # cudnn.deterministic = True
    cudnn.benchmark = False
