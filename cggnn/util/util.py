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


def load_label_to_result(path: str) -> Dict[int, str]:
    """Read in label_to_result JSON."""
    return {int(label): result for label, result in json_load(
        open(path, encoding='utf-8')).items()}


class GraphData(NamedTuple):
    """Data relevant to a cell graph instance."""
    graph: DGLGraph
    label: Optional[int]
    name: str
    specimen: str
    set: Optional[Literal['train', 'validation', 'test']]


class GraphMetadata(NamedTuple):
    """Data relevant to a cell graph instance."""
    name: str
    specimen: str
    set: Optional[Literal['train', 'validation', 'test']]


def save_cell_graphs(graphs_data: List[GraphData], output_directory: str) -> None:
    """Save cell graphs to a directory."""
    graphs: List[DGLGraph] = []
    labels: List[int] = []
    metadata: List[GraphMetadata] = []
    unlabeled_graphs: List[DGLGraph] = []
    unlabeled_metadata: List[GraphMetadata] = []
    for graph_data in graphs_data:
        if graph_data.label is not None:
            graphs.append(graph_data.graph)
            metadata.append(GraphMetadata(graph_data.name,
                                          graph_data.specimen,
                                          graph_data.set))
            labels.append(graph_data.label)
        else:
            unlabeled_graphs.append(graph_data.graph)
            unlabeled_metadata.append(GraphMetadata(graph_data.name,
                                                    graph_data.specimen,
                                                    graph_data.set))
    _save_dgl_graphs(output_directory,
                     graphs + unlabeled_graphs,
                     metadata + unlabeled_metadata,
                     labels)


def _save_dgl_graphs(output_directory: str,
                     graphs: List[DGLGraph],
                     metadata: List[GraphMetadata],
                     labels: List[int]
                     ) -> None:
    """Save DGL cell graphs to a directory."""
    save_graphs(join(output_directory, 'graphs.bin'),
                graphs,
                {'labels': IntTensor(labels)})
    save_info(join(output_directory, 'graph_info.pkl'), {'info': metadata})


def load_cell_graphs(graph_directory: str) -> Tuple[List[GraphData], List[str]]:
    """Load cell graph information from a directory.

    Assumes directory contains the files `graphs.bin`, `graph_info.pkl`, and `feature_names.txt`.
    """
    graphs, labels, metadata = _load_dgl_graphs(graph_directory)
    graph_data: List[GraphData] = []
    for i, graph in enumerate(graphs):
        graph_data.append(GraphData(graph,
                                    labels[i] if i < len(labels) else None,
                                    metadata[i].name,
                                    metadata[i].specimen,
                                    metadata[i].set))
    feature_names: List[str] = loadtxt(join(graph_directory, 'feature_names.txt'),
                                       dtype=str, delimiter=',').tolist()
    return graph_data, feature_names


def _load_dgl_graphs(graph_directory: str) -> Tuple[List[DGLGraph], List[int], List[GraphMetadata]]:
    """Load cell graphs saved as DGL files from a directory."""
    graphs, labels = load_graphs(join(graph_directory, 'graphs.bin'))
    graphs: List[DGLGraph]
    labels: Dict[str, IntTensor]
    metadata: List[GraphMetadata] = load_info(join(graph_directory, 'graph_info.pkl'))['info']
    return graphs, labels['labels'].tolist(), metadata


def split_graph_sets(graphs_data: List[GraphData]) -> Tuple[Tuple[List[DGLGraph], List[int]],
                                                            Tuple[List[DGLGraph], List[int]],
                                                            Tuple[List[DGLGraph], List[int]],
                                                            List[DGLGraph]]:
    """Split graph data list into train, validation, test, and unlabeled sets."""
    cg_train: Tuple[List[DGLGraph], List[int]] = ([], [])
    cg_val: Tuple[List[DGLGraph], List[int]] = ([], [])
    cg_test: Tuple[List[DGLGraph], List[int]] = ([], [])
    cg_unlabeled: List[DGLGraph] = []

    for gd in graphs_data:
        if gd.label is None:
            cg_unlabeled.append(gd.graph)
            continue
        which_set: Tuple[List[DGLGraph], List[int]] = cg_train
        if gd.set == 'validation':
            which_set = cg_val
        elif gd.set == 'test':
            which_set = cg_test
        which_set[0].append(gd.graph)
        which_set[1].append(gd.label)
    return cg_train, cg_val, cg_test, cg_unlabeled


class CGDataset(Dataset):
    """Cell graph dataset."""

    def __init__(
        self,
        cell_graphs: List[DGLGraph],
        cell_graph_labels: Optional[List[int]] = None,
        load_in_ram: bool = False
    ):
        """Cell graph dataset constructor.

        Args:
            cell_graphs (Tuple[List[DGLGraph], List[int]]):
                Cell graphs for a given split (e.g., test) and their labels.
            load_in_ram (bool, optional): Loading data in RAM. Defaults to False.
        """
        super(CGDataset, self).__init__()

        self.cell_graphs = cell_graphs
        self.cell_graph_labels = cell_graph_labels
        self.n_cell_graphs = len(self.cell_graphs)
        self.load_in_ram = load_in_ram

    def __getitem__(self, index):
        """Get an example.

        Args:
            index (int): index of the example.
        """
        cell_graph = self.cell_graphs[index]
        if IS_CUDA:
            cell_graph = cell_graph.to('cuda:0')
        return cell_graph if (self.cell_graph_labels is None) \
            else (cell_graph, float(self.cell_graph_labels[index]))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.n_cell_graphs


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


def collate(example_batch):
    """Collate a batch.

    Args:
        example_batch (torch.tensor): a batch of examples.
    Returns:
        data: (tuple)
        labels: (torch.LongTensor)
    """
    # collate the data
    if isinstance(example_batch[0], tuple):  # graph and label
        def collate_fn(batch, id, type):
            return COLLATE_USING[type]([example[id] for example in batch])
        num_modalities = len(example_batch[0])
        return tuple([collate_fn(example_batch, mod_id, type(example_batch[0][mod_id]).__name__)
                      for mod_id in range(num_modalities)])
    else:  # graph only
        return tuple([COLLATE_USING[type(example_batch[0]).__name__](example_batch)])


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
