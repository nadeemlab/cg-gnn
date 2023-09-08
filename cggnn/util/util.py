"""Cell/tissue graph dataset utility functions."""

from os import walk
from os.path import join
from importlib import import_module
from copy import deepcopy
from json import load as json_load
from typing import Tuple, List, Dict, Any, Optional, Iterable, NamedTuple, Literal, Set

from torch import LongTensor, IntTensor, load
from torch.cuda import is_available
from torch.utils.data import Dataset
from dgl import batch, DGLGraph
from dgl.data.utils import load_graphs

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
    label: int
    name: str
    specimen: str
    train_validation_test: Literal['train', 'validation', 'test']


def load_cell_graphs(graph_path: str,
                     train: bool = True,
                     validation: bool = True,
                     test: bool = True) -> List[GraphData]:
    """Load cell graphs. Must be in graph_path/<set>/<specimen>/<graph>.bin form."""
    which_sets: Set[Literal['train', 'validation', 'test']] = set()
    if train:
        which_sets.add('train')
    if validation:
        which_sets.add('validation')
    if test:
        which_sets.add('test')

    graphs: List[GraphData] = []
    for directory_path, set_names, _ in walk(graph_path):
        for set_name in set_names:
            if set_name in {'train', 'test', 'validation'}:
                for set_path, specimens, _ in walk(join(directory_path, set_name)):
                    for specimen in specimens:
                        for specimen_path, _, graph_names in walk(join(set_path, specimen)):
                            for graph_name in graph_names:
                                assert isinstance(graph_name, str)
                                if graph_name.endswith('.bin'):
                                    graph, label = load_graph(
                                        join(specimen_path, graph_name))
                                    graphs.append(
                                        GraphData(graph, label, graph_name[:-4], specimen,
                                                  set_name))
    return graphs


def load_graph(graph_path) -> Tuple[DGLGraph, int]:
    """Load a single graph saved in the odd histocartography method."""
    graph_packet = load_graphs(graph_path)
    return graph_packet[0][0], graph_packet[1]['label'].item()


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


def instantiate_model(cell_graphs: Tuple[List[DGLGraph], List[int]],
                      gnn_parameters: Dict[str, Any] = DEFAULT_GNN_PARAMETERS,
                      classification_parameters: Dict[str,
                                                      Any] = DEFAULT_CLASSIFICATION_PARAMETERS,
                      model_checkpoint_path: Optional[str] = None
                      ) -> CellGraphModel:
    """Return a model set up as specified."""
    model = CellGraphModel(
        gnn_params=gnn_parameters,
        classification_params=classification_parameters,
        node_dim=cell_graphs[0][0].ndata[FEATURES].shape[1],
        num_classes=int(max(cell_graphs[1]))+1
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
