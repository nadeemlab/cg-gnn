"""Cell/tissue graph dataset utility functions."""
from os import walk
from os.path import join
from importlib import import_module
from copy import deepcopy
from typing import Tuple, List, Dict, Any, Optional, Iterable, NamedTuple, Literal, Set

from torch import LongTensor, IntTensor, load
from torch.cuda import is_available
from torch.utils.data import Dataset
from dgl import batch, DGLGraph
from dgl.data.utils import load_graphs

from hactnet.util.ml.cell_graph_model import CellGraphModel


IS_CUDA = is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
COLLATE_FN = {
    'DGLGraph': batch,
    'DGLHeteroGraph': batch,
    'Tensor': lambda x: x,
    'int': lambda x: IntTensor(x).to(DEVICE),
    'int64': lambda x: IntTensor(x).to(DEVICE),
    'float': lambda x: LongTensor(x).to(DEVICE)
}
FEATURES = 'feat'

# model parameters
DEFAULT_GNN_PARAMS = {
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
DEFAULT_CLASSIFICATION_PARAMS = {
    'num_layers': 2,
    'hidden_dim': 128
}


class GraphData(NamedTuple):
    "Holds all data relevant to a cell graph instance."
    g: DGLGraph
    label: int
    name: str
    specimen: str
    train_val_test: Literal['train', 'val', 'test']


def load_cell_graphs(graph_path: str, train: bool = True, val: bool = True, test: bool = True) -> List[GraphData]:
    "Load cell graphs. Must be in graph_path/<set>/<specimen>/<graph>.bin form."

    which_sets: Set[Literal['train', 'val', 'test']] = set()
    if train:
        which_sets.add('train')
    if val:
        which_sets.add('val')
    if test:
        which_sets.add('test')

    graphs: List[GraphData] = []
    for dir_path, set_names, _ in walk(graph_path):
        for set_name in set_names:
            if set_name in {'train', 'test', 'val'}:
                for set_path, specimens, _ in walk(join(dir_path, set_name)):
                    for specimen in specimens:
                        for specimen_path, _, g_names in walk(join(set_path, specimen)):
                            for g_name in g_names:
                                assert isinstance(g_name, str)
                                if g_name.endswith('.bin'):
                                    g, l = load_graph(
                                        join(specimen_path, g_name))
                                    graphs.append(
                                        GraphData(g, l, g_name[:-4], specimen, set_name))
    return graphs


def load_graph(graph_path) -> Tuple[DGLGraph, int]:
    "Load a single graph saved in the odd histocartography method."
    graph_packet = load_graphs(graph_path)
    return graph_packet[0][0], graph_packet[1]['label'].item()


class CGDataset(Dataset):
    """Cell graph dataset."""

    def __init__(
        self,
        cell_graphs: Tuple[List[DGLGraph], List[int]],
        load_in_ram: bool = False
    ):
        """
        Cell graph dataset constructor.

        Args:
            cell_graphs (Tuple[List[DGLGraph], List[int]]):
                Cell graphs for a given split (e.g., test) and their labels.
            load_in_ram (bool, optional): Loading data in RAM. Defaults to False.
        """
        super(CGDataset, self).__init__()

        self.cell_graphs = cell_graphs[0]
        self.cell_graph_labels = cell_graphs[1]
        self.num_cg = len(self.cell_graphs)
        self.load_in_ram = load_in_ram

    def __getitem__(self, index):
        """
        Get an example.
        Args:
            index (int): index of the example.
        """
        cg = self.cell_graphs[index]
        label = float(self.cell_graph_labels[index])
        if IS_CUDA:
            cg = cg.to('cuda:0')
        return cg, label

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_cg


def instantiate_model(cell_graphs: Tuple[List[DGLGraph], List[int]],
                      gnn_params: Dict[str, Any] = DEFAULT_GNN_PARAMS,
                      classification_params: Dict[str,
                                                  Any] = DEFAULT_CLASSIFICATION_PARAMS,
                      model_checkpoint_path: Optional[str] = None
                      ) -> CellGraphModel:
    "Returns a model set up as specified."
    model = CellGraphModel(
        gnn_params=gnn_params,
        classification_params=classification_params,
        node_dim=cell_graphs[0][0].ndata['feat'].shape[1],
        num_classes=int(max(cell_graphs[1]))+1
    ).to(DEVICE)
    if model_checkpoint_path is not None:
        model.load_state_dict(load(model_checkpoint_path))
    return model


def collate(example_batch):
    """
    Collate a batch.
    Args:
        example_batch (torch.tensor): a batch of examples.
    Returns:
        data: (tuple)
        labels: (torch.LongTensor)
    """
    def collate_fn(batch, id, type):
        return COLLATE_FN[type]([example[id] for example in batch])

    # collate the data
    # should 2 if CG or TG processing or 4 if HACT
    num_modalities = len(example_batch[0])
    example_batch = tuple([collate_fn(example_batch, mod_id, type(example_batch[0][mod_id]).__name__)
                           for mod_id in range(num_modalities)])

    return example_batch


def dynamic_import_from(source_file: str, class_name: str) -> Any:
    """Do a from source_file import class_name dynamically

    Args:
        source_file (str): Where to import from
        class_name (str): What to import

    Returns:
        Any: The class to be imported
    """
    module = import_module(source_file)
    return getattr(module, class_name)


def signal_last(input_iterable: Iterable[Any]) -> Iterable[Tuple[bool, Any]]:
    iterable = iter(input_iterable)
    return_value = next(iterable)
    for value in iterable:
        yield False, return_value
        return_value = value
    yield True, return_value


def copy_graph(x):
    return deepcopy(x)


def torch_to_numpy(x):
    return x.cpu().detach().numpy()
