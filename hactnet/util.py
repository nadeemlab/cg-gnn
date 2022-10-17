"""Cell/tissue graph dataset utility functions."""
from os.path import join
from glob import glob
from typing import Tuple, List

from torch import LongTensor, IntTensor
from torch.cuda import is_available
from torch.utils.data import Dataset
from dgl import batch, DGLGraph
from dgl.data.utils import load_graphs


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


def load_cell_graphs(graph_path: str) -> Tuple[List[DGLGraph], List[int]]:
    "Load cell graphs."
    cg_fnames = glob(join(graph_path, '*.bin'))
    cg_fnames.sort()
    graph_packets = [load_graphs(join(
        graph_path, fname)) for fname in cg_fnames]
    graphs = [entry[0][0] for entry in graph_packets]
    graph_labels = [entry[1]['label'].item() for entry in graph_packets]
    return graphs, graph_labels


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
        label = self.cell_graph_labels[index]
        if IS_CUDA:
            cg = cg.to('cuda:0')
        return cg, label

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_cg


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
