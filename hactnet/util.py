"""Cell/tissue graph dataset utility functions."""
from os.path import join
from glob import glob
from typing import Tuple, List, Optional

from h5py import File
from torch import LongTensor, IntTensor, from_numpy
from torch.cuda import is_available
from torch.utils.data import Dataset
from numpy import array
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


def h5_to_tensor(h5_path):
    h5_object = File(h5_path, 'r')
    out = from_numpy(array(h5_object['assignment_matrix']))
    return out


def load_cgtg_graphs(graph_path: str) -> Tuple[List[DGLGraph], List[int]]:
    "Load cell or tissue graphs."
    cg_fnames = glob(join(graph_path, '*.bin'))
    cg_fnames.sort()
    graph_packets = [load_graphs(join(
        graph_path, fname)) for fname in cg_fnames]
    graphs = [entry[0][0] for entry in graph_packets]
    graph_labels = [entry[1]['label'].item() for entry in graph_packets]
    return graphs, graph_labels


class CGTGDataset(Dataset):
    """Cell/tissue graph dataset."""

    def __init__(
        self,
        cell_graphs: Optional[Tuple[List[DGLGraph], List[int]]] = None,
        tissue_graphs: Optional[Tuple[List[DGLGraph], List[int]]] = None,
        assign_mat_path: str = None,
        load_in_ram: bool = False,
    ):
        """
        Cell graph and/or tissue graph dataset constructor.

        Args:
            cell_graphs (Optional[Tuple[List[DGLGraph], List[int]]]):
                Cell graphs for a given split (e.g., test) and their labels. Defaults to None.
            tissue_graphs (Optional[Tuple[List[DGLGraph], List[int]]]):
                Tissue graphs for a given split (e.g., test) and their labels. Defaults to None.
            assign_mat_path (str, optional): Assignment matrices path. Defaults to None.
            load_in_ram (bool, optional): Loading data in RAM. Defaults to False.
        """
        super(CGTGDataset, self).__init__()

        if (cell_graphs is None) and (tissue_graphs is None):
            raise ValueError("You must provide at least 1 modality.")

        if cell_graphs is not None:
            self.cell_graphs = cell_graphs[0]
            self.cell_graph_labels = cell_graphs[1]
            self.num_cg = len(self.cell_graphs)
        if tissue_graphs is not None:
            self.tissue_graphs = tissue_graphs[0]
            self.tissue_graph_labels = tissue_graphs[1]
            self.num_tg = len(self.tissue_graphs)
        self.assign_mat_path = assign_mat_path
        self.load_in_ram = load_in_ram

        if assign_mat_path is not None:
            self._load_assign_mat()

    def _load_assign_mat(self):
        """
        Load assignment matrices
        """
        self.assign_fnames = glob(join(self.assign_mat_path, '*.h5'))
        self.assign_fnames.sort()
        self.num_assign_mat = len(self.assign_fnames)
        if self.load_in_ram:
            self.assign_matrices = [
                h5_to_tensor(join(
                    self.assign_mat_path, fname)).float().t()
                for fname in self.assign_fnames
            ]

    def __getitem__(self, index):
        """
        Get an example.
        Args:
            index (int): index of the example.
        """

        # 1. HACT configuration
        if hasattr(self, 'num_tg') and hasattr(self, 'num_cg'):
            cg = self.cell_graphs[index]
            tg = self.tissue_graphs[index]
            assert self.cell_graph_labels[index] == self.tissue_graph_labels[
                index], "The CG and TG are not the same. There was an issue while creating HACT."
            label = self.cell_graph_labels[index]

            if self.load_in_ram:
                assign_mat = self.assign_matrices[index]
            else:
                assign_mat = h5_to_tensor(
                    self.assign_fnames[index]).float().t()

            if IS_CUDA:
                cg = cg.to('cuda:0')
                tg = tg.to('cuda:0')
            assign_mat = assign_mat.cuda() if IS_CUDA else assign_mat

            return cg, tg, assign_mat, label

        # 2. TG-GNN configuration
        elif hasattr(self, 'num_tg'):
            tg = self.tissue_graphs[index]
            label = self.tissue_graph_labels[index]
            if IS_CUDA:
                tg = tg.to('cuda:0')
            return tg, label

        # 3. CG-GNN configuration
        else:
            cg = self.cell_graphs[index]
            label = self.cell_graph_labels[index]
            if IS_CUDA:
                cg = cg.to('cuda:0')
            return cg, label

    def __len__(self):
        """Return the number of samples in the dataset."""
        if hasattr(self, 'num_cg'):
            return self.num_cg
        else:
            return self.num_tg


def collate(batch):
    """
    Collate a batch.
    Args:
        batch (torch.tensor): a batch of examples.
    Returns:
        data: (tuple)
        labels: (torch.LongTensor)
    """
    def collate_fn(batch, id, type):
        return COLLATE_FN[type]([example[id] for example in batch])

    # collate the data
    # should 2 if CG or TG processing or 4 if HACT
    num_modalities = len(batch[0])
    batch = tuple([collate_fn(batch, mod_id, type(batch[0][mod_id]).__name__)
                  for mod_id in range(num_modalities)])

    return batch
