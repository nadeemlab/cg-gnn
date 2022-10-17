"Generates graph from saved SPT files."
from os import path, makedirs, listdir
from random import shuffle, randint
from warnings import warn
from typing import Optional, Tuple, List, Dict, DefaultDict

from torch import Tensor, FloatTensor, IntTensor
from numpy import ndarray, round, prod, percentile, argmin, nonzero
from dgl import DGLGraph, graph
from dgl.data.utils import save_graphs
from sklearn.neighbors import kneighbors_graph
from pandas import DataFrame
from scipy.spatial.distance import pdist, squareform

LABEL = "label"
CENTROID = "centroid"
FEATURES = "feat"
INDICES = "histological_structure"
PHENOTYPES = "phenotypes"


def _create_graphs_from_spt_file(df_cell_all_specimens: DataFrame,
                                 df_label_all_specimens: DataFrame,
                                 image_size: Tuple[int, int],
                                 k: int = 5,
                                 thresh: Optional[int] = None
                                 ) -> Tuple[Dict[int, Dict[str, List[DGLGraph]]],
                                            Dict[DGLGraph, str]]:
    "Create graphs from cell and label files created from SPT."

    # Split the data by specimen (slide)
    graphs_by_specimen: Dict[str, List[DGLGraph]] = DefaultDict(list)
    roi_names: Dict[DGLGraph, str] = {}
    for specimen, df_specimen in df_cell_all_specimens.groupby('specimen'):

        # Initialize data structures
        bboxes: List[Tuple[int, int, int, int, int, int]] = []
        slide_size = df_specimen[['center_x', 'center_y']].max() + 100
        p_tumor = df_specimen['PH_Tumor'].sum()/df_specimen.shape[0]
        df_tumor = df_specimen.loc[df_specimen['PH_Tumor'], :]
        d_square = squareform(pdist(df_tumor[['center_x', 'center_y']]))

        # Create as many ROIs as images will add up to the proportion of
        # the slide's cells are tumors
        n_rois = round(
            p_tumor * prod(slide_size) / prod(image_size))
        while (len(bboxes) < n_rois) and (df_tumor.shape[0] > 0):
            p_dist = percentile(d_square, p_tumor, axis=0)
            x, y = df_specimen.iloc[argmin(
                p_dist), :][['center_x', 'center_y']].tolist()
            x_min = x - image_size[0]//2
            x_max = x + image_size[0]//2
            y_min = y - image_size[1]//2
            y_max = y + image_size[1]//2
            bboxes.append((x_min, x_max, y_min, y_max, x, y))
            p_tumor -= prod(image_size) / prod(slide_size)
            cells_to_keep = ~df_tumor['center_x'].between(
                x_min, x_max) & ~df_tumor['center_y'].between(y_min, y_max)
            df_tumor = df_tumor.loc[cells_to_keep, :]
            d_square = d_square[cells_to_keep, :][:, cells_to_keep]

        # Create feature, centroid, and label arrays and then the graph
        df_features = df_specimen.loc[:,
                                      df_specimen.columns.str.startswith('FT_')]
        df_phenotypes = df_specimen.loc[:,
                                        df_specimen.columns.str.startswith('PH_')]
        for i, (x_min, x_max, y_min, y_max, x, y) in enumerate(bboxes):
            df_roi: DataFrame = df_specimen.loc[df_specimen['center_x'].between(
                x_min, x_max) & df_specimen['center_y'].between(y_min, y_max), ]
            centroids = df_roi[['center_x', 'center_y']].values
            features = df_features.loc[df_roi.index, ].astype(int).values
            phenotypes = df_phenotypes.loc[df_roi.index, ].astype(int).values
            graph_instance = _create_graph(
                df_roi.index.to_numpy(), centroids, features, phenotypes, k=k, thresh=thresh)
            graphs_by_specimen[specimen].append(graph_instance)
            roi_names[graph_instance] = \
                f'melanoma_{specimen}_{i}_{image_size[0]}x{image_size[1]}_x{x}_y{y}'

    # Split the graphs by specimen and label
    graphs_by_label_and_specimen: Dict[int,
                                       Dict[str, List[DGLGraph]]] = DefaultDict(dict)
    for specimen, graphs in graphs_by_specimen.items():
        label = df_label_all_specimens.loc[specimen, 'result']
        graphs_by_label_and_specimen[label][specimen] = graphs
    return graphs_by_label_and_specimen, roi_names


def _create_graph(node_indices: ndarray,
                  centroids: ndarray,
                  features: ndarray,
                  phenotypes: ndarray,
                  k: int = 5,
                  thresh: Optional[int] = None
                  ) -> DGLGraph:
    """Generate the graph topology from the provided instance_map using (thresholded) kNN
    Args:
        node_indices (array): Indices for each node.
        centroids (array): Node centroids
        features (array): Features of each node. Shape (nr_nodes, nr_features)
        phenotypes (array): A set of alternative features for each node based on phenotypes.
        k (int, optional): Number of neighbors. Defaults to 5.
        thresh (int, optional): Maximum allowed distance between 2 nodes.
                                    Defaults to None (no thresholding).
    Returns:
        DGLGraph: The constructed graph
    """

    # add nodes
    num_nodes = features.shape[0]
    graph_instance = graph([])
    graph_instance.add_nodes(num_nodes)
    graph_instance.ndata[INDICES] = IntTensor(node_indices)
    graph_instance.ndata[CENTROID] = FloatTensor(centroids)
    graph_instance.ndata[FEATURES] = FloatTensor(features)
    graph_instance.ndata[PHENOTYPES] = FloatTensor(phenotypes)
    # Note: features and phenotypes are binary variables, but DGL only supports FloatTensors

    # build kNN adjacency
    adj = kneighbors_graph(
        centroids,
        k,
        mode="distance",
        include_self=False,
        metric="euclidean").toarray()

    # filter edges that are too far (ie larger than thresh)
    if thresh is not None:
        adj[adj > thresh] = 0

    edge_list = nonzero(adj)
    graph_instance.add_edges(list(edge_list[0]), list(edge_list[1]))

    return graph_instance


def _split_rois(graphs_by_label_and_specimen: Dict[int, Dict[str, List[DGLGraph]]],
                p_val: float, p_test: float
                ) -> Tuple[Dict[str, List[DGLGraph]],
                           Dict[str, List[DGLGraph]],
                           Dict[str, List[DGLGraph]]]:
    "Randomly allocate graphs to train, val, and test sets."
    train_graphs: Dict[str, List[DGLGraph]] = {}
    val_graphs: Dict[str, List[DGLGraph]] = {}
    test_graphs: Dict[str, List[DGLGraph]] = {}
    p_train = 1 - p_val - p_test

    # Shuffle the order of the specimens in each class and divvy them up.
    for label, graphs_by_specimen in graphs_by_label_and_specimen.items():
        n_graphs = sum(len(l) for l in graphs_by_specimen.values())
        if n_graphs == 0:
            warn(f'Class {label} doesn\'t have any examples.')
            continue
        specimens = list(graphs_by_specimen.keys())
        shuffle(specimens)

        # If there's at least one specimen of this class, add it to the training set.
        specimen = specimens[0]
        train_graphs[specimen] = graphs_by_specimen[specimen]
        n_specimens = len(specimens)
        if n_specimens == 1:
            warn(
                f'Class {label} only has one specimen. Allocating to training set.')
        elif n_specimens == 2:
            specimen = specimens[1]
            if (p_val == 0) and (p_test == 0):
                train_graphs[specimen] = graphs_by_specimen[specimen]
            elif p_test == 0:
                val_graphs[specimen] = graphs_by_specimen[specimen]
            elif p_val == 0:
                test_graphs[specimen] = graphs_by_specimen[specimen]
            else:
                warn(f'Class {label} only has two specimens. '
                     'Allocating one for training and the other randomly to val or test.')
                if randint(0, 1) == 0:
                    val_graphs[specimen] = graphs_by_specimen[specimen]
                else:
                    test_graphs[specimen] = graphs_by_specimen[specimen]
        else:
            i_specimen = 1
            specimen = specimens[i_specimen]

            # First, allocate at least one example to each of the val and test sets if necessary.
            n_allocated_val = 0
            n_allocated_test = 0
            if p_val > 0:
                val_graphs[specimen] = graphs_by_specimen[specimen]
                n_allocated_val = len(graphs_by_specimen[specimen])
                i_specimen += 1
            if p_test > 0:
                test_graphs[specimen] = graphs_by_specimen[specimen]
                n_allocated_test = len(graphs_by_specimen[specimen])
                i_specimen += 1

            # Calculate the number of ROIs we want in the train/test/val sets, correcting for how
            # there's already one specimen allocated to each.
            n_train = n_graphs*p_train - len(graphs_by_specimen[specimen])
            n_val = n_graphs*p_val - n_allocated_val
            n_test = n_graphs*p_test - n_allocated_test
            if (n_train < 0) or (n_val < 0) or (n_test < 0):
                which_sets: List[str] = []
                if n_train < 0:
                    which_sets.append('train')
                if n_val < 0:
                    which_sets.append('val')
                if n_test < 0:
                    which_sets.append('test')
                warn(f'Class {label} doesn\'t have enough specimens to maintain the specified '
                     f'{"/".join(which_sets)} proportion. Consider adding more samples of this '
                     'class and/or increasing their allocation percentage.')

            # Finish the allocation.
            # This method prioritizes bolstering the training and validation sets in that order.
            n_used_of_remainder = 0
            for specimen in specimens[i_specimen:]:
                specimen_files = graphs_by_specimen[specimen]
                if n_used_of_remainder < n_train:
                    train_graphs[specimen] = specimen_files
                elif n_used_of_remainder < n_train + n_val:
                    val_graphs[specimen] = specimen_files
                else:
                    test_graphs[specimen] = specimen_files
                n_used_of_remainder += len(specimen_files)

    return train_graphs, val_graphs, test_graphs


def generate_graphs(df_feat_all_specimens: DataFrame,
                    df_label_all_specimens: DataFrame,
                    val_data_prc: int,
                    test_data_prc: int,
                    roi_side_length: int,
                    save_path: Optional[str] = None
                    ) -> Tuple[Tuple[Dict[str, List[DGLGraph]],
                                     Dict[str, List[DGLGraph]],
                                     Dict[str, List[DGLGraph]]],
                               Dict[DGLGraph, int],
                               Dict[DGLGraph, str]]:
    "Query the SPT server for cell-level data and slide-level labels and saves them to file."

    # Handle inputs
    if not 0 <= val_data_prc < 100:
        raise ValueError(
            "Validation set percentage must be between 0 and 100.")
    if not 0 <= test_data_prc < 100:
        raise ValueError(
            "Test set percentage must be between 0 and 100.")
    if not 0 <= val_data_prc + test_data_prc < 100:
        raise ValueError(
            "Remaining data set percentage for training use must be between 0 and 50.")
    val_prop: float = val_data_prc/100
    test_prop: float = test_data_prc/100
    roi_size: Tuple[int, int] = (roi_side_length, roi_side_length)

    if save_path is not None:
        # Create save directory if it doesn't exist yet
        makedirs(save_path, exist_ok=True)
        makedirs(path.join(save_path,
                           'histological_structure_ids'), exist_ok=True)

        # Check if work has already been done by checking whether train, val, and test folders have
        # been created and populated
        set_directories: List[str] = []
        for set_type in ('train', 'val', 'test'):
            if (set_type == 'val') and (val_prop == 0):
                continue
            if (set_type == 'test') and (test_prop == 0):
                continue
            directory = path.join(save_path, set_type)
            if path.isdir(directory) and (len(listdir(directory)) > 0):
                raise RuntimeError(
                    f'{set_type} set directory has already been created. '
                    'Assuming work is done and terminating.')
            makedirs(directory, exist_ok=True)
            set_directories.append(directory)

    # Create the graphs
    graphs_by_label_and_specimen, graph_names = _create_graphs_from_spt_file(
        df_feat_all_specimens, df_label_all_specimens, image_size=roi_size)

    # Split graphs into train/val/test sets as requested
    sets_data = _split_rois(graphs_by_label_and_specimen, val_prop, test_prop)

    # Create dict of graph to label
    graph_to_label: Dict[DGLGraph, int] = {}
    for label, graphs_by_specimen in graphs_by_label_and_specimen.items():
        for graph_list in graphs_by_specimen.values():
            for graph_instance in graph_list:
                graph_to_label[graph_instance] = label

    # Write graphs to file in train/val/test sets if requested
    if save_path is not None:
        for i, set_data in enumerate(sets_data):
            for graphs in set_data.values():
                if (len(graphs) > 0) and (len(set_directories) < i):
                    raise RuntimeError(
                        'Created a val or test entry that shouldn\' be there.')
                for graph_instance in graphs:
                    save_graphs(path.join(set_directories[i], graph_names[graph_instance] + '.bin'),
                                [graph_instance],
                                {'label': Tensor([graph_to_label[graph_instance]])})

    return sets_data, graph_to_label, graph_names