"""Generates graphs from saved SPT files."""

from os import makedirs, listdir
from os.path import join, isdir
from random import shuffle, randint
from warnings import warn
from typing import Optional, Tuple, List, Dict, DefaultDict

from torch import Tensor, FloatTensor, IntTensor  # pylint: disable=no-name-in-module
from numpy import round, prod, percentile, argmin, nonzero, savetxt  # pylint: disable=redefined-builtin
from numpy.typing import NDArray
from dgl import DGLGraph, graph  # type: ignore
from dgl.data.utils import save_graphs  # type: ignore
from sklearn.neighbors import kneighbors_graph  # type: ignore
from pandas import DataFrame
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from cggnn.util import GraphData
from cggnn.util.constants import CENTROIDS, FEATURES, INDICES, TRAIN_VALIDATION_TEST


def _create_graphs_from_spt(df_cell: DataFrame,
                            df_label: DataFrame,
                            roi_size: Tuple[int, int],
                            use_channels: bool = True,
                            use_phenotypes: bool = True,
                            target_name: Optional[str] = None,
                            n_neighbors: int = 5,
                            threshold: Optional[int] = None
                            ) -> Tuple[Dict[int, Dict[str, List[DGLGraph]]], Dict[DGLGraph, str],
                                       List[str]]:
    """Create graphs from cell and label files created from SPT."""
    if df_label['label'].nunique() < 2:
        raise ValueError('Less than two unique labels. No point to training.')
    if (not use_channels) and (not use_phenotypes):
        raise ValueError('Must use at least one of channels or phenotypes.')

    features_to_use: List[str] = []
    channels = df_cell.columns[df_cell.columns.str.startswith('C ')]
    phenotypes = df_cell.columns[df_cell.columns.str.startswith('P ')]
    if use_channels:
        features_to_use.extend(channels)
    else:
        df_cell.drop(columns=channels, inplace=True)
    if use_phenotypes:
        features_to_use.extend(phenotypes)
    else:
        df_cell.drop(columns=phenotypes, inplace=True)
    if len(features_to_use) == 0:
        raise ValueError('No features to use.')

    roi_area = prod(roi_size)

    # Split the data by specimen (slide)
    graphs_by_specimen: Dict[str, List[DGLGraph]] = DefaultDict(list)
    roi_names: Dict[DGLGraph, str] = {}
    print('Creating graphs for identified regions in each specimen...')
    for specimen, df_specimen in tqdm(df_cell.groupby('specimen')):

        # Skip specimens without labels
        if specimen not in df_label.index:
            continue

        # Initialize data structures
        bounding_boxes: List[Tuple[int, int, int, int, int, int]] = []
        slide_size = df_specimen[['pixel x', 'pixel y']].max() + 100
        if target_name is not None:
            # Invert the column to name dict to find the column name for the target
            proportion_of_target = df_specimen[target_name].sum()/df_specimen.shape[0]
            df_target = df_specimen.loc[df_specimen[target_name], :]
        else:
            proportion_of_target = 1.
            df_target = df_specimen
        distance_square = squareform(pdist(df_target[['pixel x', 'pixel y']]))
        slide_area = prod(slide_size)

        # Create as many ROIs such that the total area of the ROIs will equal the area of the source
        # image times the proportion of cells on that image that have the target phenotype
        n_rois = round(proportion_of_target * slide_area / roi_area)
        while (len(bounding_boxes) < n_rois) and (df_target.shape[0] > 0):
            p_dist = percentile(distance_square, proportion_of_target, axis=0)
            x, y = df_specimen.iloc[argmin(p_dist), :][['pixel x', 'pixel y']].tolist()
            x_min = x - roi_size[0]//2
            x_max = x + roi_size[0]//2
            y_min = y - roi_size[1]//2
            y_max = y + roi_size[1]//2

            # Check that this bounding box contains enough cells to do nearest neighbors on
            if (df_specimen['pixel x'].between(x_min, x_max) &
                    df_specimen['pixel y'].between(y_min, y_max)).shape[0] < n_neighbors + 1:
                # If not, terminate the ROI creation process early
                break

            # Log the new bounding box and track which and how many cells haven't been captured yet
            bounding_boxes.append((x_min, x_max, y_min, y_max, x, y))
            proportion_of_target -= roi_area / slide_area
            cells_not_yet_captured = ~(df_target['pixel x'].between(x_min, x_max) &
                                       df_target['pixel y'].between(y_min, y_max))
            df_target = df_target.loc[cells_not_yet_captured, :]
            distance_square = distance_square[cells_not_yet_captured, :][:, cells_not_yet_captured]

        # Create features, centroid, and label arrays and then the graph
        for i, (x_min, x_max, y_min, y_max, x, y) in enumerate(bounding_boxes):
            df_roi: DataFrame = df_specimen.loc[df_specimen['pixel x'].between(x_min, x_max) &
                                                df_specimen['pixel y'].between(y_min, y_max), ]
            centroids = df_roi[['pixel x', 'pixel y']].values
            features = df_roi[features_to_use].astype(int).values
            graph_instance = _create_graph(
                df_roi.index.to_numpy(), centroids, features, n_neighbors=n_neighbors,
                threshold=threshold)
            graphs_by_specimen[specimen].append(graph_instance)
            roi_names[graph_instance] = \
                f'melanoma_{specimen}_{i}_{roi_size[0]}x{roi_size[1]}_x{x}_y{y}'

    # Split the graphs by specimen and label
    graphs_by_label_and_specimen: Dict[int, Dict[str, List[DGLGraph]]] = DefaultDict(dict)
    for specimen, graphs in graphs_by_specimen.items():
        label = df_label.loc[specimen, 'label']
        graphs_by_label_and_specimen[label][specimen] = graphs
    return graphs_by_label_and_specimen, roi_names, features_to_use


def _create_graph(node_indices: NDArray,
                  centroids: NDArray,
                  features: NDArray,
                  n_neighbors: int = 5,
                  threshold: Optional[int] = None
                  ) -> DGLGraph:
    """Generate the graph topology from the provided instance_map using (thresholded) kNN.

    Args:
        node_indices (array): Indices for each node.
        centroids (array): Node centroids
        features (array): Features of each node based on chemical channels.
        n_neighbors (int, optional): Number of neighbors. Defaults to 5.
        threshold (int, optional): Maximum allowed distance between 2 nodes.
                                    Defaults to None (no thresholding).
    Returns:
        DGLGraph: The constructed graph
    """
    # add nodes
    num_nodes = features.shape[0]
    graph_instance = graph([])
    graph_instance.add_nodes(num_nodes)
    graph_instance.ndata[INDICES] = IntTensor(node_indices)
    graph_instance.ndata[CENTROIDS] = FloatTensor(centroids)
    graph_instance.ndata[FEATURES] = FloatTensor(features)
    # Note: channels and phenotypes are binary variables, but DGL only supports FloatTensors

    # build kNN adjacency
    adj = kneighbors_graph(
        centroids,
        n_neighbors,
        mode="distance",
        include_self=False,
        metric="euclidean").toarray()

    # filter edges that are too far (i.e., larger than the threshold)
    if threshold is not None:
        adj[adj > threshold] = 0

    edge_list = nonzero(adj)
    graph_instance.add_edges(list(edge_list[0]), list(edge_list[1]))

    return graph_instance


def _split_rois(graphs_by_label_and_specimen: Dict[int, Dict[str, List[DGLGraph]]],
                p_validation: float, p_test: float
                ) -> Tuple[Dict[str, List[DGLGraph]],
                           Dict[str, List[DGLGraph]],
                           Dict[str, List[DGLGraph]]]:
    """Randomly allocate graphs to train, validation, and test sets."""
    train_graphs: Dict[str, List[DGLGraph]] = {}
    validation_graphs: Dict[str, List[DGLGraph]] = {}
    test_graphs: Dict[str, List[DGLGraph]] = {}
    p_train = 1 - p_validation - p_test

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
            if (p_validation == 0) and (p_test == 0):
                train_graphs[specimen] = graphs_by_specimen[specimen]
            elif p_test == 0:
                validation_graphs[specimen] = graphs_by_specimen[specimen]
            elif p_validation == 0:
                test_graphs[specimen] = graphs_by_specimen[specimen]
            else:
                warn(f'Class {label} only has two specimens. '
                     'Allocating one for training and the other randomly to validation or test.')
                if randint(0, 1) == 0:
                    validation_graphs[specimen] = graphs_by_specimen[specimen]
                else:
                    test_graphs[specimen] = graphs_by_specimen[specimen]
        else:
            # Prepare to iterate through the remaining specimens.
            i_specimen: int = 1
            specimen = specimens[i_specimen]

            # Allocate at least one specimen to each of the validation and test sets if necessary.
            n_allocated_val = 0
            n_allocated_test = 0
            if p_validation > 0:
                validation_graphs[specimen] = graphs_by_specimen[specimen]
                n_allocated_val = len(graphs_by_specimen[specimen])
                i_specimen += 1
            if p_test > 0:
                test_graphs[specimen] = graphs_by_specimen[specimen]
                n_allocated_test = len(graphs_by_specimen[specimen])
                i_specimen += 1

            # Calculate the number of ROIs we want in the train/test/validation sets, correcting
            # for how there's already one specimen allocated to each.
            n_train = n_graphs*p_train - len(graphs_by_specimen[specimens[0]])
            n_validation = n_graphs*p_validation - n_allocated_val
            n_test = n_graphs*p_test - n_allocated_test
            if (n_train < 0) or (n_validation < 0) or (n_test < 0):
                which_sets: List[str] = []
                if n_train < 0:
                    which_sets.append('train')
                if n_validation < 0:
                    which_sets.append('validation')
                if n_test < 0:
                    which_sets.append('test')
                warn(f'Class {label} doesn\'t have enough specimens to maintain the specified '
                     f'{"/".join(which_sets)} proportion. Consider adding more specimens of this '
                     'class and/or increasing their allocation percentage.')

            # Finish the allocation.
            # This method prioritizes bolstering the training and validation sets in that order.
            n_used_of_remainder = 0
            for specimen in specimens[i_specimen:]:
                specimen_files = graphs_by_specimen[specimen]
                if n_used_of_remainder < n_train:
                    train_graphs[specimen] = specimen_files
                elif n_used_of_remainder < n_train + n_validation:
                    validation_graphs[specimen] = specimen_files
                else:
                    test_graphs[specimen] = specimen_files
                n_used_of_remainder += len(specimen_files)

    return train_graphs, validation_graphs, test_graphs


def generate_graphs(df_cell: DataFrame,
                    df_label: DataFrame,
                    validation_data_percent: int,
                    test_data_percent: int,
                    roi_side_length: int,
                    use_channels: bool = True,
                    use_phenotypes: bool = True,
                    target_name: Optional[str] = None,
                    output_directory: Optional[str] = None
                    ) -> Tuple[List[GraphData], List[str]]:
    """Generate cell graphs from SPT server files and save to disk if requested.

    Parameters
    ----------
    df_cell: DataFrame
        Rows are individual cells, indexed by an integer ID.
        Column or column groups are, named and in order:
            1. The 'specimen' the cell is from
            2. Cell centroid positions 'pixel x' and 'pixel y'
            3. Channel expressions starting with 'C ' and followed by a human-readable symbol
            4. Phenotype expressions starting with 'P ' followed by a symbol
    df_label: DataFrame
        Rows are specimens, the sole column 'label' is its class label as an integer.
    validation_data_percent: int
    test_data_percent: int
        Percent of regions of interest (ROIs) to reserve for the validation and test sets. Actual
        percentage of ROIs may not match because different specimens may yield different ROI counts,
        and the splitting process ensures that ROIs from the same specimen are not split among
        different train/validation/test sets.
        Set validation_data_percent to 0 if you want to do k-fold cross-validation later.
        (Training data percent is calculated from these two percentages.)
    roi_side_length: int
        How long to make the side length of each square ROI, in pixels.
    use_channels: bool = True
    use_phenotypes: bool = True
        Whether to include channel or phenotype features (columns in df_cell beginning with 'C ' and
        'P ', respectively) in the graph.
    target_name: Optional[str] = None
        If provided, build ROIs based on only cells with this channel or phenotype. Should be a
        column name in df_feat_all_specimens.
        If not, orient ROIs where all cells are densest.
    output_directory: Optional[str] = None
        If provided, save the graphs to disk in the specified directory.

    Returns
    -------
    graphs_data: List[GraphData]
    feature_names: List[str]
        The names of the features in graph.ndata['features'] in the order they appear in the array.
    """
    if not 0 <= validation_data_percent < 100:
        raise ValueError("Validation set percentage must be between 0 and 100.")
    if not 0 <= test_data_percent < 100:
        raise ValueError("Test set percentage must be between 0 and 100.")
    if not 0 <= validation_data_percent + test_data_percent < 100:
        raise ValueError(
            "Remaining data set percentage for training use must be between 0 and 100.")
    p_validation: float = validation_data_percent/100
    p_test: float = test_data_percent/100
    roi_size: Tuple[int, int] = (roi_side_length, roi_side_length)

    if output_directory is not None:
        # Create graph directory if it doesn't exist yet
        output_directory = join(output_directory, 'graphs')
        makedirs(output_directory, exist_ok=True)

        # Check if work has already been done by checking whether train, validation, and test
        # folders have been created and populated
        set_directories: List[str] = []
        for set_type in TRAIN_VALIDATION_TEST:
            directory = join(output_directory, set_type)
            if isdir(directory) and (len(listdir(directory)) > 0):
                raise RuntimeError(f'{set_type} set directory has already been created. '
                                   'Assuming work is done and terminating.')
            set_directories.append(directory)

            # Ensure directory exists IFF graphs are going in it
            if (set_type == 'validation') and (p_validation == 0):
                continue
            if (set_type == 'test') and (p_test == 0):
                continue
            makedirs(directory, exist_ok=True)

    # Create the graphs
    graphs_by_label_and_specimens, graph_names, feature_names = _create_graphs_from_spt(
        df_cell, df_label, roi_size, use_channels=use_channels, use_phenotypes=use_phenotypes,
        target_name=target_name)

    # Split graphs into train/validation/test sets as requested
    sets_data = _split_rois(graphs_by_label_and_specimens, p_validation, p_test)

    # Create dict of graph to label
    graph_to_label: Dict[DGLGraph, int] = {}
    for label, graphs_by_specimen in graphs_by_label_and_specimens.items():
        for graph_list in graphs_by_specimen.values():
            for graph_instance in graph_list:
                graph_to_label[graph_instance] = label

    # Write graphs to file in train/validation/test sets if requested
    graphs_data: List[GraphData] = []
    for i, set_data in enumerate(sets_data):
        for specimen, graphs in set_data.items():
            if output_directory is not None:
                if (len(graphs) > 0) and (len(set_directories) < i):
                    raise RuntimeError(
                        'Created a validation or test entry that shouldn\'t be there.')
                specimen_directory = join(set_directories[i], specimen)
                makedirs(specimen_directory, exist_ok=True)
            for graph_instance in graphs:
                label = graph_to_label[graph_instance]
                name = graph_names[graph_instance]
                graphs_data.append(GraphData(graph_instance, label, name, specimen,
                                             TRAIN_VALIDATION_TEST[i]))
                if output_directory is not None:
                    save_graphs(join(specimen_directory, name + '.bin'),
                                [graph_instance],
                                {'label': Tensor([label])})
    if output_directory is not None:
        savetxt(join(output_directory, 'feature_names.txt'), feature_names, fmt='%s')

    return graphs_data, feature_names
