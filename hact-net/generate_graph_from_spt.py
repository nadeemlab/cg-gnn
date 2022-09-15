"Queries the SPT server for cell-level data and slide-level labels and saves them to CSVs."
from os import path, makedirs, listdir, replace
from logging import info
from argparse import ArgumentParser
from pathlib import Path
from random import shuffle, randint
from warnings import warn
from typing import Optional, Tuple, Union, List, Dict

from torch import Tensor, FloatTensor, is_tensor
from numpy import ndarray, round, prod, percentile, argmin, nonzero
from dgl import DGLGraph
from dgl.data.utils import load_graphs, save_graphs
from sklearn.neighbors import kneighbors_graph
from pandas import read_csv, DataFrame
from scipy.spatial.distance import pdist, squareform

LABEL = "label"
CENTROID = "centroid"
FEATURES = "feat"


def parse_arguments():
    "Process command line arguments."
    parser = ArgumentParser()
    parser.add_argument(
        '--spt_csv_feat_filename',
        type=str,
        help='Path to the SPT features CSV.',
        required=True
    )
    parser.add_argument(
        '--spt_csv_label_filename',
        type=str,
        help='Path to the SPT labels CSV.',
        required=True
    )
    parser.add_argument(
        '--save_path',
        type=str,
        help='Path to save the cell graphs.',
        default='/data/',
        required=False
    )
    parser.add_argument(
        '--val_data_prc',
        type=int,
        help='Percentage of data to use as validation data. Set to 0 if you want to do k-fold '
        'cross-validation later. (Training percentage is implicit.) Default 15%.',
        default=15,
        required=False
    )
    parser.add_argument(
        '--test_data_prc',
        type=int,
        help='Percentage of data to use as the test set. (Training percentage is implicit.) '
        'Default 15%.',
        default=15,
        required=False
    )
    parser.add_argument(
        '--roi_side_length',
        type=int,
        help='Side length in pixels of the ROI areas we wish to generate.',
        default=600,
        required=False
    )
    return parser.parse_args()


def create_graphs_from_spt_csv(spt_csv_feat_filename: str,
                               spt_csv_label_filename: str,
                               output_directory: str,
                               image_size: Tuple[int, int],
                               k: int = 5,
                               thresh: Optional[int] = None
                               ) -> Dict[int, List[List[str]]]:
    "Create graphs from a feature, location, and label CSV created from SPT."

    # Read in the SPT data and convert the labels from categorical to numeric
    df_feat_all_specimens: DataFrame = read_csv(
        spt_csv_feat_filename, index_col=0)
    df_label_all_specimens: DataFrame = read_csv(
        spt_csv_label_filename, index_col=0)

    # Split the data by specimen (slide)
    filenames: Dict[str, List[str]] = {}
    for specimen, df_specimen in df_feat_all_specimens.groupby('specimen'):

        # Initialize data structures
        bboxes: List[Tuple[int, int, int, int, int, int]] = []
        slide_size = df_specimen[['center_x', 'center_y']].max() + 100
        p_tumor = df_specimen['Tumor'].sum()/df_specimen.shape[0]
        df_tumor = df_specimen.loc[df_specimen['Tumor'], :]
        d_square = squareform(pdist(df_tumor[['center_x', 'center_y']]))
        filenames[specimen] = []

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
        df_features = df_specimen.drop(
            ['center_x', 'center_y', 'specimen'], axis=1)
        label: int = df_label_all_specimens.loc[specimen, 'result']
        for i, (x_min, x_max, y_min, y_max, x, y) in enumerate(bboxes):
            df_roi = df_specimen.loc[df_specimen['center_x'].between(
                x_min, x_max) & df_specimen['center_y'].between(y_min, y_max), ]
            centroids = df_roi[['center_x', 'center_y']].values
            features = df_features.loc[df_roi.index, ].astype(int).values
            roi_name = f'melanoma_{specimen}_{i}_{image_size[0]}x{image_size[1]}_x{x}_y{y}'
            create_and_save_graph(output_directory,
                                  centroids, features, label,
                                  output_name=roi_name,
                                  k=k, thresh=thresh)
            df_roi.reset_index()['histological_structure'].to_csv(
                path.join(output_directory, 'histological_structure_ids',
                          f'{roi_name}_hist_structs.csv'))
            filenames[specimen].append(f'{roi_name}.bin')

    # Split the graphs by specimen and label
    specimen_graphs_by_label: Dict[int, List[List[str]]] = {
        i: [] for i in df_label_all_specimens['result'].unique()}
    for specimen, specimen_files in filenames.items():
        specimen_graphs_by_label[df_label_all_specimens.loc[specimen, 'result']
                                 ].append(specimen_files)
    return specimen_graphs_by_label


def create_graph(centroids: ndarray,
                 features: Tensor,
                 labels: Optional[ndarray] = None,
                 k: int = 5,
                 thresh: Optional[int] = None
                 ) -> DGLGraph:
    """Generate the graph topology from the provided instance_map using (thresholded) kNN
    Args:
        centroids (array): Node centroids
        features (Tensor): Features of each node. Shape (nr_nodes, nr_features)
        labels (array): Node levels.
        k (int, optional): Number of neighbors. Defaults to 5.
        thresh (int, optional): Maximum allowed distance between 2 nodes.
                                    Defaults to None (no thresholding).
    Returns:
        DGLGraph: The constructed graph
    """

    # add nodes
    num_nodes = features.shape[0]
    graph = DGLGraph()
    graph.add_nodes(num_nodes)
    graph.ndata[CENTROID] = FloatTensor(centroids)

    # add node features
    if not is_tensor(features):
        features = FloatTensor(features)
    graph.ndata[FEATURES] = features

    # add node labels/features
    if labels is not None:
        assert labels.shape[0] == centroids.shape[0], \
            "Number of labels do not match number of nodes"
        graph.ndata[LABEL] = FloatTensor(labels.astype(float))
        graph.ndata[LABEL] = FloatTensor(labels.astype(float))
        graph.ndata[LABEL] = FloatTensor(labels.astype(float))

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
    graph.add_edges(list(edge_list[0]), list(edge_list[1]))

    return graph


def create_and_save_graph(save_path: Union[str, Path],
                          centroids: ndarray,
                          features: Tensor,
                          label: int,
                          output_name: str = None,
                          k: int = 5,
                          thresh: Optional[int] = None
                          ) -> None:
    """Process and save graphs to provided directory
    Args:
        save_path (Union[str, Path]): Base path to save results to.
        output_name (str): Name of output file
    """
    output_path = Path(save_path) / f"{output_name}.bin"
    if output_path.exists():
        info(
            f"Output of {output_name} already exists, using it instead of recomputing")
        graphs, _ = load_graphs(str(output_path))
        assert len(graphs) == 1
        graph = graphs[0]
    else:
        graph = create_graph(
            centroids, features, k=k, thresh=thresh)
        save_graphs(str(output_path), [graph],
                    {'label': Tensor([label])})
    return graph


def split_rois(graphs_by_specimen_and_label: Dict[int, List[List[str]]],
               p_val: float, p_test: float) -> Tuple[List[str], List[str], List[str]]:
    "Randomly allocate graphs to train, val, and test sets."
    train_files: List[str] = []
    val_files: List[str] = []
    test_files: List[str] = []
    p_train = 1 - p_val - p_test

    # Shuffle the order of the specimens in each class and divvy them up.
    for label, graph_files_by_specimen in graphs_by_specimen_and_label.items():
        n_graphs = sum(len(l) for l in graph_files_by_specimen)
        if n_graphs == 0:
            warn(f'Class {label} doesn\'t have any examples.')
            continue
        shuffle(graph_files_by_specimen)

        # If there's at least one specimen of this class, add it to the training set.
        train_files += graph_files_by_specimen[0]
        n_specimens = len(graph_files_by_specimen)
        if n_specimens == 1:
            warn(
                f'Class {label} only has one specimen. Allocating to training set.')
        elif n_specimens == 2:
            if (p_val == 0) and (p_test == 0):
                train_files += graph_files_by_specimen[1]
            elif p_test == 0:
                val_files += graph_files_by_specimen[1]
            elif p_val == 0:
                test_files += graph_files_by_specimen[1]
            else:
                warn(f'Class {label} only has two specimens. '
                     'Allocating one for training and the other randomly to val or test.')
                if randint(0, 1) == 0:
                    val_files += graph_files_by_specimen[1]
                else:
                    test_files += graph_files_by_specimen[1]
        else:
            i_specimen = 1

            # First, allocate at least one example to each of the val and test sets if necessary.
            n_allocated_val = 0
            n_allocated_test = 0
            if p_val > 0:
                val_files += graph_files_by_specimen[i_specimen]
                n_allocated_val = len(graph_files_by_specimen[i_specimen])
                i_specimen += 1
            if p_test > 0:
                test_files += graph_files_by_specimen[i_specimen]
                n_allocated_test = len(graph_files_by_specimen[i_specimen])
                i_specimen += 1

            # Calculate the number of ROIs we want in the train/test/val sets, correcting for how
            # there's already one specimen allocated to each.
            n_train = n_graphs*p_train - len(graph_files_by_specimen[0])
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
            for specimen_files in graph_files_by_specimen[i_specimen:]:
                if n_used_of_remainder < n_train:
                    train_files += specimen_files
                elif n_used_of_remainder < n_train + n_val:
                    val_files += specimen_files
                else:
                    test_files += specimen_files
                n_used_of_remainder += len(specimen_files)

    return train_files, val_files, test_files


if __name__ == "__main__":

    # Handle inputs
    args = parse_arguments()
    if not (path.exists(args.spt_csv_feat_filename) and path.exists(args.spt_csv_label_filename)):
        raise ValueError("SPT CSVs to read from do not exist.")
    if not 0 <= args.val_data_prc < 100:
        raise ValueError(
            "Validation set percentage must be between 0 and 100.")
    if not 0 <= args.test_data_prc < 100:
        raise ValueError(
            "Test set percentage must be between 0 and 100.")
    if not 0 <= args.val_data_prc + args.test_data_prc < 100:
        raise ValueError(
            "Remaining data set percentage for training use must be between 0 and 50.")
    val_prop: float = args.val_data_prc/100
    test_prop: float = args.test_data_prc/100
    roi_size: Tuple[int, int] = (args.roi_side_length, args.roi_side_length)
    save_path = path.join(args.save_path)

    # Create save directory if it doesn't exist yet
    makedirs(save_path, exist_ok=True)
    makedirs(path.join(save_path,
                       'histological_structure_ids'), exist_ok=True)

    # Check if work has already been done by checking whether train, val, and test folders have
    # been created and populated
    set_directories: List[str] = []
    for set_type in ('train', 'val', 'test'):
        if (set_type is 'val') and (val_prop == 0):
            continue
        if (set_type is 'test') and (test_prop == 0):
            continue
        directory = path.join(save_path, set_type)
        if path.isdir(directory) and (len(listdir(directory)) > 0):
            raise RuntimeError(
                f'{set_type} set directory has already been created. '
                'Assuming work is done and terminating.')
        makedirs(directory, exist_ok=True)
        set_directories.append(directory)

    # Create the graphs
    graphs_by_specimen_and_label = create_graphs_from_spt_csv(
        args.spt_csv_feat_filename, args.spt_csv_label_filename, save_path, image_size=roi_size)

    # Move the train, val, and test sets into their own dedicated folders
    sets_data = split_rois(graphs_by_specimen_and_label, val_prop, test_prop)
    for i in range(3):
        for filename in sets_data[i]:
            replace(path.join(save_path, filename),
                    path.join(set_directories[i], filename))
