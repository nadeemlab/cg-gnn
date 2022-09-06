"TODO: docstring"
from os import path, makedirs, listdir, replace
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, Union, List

import dgl
import numpy as np
import torch
from dgl.data.utils import load_graphs, save_graphs
from sklearn.neighbors import kneighbors_graph
from pandas import read_csv, DataFrame
from scipy.spatial.distance import pdist, squareform


# BRACS subtype to 7-class label
TUMOR_TYPE_TO_LABEL = {
    'Untreated': 0,
    'Non-complete response': 1,
    'Complete response': 2
}

LABEL = "label"
CENTROID = "centroid"
FEATURES = "feat"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--spt_csv_path',
        type=str,
        help='Path to the SPT CSV.',
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
        help='Percentage of data to use as validation and test set, each. Must be between 0% and 50%.',
        default='15',
        required=False
    )
    return parser.parse_args()


def create_graphs_from_spt_csv(spt_csv_filename: str,
                               output_directory: str,
                               image_size: Tuple[int, int] = (1600, 1600),
                               k: int = 5,
                               thresh: Optional[int] = None
                               ) -> None:
    "Create graphs from a feature, location, and label CSV created from SPT."

    # Read in the SPT data and convert the labels from categorical to numeric
    df_all_specimens: DataFrame = read_csv(spt_csv_filename)
    df_all_specimens['result'].replace(TUMOR_TYPE_TO_LABEL, inplace=True)

    # Split the data by specimen (slide)
    for specimen, df_specimen in df_all_specimens.groupby('specimen'):

        # Initialize data structures
        bboxes: List[Tuple[int, int, int, int, int, int]] = []
        slide_size = df_specimen[['center_x', 'center_y']].max() + 100
        p_tumor = df_specimen['Tumor'].sum()/df_specimen.shape[0]
        df_tumor = df_specimen.loc[df_specimen['Tumor'], :]
        d_square = squareform(pdist(df_tumor[['center_x', 'center_y']]))

        # Create as many ROIs as images will add up to the proportion of
        # the slide's cells are tumors
        n_rois = np.round(
            p_tumor * np.prod(slide_size) / np.prod(image_size))
        while (len(bboxes) < n_rois) and (df_tumor.shape[0] > 0):
            p_dist = np.percentile(d_square, p_tumor, axis=0)
            x, y = df_specimen.iloc[np.argmin(
                p_dist), :][['center_x', 'center_y']].tolist()
            x_min = x - image_size[0]//2
            x_max = x + image_size[0]//2
            y_min = y - image_size[1]//2
            y_max = y + image_size[1]//2
            bboxes.append((x_min, x_max, y_min, y_max, x, y))
            p_tumor -= np.prod(image_size) / np.prod(slide_size)
            cells_to_keep = ~df_tumor['center_x'].between(
                x_min, x_max) & ~df_tumor['center_y'].between(y_min, y_max)
            df_tumor = df_tumor.loc[cells_to_keep, :]
            d_square = d_square[cells_to_keep, :][:, cells_to_keep]

        # Create feature, centroid, and label arrays and then the graph
        df_features = df_specimen.drop(
            ['center_x', 'center_y', 'specimen', 'result'], axis=1)
        df_labels = df_specimen[['result']]
        for i, (x_min, x_max, y_min, y_max, x, y) in enumerate(bboxes):
            df_roi = df_specimen.loc[df_specimen['center_x'].between(
                x_min, x_max) & df_specimen['center_y'].between(y_min, y_max), ]
            centroids = df_roi[['center_x', 'center_y']].values
            features = df_features.loc[df_roi.index, ].astype(int).values
            labels = df_labels.loc[df_roi.index, ].values
            roi_name = f'melanoma_{specimen}_{i}_{image_size[0]}x{image_size[1]}_x{x}_y{y}'
            create_and_save_graph(output_directory,
                                  centroids, features, labels,
                                  output_name=roi_name,
                                  k=k, thresh=thresh)
            df_roi.reset_index()['histological_structure'].to_csv(
                path.join(output_directory, 'histological_structure_ids',
                          f'{roi_name}_hist_structs.csv'))


def create_graph(centroids: np.ndarray,
                 features: torch.Tensor,
                 labels: np.ndarray,
                 k: int = 5,
                 thresh: Optional[int] = None
                 ) -> dgl.DGLGraph:
    """Generate the graph topology from the provided instance_map using (thresholded) kNN
    Args:
        centroids (np.array): Node centroids
        features (torch.Tensor): Features of each node. Shape (nr_nodes, nr_features)
        labels (np.array): Node levels.
        k (int, optional): Number of neighbors. Defaults to 5.
        thresh (int, optional): Maximum allowed distance between 2 nodes.
                                    Defaults to None (no thresholding).
    Returns:
        dgl.DGLGraph: The constructed graph
    """

    # add nodes
    num_nodes = features.shape[0]
    graph = dgl.DGLGraph()
    graph.add_nodes(num_nodes)
    graph.ndata[CENTROID] = torch.FloatTensor(centroids)

    # add node features
    if not torch.is_tensor(features):
        features = torch.FloatTensor(features)
    graph.ndata[FEATURES] = features

    # add node labels/features
    assert labels.shape[0] == centroids.shape[0], \
        "Number of labels do not match number of nodes"
    graph.ndata[LABEL] = torch.FloatTensor(labels.astype(float))

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

    edge_list = np.nonzero(adj)
    graph.add_edges(list(edge_list[0]), list(edge_list[1]))

    return graph


def create_and_save_graph(save_path: Union[str, Path],
                          centroids: np.ndarray,
                          features: torch.Tensor,
                          labels: np.ndarray,
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
        logging.info(
            f"Output of {output_name} already exists, using it instead of recomputing"
        )
        graphs, _ = load_graphs(str(output_path))
        assert len(graphs) == 1
        graph = graphs[0]
    else:
        graph = create_graph(
            centroids, features, labels, k=k, thresh=thresh)
        save_graphs(str(output_path), [graph])
    return graph


if __name__ == "__main__":

    # Handle inputs
    args = parse_arguments()
    if not path.exists(args.spt_csv_path):
        raise ValueError("SPT CSV to read from does not exist.")
    if not (0 < args.val_data_prc < 50):
        raise ValueError(
            "Validation/test set percentage must be between 0 and 50.")
    val_prop: float = args.val_data_prc/100

    save_path = path.join(args.save_path)

    # Create save directory if it doesn't exist yet
    makedirs(save_path, exist_ok=True)
    makedirs(path.join(save_path,
                       'histological_structure_ids'), exist_ok=True)

    # Check if work has already been done.
    set_directories: List[str] = []
    for set_type in ('train', 'val', 'test'):
        directory = path.join(save_path, set_type)
        if path.isdir(directory) and (len(listdir(directory)) > 0):
            raise RuntimeError(
                f'{set_type} set directory has already been created. Assuming work is done and terminating.')
        makedirs(directory)
        set_directories.append(directory)

    # Create the graphs
    create_graphs_from_spt_csv(args.spt_csv_path, save_path)

    # Count up all the nets created and prepare for aggregation
    net_filenames: List[str] = []
    for filename in listdir(save_path):
        if filename.endswith('.bin'):
            net_filenames.append(filename)

    assert len(net_filenames)*val_prop > 0, \
        f"Validation set percentage too small for list of size {len(net_filenames)}."
    assert len(net_filenames)*(1 - 2*val_prop) > 0, \
        f"Training set percentage too small for list of size {len(net_filenames)}."

    # Randomly allocate set percentage as train, val, test
    np.random.shuffle(net_filenames)
    validate, test, train = np.split(net_filenames,
                                     [int(val_prop*len(net_filenames)),
                                      int(2*val_prop*len(net_filenames))])
    sets_data = (train, validate, test)

    # Create new files.
    for i in range(3):
        for filename in sets_data[i]:
            replace(path.join(save_path, filename),
                    path.join(set_directories[i], filename))
