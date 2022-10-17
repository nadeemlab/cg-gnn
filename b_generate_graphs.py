"Generates graph from saved SPT files."
from argparse import ArgumentParser

from pandas import read_hdf

from hactnet.generate_graph_from_spt import generate_graphs


def parse_arguments():
    "Process command line arguments."
    parser = ArgumentParser()
    parser.add_argument(
        '--spt_hdf_cell_filename',
        type=str,
        help='Path to the SPT cell attributes HDF.',
        required=True
    )
    parser.add_argument(
        '--spt_hdf_label_filename',
        type=str,
        help='Path to the SPT labels HDF.',
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


if __name__ == "__main__":
    args = parse_arguments()
    generate_graphs(read_hdf(args.spt_hdf_cell_filename), read_hdf(args.spt_hdf_label_filename),
                    args.val_data_prc, args.test_data_prc, args.roi_side_length, args.save_path)
