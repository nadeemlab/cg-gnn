"""Generates graph from saved SPT files."""

from argparse import ArgumentParser
from json import load as json_load
from typing import Dict

from pandas import read_hdf

from cggnn import generate_graphs


def parse_arguments():
    """Process command line arguments."""
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
        '--phenotype_names_by_column_name_path',
        type=str,
        help='Path to JSON translating cell DataFrame phenotype names to readable symbols.',
        required=True
    )
    parser.add_argument(
        '--validation_data_percent',
        type=int,
        help='Percentage of data to use as validation data. Set to 0 if you want to do k-fold '
        'cross-validation later. (Training percentage is implicit.) Default 15%.',
        default=15,
        required=False
    )
    parser.add_argument(
        '--test_data_percent',
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
    parser.add_argument(
        '--target_column',
        type=str,
        help='Phenotype column to use to build ROIs around.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--output_directory',
        type=str,
        help='Directory to save the (sub)directory of graph files to.',
        default=None,
        required=False
    )
    return parser.parse_args()


def read_symbols_by_column_name(path: str) -> Dict[str, str]:
    """Read in *_symbols_by_column_name JSON."""
    return {column_name: symbol for column_name, symbol in json_load(
        open(path, encoding='utf-8')).items()}


if __name__ == "__main__":
    args = parse_arguments()
    generate_graphs(read_hdf(args.spt_hdf_cell_filename), read_hdf(args.spt_hdf_label_filename),
                    read_symbols_by_column_name(args.phenotype_names_by_column_name_path),
                    args.validation_data_percent, args.test_data_percent, args.roi_side_length,
                    args.target_column, args.output_directory)
