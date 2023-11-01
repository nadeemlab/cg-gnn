"""Run through the entire SPT CG-GNN pipeline."""

from argparse import ArgumentParser

from pandas import read_hdf  # type: ignore

from cggnn import run
from cggnn.util import load_label_to_result


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
        '--label_to_result_path',
        type=str,
        help='Where to find the data mapping label ints to their string results.',
        required=False
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
        '--disable_channels',
        action='store_true',
        help='Disable the use of channel information in the graph.',
    )
    parser.add_argument(
        '--disable_phenotypes',
        action='store_true',
        help='Disable the use of phenotype information in the graph.',
    )
    parser.add_argument(
        '--roi_side_length',
        type=int,
        help='Side length in pixels of the ROI areas we wish to generate.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--cells_per_slide_target',
        type=int,
        help='Used with the median cell density across all slides to determine the ROI size.',
        default=5_000,
        required=False
    )
    parser.add_argument(
        '--target_name',
        type=str,
        help='If given, build ROIs based only on cells with true values in this DataFrame column.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--in_ram',
        help='If the data should be stored in RAM.',
        action='store_true',
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        help='Batch size to use during training.',
        default=1,
        required=False
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs to do.',
        default=10,
        required=False
    )
    parser.add_argument(
        '-l',
        '--learning_rate',
        type=float,
        help='Learning rate to use during training.',
        default=10e-3,
        required=False
    )
    parser.add_argument(
        '-k',
        '--k_folds',
        type=int,
        help='Folds to use in k-fold cross validation. 0 means don\'t use k-fold cross validation '
        'unless no validation dataset is provided, in which case k defaults to 3.',
        required=False,
        default=0
    )
    parser.add_argument(
        '--explainer',
        type=str,
        help='Which explainer type to use.',
        default='pp',
        required=False
    )
    parser.add_argument(
        '--merge_rois',
        help='Merge ROIs together by specimen.',
        action='store_true'
    )
    parser.add_argument(
        '--prune_misclassified',
        help='Remove entries for misclassified cell graphs when calculating separability scores.',
        action='store_true'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        help='Random seed to use for reproducibility.',
        default=None,
        required=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run(read_hdf(args.spt_hdf_cell_filename),  # type: ignore
        read_hdf(args.spt_hdf_label_filename),  # type: ignore
        load_label_to_result(args.label_to_result_path),
        validation_data_percent=args.validation_data_percent,
        test_data_percent=args.test_data_percent,
        use_channels=not args.disable_channels,
        use_phenotypes=not args.disable_phenotypes,
        roi_side_length=args.roi_side_length,
        cells_per_slide_target=args.cells_per_slide_target,
        target_name=args.target_name,
        in_ram=args.in_ram,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        k_folds=args.k_folds,
        explainer_model=args.explainer,
        merge_rois=args.merge_rois,
        prune_misclassified=args.prune_misclassified,
        random_seed=args.random_seed)
