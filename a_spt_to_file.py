"""Query SPT PSQL database for cell-level features and slide-level labels and save to two CSVs."""
from argparse import ArgumentParser

from cggnn.spt_to_df import spt_to_dataframes


def parse_arguments():
    """Process command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        '--study',
        type=str,
        help='Name of the study in SPT.',
        required=True
    )
    parser.add_argument(
        '--host',
        type=str,
        help='Host SQL server IP.',
        required=True
    )
    parser.add_argument(
        '--dbname',
        type=str,
        help='Database in SQL server to query.',
        required=True
    )
    parser.add_argument(
        '--user',
        type=str,
        help='Server login username.',
        required=True
    )
    parser.add_argument(
        '--password',
        type=str,
        help='Server login password.',
        required=True
    )
    parser.add_argument(
        '--output_directory',
        type=str,
        help='Where to create a subdirectory for this study (if not already created).',
        required=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    spt_to_dataframes(args.study, args.host, args.dbname,
                      args.user, args.password, args.output_directory)
