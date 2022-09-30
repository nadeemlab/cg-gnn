"""
Query SPT PSQL database for cell-level features and slide-level labels and save to two CSV files.

Note: use with the melanoma_psql conda env and not the hactnet_hpc env.
"""
from argparse import ArgumentParser

from hactnet.spt_to_file import spt_to_dataframes


def parse_arguments():
    "Process command line arguments."
    parser = ArgumentParser()
    parser.add_argument(
        '--measurement_study',
        type=str,
        help='Name of the measurement study table in SPT.',
        required=True
    )
    parser.add_argument(
        '--analysis_study',
        type=str,
        help='Name of the analysis study table in SPT.',
        required=True
    )
    parser.add_argument(
        '--specimen_study',
        type=str,
        help='Name of the specimen study table in SPT.',
        required=True
    )
    parser.add_argument(
        '--output_name',
        type=str,
        help='What to call the resulting CSVs.',
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    spt_to_dataframes(args.analysis_study, args.measurement_study, args.specimen_study, args.host,
                      args.dbname, args.user, args.password, args.output_name)
