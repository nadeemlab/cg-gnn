"""
Query SPT PSQL database for cell-level features and slide-level labels and save to two CSV files.

Note: use with the melanoma_psql conda env and not the hactnet_hpc env.
"""
from os.path import exists
from base64 import b64decode
from mmap import mmap
from argparse import ArgumentParser
from typing import List, Union, Tuple

from psycopg2 import connect
from pandas import DataFrame, Series, read_sql
from shapefile import Reader


# Treatment response to label
RESPONSE_TO_LABEL = {
    'Untreated': 0,
    'Non-complete response': 1,
    'Complete response': 2
}


def parse_arguments():
    parser = ArgumentParser()
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


def get_targets(conn) -> DataFrame:
    "Get all target values for all cells."
    df_targets = read_sql("""
        SELECT
            eq.histological_structure,
            eq.target,
            CASE WHEN discrete_value='positive' THEN 1 ELSE 0 END AS coded_value,
            sdmp.specimen as specimen
        FROM expression_quantification eq
            JOIN histological_structure hs
                ON eq.histological_structure=hs.identifier
            JOIN histological_structure_identification hsi
                ON hs.identifier=hsi.histological_structure
            JOIN data_file df
                ON hsi.data_source=df.sha256_hash
            JOIN specimen_data_measurement_process sdmp
                ON df.source_generation_process=sdmp.identifier
        WHERE
            sdmp.study='Melanoma intralesional IL2 (Hollmann lab) - measurement'
        ORDER BY sdmp.specimen, eq.histological_structure, eq.target;
    """, conn)
    df_targets['histological_structure'] = df_targets['histological_structure'].astype(
        int)
    df_targets['target'] = df_targets['target'].astype(int)
    return df_targets


def get_phenotypes(conn) -> DataFrame:
    "Get all phenotype signatures."
    df_phenotypes = read_sql("""
        SELECT
            cp.name,
            marker,
            CASE WHEN polarity='positive' THEN 1 ELSE 0 END AS coded_value
        FROM cell_phenotype cp
            JOIN cell_phenotype_criterion cpc
                ON cpc.cell_phenotype=cp.identifier
        WHERE study='Melanoma intralesional IL2 (Hollmann lab) - data analysis';
    """, conn)
    df_phenotypes['marker'] = df_phenotypes['marker'].astype(int)
    return df_phenotypes


def get_shape_strings(conn) -> DataFrame:
    "Get the shapefile strings for each histological structure."
    df_shapes = read_sql("""
        SELECT  
            histological_structure,
            base64_contents AS shp_string
        FROM histological_structure_identification hsi
            JOIN shape_file sf
                ON hsi.shape_file=sf.identifier
            JOIN data_file df
                ON hsi.data_source=df.sha256_hash
            JOIN specimen_data_measurement_process sdmp
                ON df.source_generation_process=sdmp.identifier
        WHERE
            sdmp.study='Melanoma intralesional IL2 (Hollmann lab) - measurement'
        ORDER BY histological_structure;
    """, conn)
    df_shapes['histological_structure'] = df_shapes['histological_structure'].astype(
        int)
    return df_shapes


def extract_points(row: Series) -> Tuple[float, float]:
    "Convert shapefile string to center coordinate."
    shapefile_base64_ascii: str = row['shp_string']
    bytes_original = b64decode(shapefile_base64_ascii.encode('utf-8'))
    mm = mmap(-1, len(bytes_original))
    mm.write(bytes_original)
    mm.seek(0)
    sf = Reader(shp=mm)
    shape_type = sf.shape(0).shapeType
    shape_type_name = sf.shape(0).shapeTypeName
    # 5 is "Polygon" according to page 4 of specification
    if shape_type != 5:
        raise ValueError(f'Expected shape type index is 5, not {shape_type}.')
    if shape_type_name != 'POLYGON':
        raise ValueError(
            f'Expected shape type is "POLYGON", not {shape_type_name}.')
    coords = sf.shape(0).points[:-1]
    row['center_x'] = sum((coord[0] for coord in coords))/len(coords)
    row['center_y'] = sum((coord[1] for coord in coords))/len(coords)
    return row


def get_centroids(df: DataFrame) -> DataFrame:
    "Get the centroids from a dataframe with histological structure and shapefile strings."
    df = df.copy()
    df = df.apply(extract_points, axis=1)
    df.drop('shp_string', axis=1, inplace=True)
    df.set_index('histological_structure', inplace=True)
    return df


def create_feature_df(df_targets: DataFrame, df_phenotypes: DataFrame, df_shapes: DataFrame) -> DataFrame:
    "Create phenotype features and structure centers and merge with feature DataFrame."

    # Reorganize targets data so that the indices is the histological structure
    # and the columns are the target values / features
    columns: List[Union[int, str]] = list(range(df_targets['target'].min(),
                                                df_targets['target'].max()+1)) \
        + ['specimen']
    df = DataFrame(columns=columns,
                   index=df_targets['histological_structure'].unique())
    df.index.name = 'histological_structure'
    for hs, df_hs in df_targets.groupby('histological_structure'):
        data = df_hs[['target', 'coded_value']].sort_values(
            'target').set_index('target').T.iloc[0, ].to_dict()
        data['specimen'] = df_hs['specimen'].iloc[0]
        df.loc[hs, ] = Series(data)

    # Check if each cell matches each phenotype signature and add as features
    for phenotype, df_p in df_phenotypes.groupby('name'):
        criteria = df_p[['marker', 'coded_value']
                        ].set_index('marker').T.iloc[0]
        df[phenotype] = (df.loc[:, criteria.index] == criteria).all(axis=1)

    # Merge in the shapes
    df = df.join(df_shapes, on='histological_structure')

    return df


def create_label_df(conn) -> DataFrame:
    "Get slide-level results."
    return read_sql("SELECT slide, result FROM histology_assessment_process;",
                    conn).set_index('slide').replace(RESPONSE_TO_LABEL)


if __name__ == "__main__":
    args = parse_arguments()
    label_filename = args.output_name + '_labels.csv'
    feature_filename = args.output_name + '_features.csv'
    if not (exists(label_filename) and exists(feature_filename)):
        conn = connect(host=args.host, dbname=args.dbname,
                       user=args.user, password=args.password)
        create_label_df(conn).to_csv(label_filename)
        create_feature_df(get_targets(conn),
                          get_phenotypes(conn),
                          get_centroids(get_shape_strings(conn))
                          ).to_csv(feature_filename)
