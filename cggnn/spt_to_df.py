"""
Query SPT PSQL database for cell-level attributes and slide-level labels and return as DataFrames.
"""
from os.path import exists
from base64 import b64decode
from mmap import mmap
from json import dump
from typing import List, Union, Tuple, Optional, Dict

from psycopg2 import connect
from numpy import sort
from pandas import DataFrame, Series, read_sql, read_hdf
from shapefile import Reader

from cggnn.util import load_label_to_result


def _get_targets(conn, measurement_study: str) -> DataFrame:
    "Get all target values for all cells."
    df_targets = read_sql(f"""
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
            JOIN histology_assessment_process hap
                ON sdmp.specimen=hap.slide
        WHERE
            sdmp.study='{measurement_study}' AND
            hs.anatomical_entity='cell'
        ORDER BY sdmp.specimen, eq.histological_structure, eq.target;
    """, conn)
    df_targets['histological_structure'] = df_targets['histological_structure'].astype(
        int)
    df_targets['target'] = df_targets['target'].astype(int)
    return df_targets


def _get_target_names(conn) -> Dict[int, str]:
    "Get names of each chemical species."
    s_target_names = read_sql("""
        SELECT
            identifier,
            symbol
        FROM chemical_species;
    """, conn).set_index('identifier')['symbol']
    renames: Dict[int, str] = {}
    for identifier, symbol in s_target_names.items():
        renames[int(identifier)] = 'FT_' + symbol
    return renames


def _get_phenotypes(conn, analysis_study: str) -> DataFrame:
    "Get all phenotype signatures."
    df_phenotypes = read_sql(f"""
        SELECT
            cp.name,
            marker,
            CASE WHEN polarity='positive' THEN 1 ELSE 0 END AS coded_value
        FROM cell_phenotype cp
            JOIN cell_phenotype_criterion cpc
                ON cpc.cell_phenotype=cp.identifier
        WHERE study='{analysis_study}';
    """, conn)
    df_phenotypes['marker'] = df_phenotypes['marker'].astype(int)
    return df_phenotypes


def _get_shape_strings(conn, measurement_study: str) -> DataFrame:
    "Get the shapefile strings for each histological structure."
    df_shapes = read_sql(f"""
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
            JOIN histology_assessment_process hap
                ON sdmp.specimen=hap.slide
            JOIN histological_structure hs
                ON hsi.histological_structure=hs.identifier
        WHERE
            sdmp.study='{measurement_study}' AND
            hs.anatomical_entity='cell'
        ORDER BY histological_structure;
    """, conn)
    df_shapes['histological_structure'] = df_shapes['histological_structure'].astype(
        int)
    return df_shapes


def _extract_points(row: Series) -> Tuple[float, float]:
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


def _get_centroids(df: DataFrame) -> DataFrame:
    "Get the centroids from a dataframe with histological structure and shapefile strings."
    df = df.copy()
    df = df.apply(_extract_points, axis=1)
    df.drop('shp_string', axis=1, inplace=True)
    df.set_index('histological_structure', inplace=True)
    return df


def _create_cell_df(df_targets: DataFrame,
                    target_names: Dict[int, str],
                    df_phenotypes: DataFrame,
                    df_shapes: DataFrame) -> DataFrame:
    "Find chemical species, phenotypes, and locations and merge into a DataFrame."

    # Reorganize targets data so that the indices is the histological structure
    # and the columns are the target values / chemical species
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

    # Check if each cell matches each phenotype signature and add
    for phenotype, df_p in df_phenotypes.groupby('name'):
        criteria = df_p[['marker', 'coded_value']
                        ].set_index('marker').T.iloc[0]
        df['PH_' + phenotype] = (df.loc[:, criteria.index]
                                 == criteria).all(axis=1)

    # Rename columns from target int indices to their text names
    df.rename(target_names, axis=1, inplace=True)

    # Merge in the shapes
    df = df.join(df_shapes, on='histological_structure')

    return df


def _create_label_df(conn, specimen_study: str) -> Tuple[DataFrame, Dict[int, str]]:
    "Get slide-level results."
    df = read_sql(f"""
        SELECT 
            slide,
            d.result
        FROM histology_assessment_process hap
            JOIN specimen_collection_process scp
                ON scp.specimen=hap.slide
            JOIN diagnosis d
                ON scp.source=d.subject
        WHERE
            scp.study='{specimen_study}';
    """, conn).set_index('slide')
    label_to_result = {i: res for i, res in enumerate(
        sort(df['result'].unique()))}
    return df.replace({res: i for i, res in label_to_result.items()}), label_to_result


def spt_to_dataframes(analysis_study: str,
                      measurement_study: str,
                      specimen_study: str,
                      host: str,
                      dbname: str,
                      user: str,
                      password: str,
                      output_name: Optional[str] = None
                      ) -> Tuple[DataFrame, DataFrame, Dict[int, str]]:
    "Query SPT PSQL database for cell-level attributes and slide-level labels and return."
    if output_name is not None:
        dict_filename = output_name + '_label_to_result.json'
        label_filename = output_name + '_labels.h5'
        cells_filename = output_name + '_cells.h5'
        if exists(label_filename) and exists(cells_filename) and exists(dict_filename):
            return (read_hdf(cells_filename), read_hdf(label_filename),
                    load_label_to_result(dict_filename))
    conn = connect(host=host, dbname=dbname,
                   user=user, password=password)
    df_label, label_to_result = _create_label_df(conn, specimen_study)
    df_cells = _create_cell_df(_get_targets(conn, measurement_study),
                               _get_target_names(conn),
                               _get_phenotypes(conn, analysis_study),
                               _get_centroids(_get_shape_strings(conn, measurement_study)))
    if output_name is not None:
        dump(label_to_result, open(dict_filename, 'w', encoding='utf-8'))
        df_label.to_hdf(label_filename, 'labels')
        df_cells.to_hdf(cells_filename, 'cells')
    return df_cells, df_label, label_to_result
