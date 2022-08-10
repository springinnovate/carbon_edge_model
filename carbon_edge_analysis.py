"""
********
Start with two LULC maps:

1) restoration_limited
2) ESACCI-LC-L4-LCCS

build two forest masks off of these:

3) restoration_limited_forest_mask
4) ESACCI-LC-L4-LCCS_forest_mask
5) ESA_to_restoration_new_forest_mask

Build ESA carbon map since the change is just static and covert to co2
* note, requires access to
    * IPCC_carbon_table_md5_a91f7ade46871575861005764d85cfa7
    * carbon_zones_md5_aa16830f64d1ef66ebdf2552fb8a9c0d

6) restoration_limited_new_forest
7) ESACCI-LC-L4-LCCS_new_forest

Build regression carbon maps

8) restoration_limited_regression
9) ESACCI-LC-L4-LCCS_regression

Build ESA marginal value map:

10) (calculate 6-7) ESA_marginal_value_co2_new_forest

Build regression marginal value:

* calculate convolution on new forest mask to show how many pixels from the
  new forest benefit the pixel under question
  11) new_forest_coverage_5km

* calculate 8-9 to find marginal value as a whole
  12) regression_marginal_value_raw

* calculate 12/11 to determine average coverage
  13) regression_marginal_value_average

* convolve 13 to 5km to estimate local marginal value benefit
  14) regression_marginal_value_co2_raw

* mask 14 to new forest
  15) regression_marginal_value_co2_new_forest
"""
import logging
import os
import tempfile

from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal
import pandas
import numpy

from run_model import regression_carbon_model

gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = './output'

# Base data
LULC_RESTORATION_PATH = "./ipcc_carbon_data/restoration_limited_md5_372bdfd9ffaf810b5f68ddeb4704f48f.tif"
LULC_ESA_PATH = "./ipcc_carbon_data/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif"
FOREST_LULC_CODES = (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170)

CARBON_TABLE_PATH = "./ipcc_carbon_data/IPCC_carbon_table_md5_a91f7ade46871575861005764d85cfa7.csv"
CARBON_ZONES_PATH = "./ipcc_carbon_data/carbon_zones_md5_aa16830f64d1ef66ebdf2552fb8a9c0d.gpkg"

# Forest masks created by script
FOREST_MASK_RESTORATION_PATH = './output/forest_mask_restoration_limited.tif'
FOREST_MASK_ESA_PATH = './output/forest_mask_esa.tif'
NEW_FOREST_MASK_ESA_TO_RESTORATION_PATH = './output/new_forest_mask_esa_to_restoration.tif'

# IPCC based carbon maps
IPCC_CARBON_RESTORATION_PATH = './output/ipcc_carbon_restoration_limited.tif'
IPCC_CARBON_ESA_PATH = './output/ipcc_carbon_esa.tif'

# Regression based carbon maps:
REGRESSION_CARBON_RESTORATION_PATH = './output/regression_carbon_restoration.tif'
REGRESSION_CARBON_ESA_PATH = './output/regression_carbon_esa.tif'


def build_ipcc_carbon(lulc_path, lulc_table_path, zone_path, lulc_codes, target_carbon_path):
    """Calculate IPCC carbon.

    Args:
        lulc_path (str): path to raster with LULC codes found in lulc_table_path and lulc_codes.
        lulc_table_path (str): maps carbon zone (rows) to lulc codes (columns) to map carbon total.
        zone_path (str): path to vector with carbon zones in 'CODE' field.
        lulc_codes (tuple): only evaluate codes in this tuple
        target_carbon_path (str): created raster that contains carbon values

    Returns:
        None
    """
    raster_info = geoprocessing.get_raster_info(lulc_path)

    # rasterize zones
    working_dir = tempfile.mkdtemp(dir=os.path.dirname(target_carbon_path))
    projected_zone_path = os.path.join(working_dir, 'proj_zone.gpkg')
    geoprocessing.reproject_vector(
        zone_path, raster_info['projection_wkt'], projected_zone_path,
        layer_id=0, driver_name='GPKG')

    zone_raster_path = os.path.join(working_dir, 'zone.tif')
    geoprocessing.new_raster_from_base(
        lulc_path, zone_raster_path, gdal.GDT_Int32, [None])

    geoprocessing.rasterize(
        projected_zone_path, zone_raster_path, option_list=['ATTRIBUTE=CODE'])
    # load table
    table = pandas.read_csv(lulc_table_path, index_col=0, header=0).to_dict()
    lulc_zone_carbon_map = {int(k): v for k, v in table.items()}

    # raster calculator of lulc, zones, table, and codes
    def _lulc_zone_to_carbon(lulc_array, zone_array):
        result = numpy.empty(lulc_array.shape, dtype=numpy.int)
        for lulc_code in numpy.unique(lulc_array):
            lulc_mask = lulc_array == lulc_code
            for zone_code in numpy.unique(zone_array):
                zone_mask = zone_array == zone_code
                if (lulc_code in lulc_zone_carbon_map and
                        zone_code in lulc_zone_carbon_map[lulc_code]):
                    carbon_val = lulc_zone_carbon_map[lulc_code][zone_code]
                else:
                    carbon_val = 0
                result[lulc_mask & zone_mask] = carbon_val
        return result

    geoprocessing.raster_calculator(
        [(lulc_path, 1), (zone_raster_path, 1)], _lulc_zone_to_carbon,
        target_carbon_path, gdal.GDT_Int32, None)


def create_mask(base_path, code_list, target_path):
    """Mask out base with values in code list, 0 otherwise.

    Args:
        base_path (str): path to an integer raster
        code_list (list): list of integer values to set to 1 in base
        target_path (str): path to created raster

    Returns:
        None.
    """
    def _code_mask(base_array):
        result = numpy.zeros(base_array.shape, dtype=numpy.byte)
        for code in numpy.unique(base_array):
            if code in code_list:
                result[base_array == code] = 1
        return result

    geoprocessing.raster_calculator(
        [(base_path, 1)], _code_mask, target_path, gdal.GDT_Byte, None)


def sub_rasters(base_a_path, base_b_path, target_path):
    """Result is A&B

    Args:
        base_a_path (str): path to 0/1 raster
        base_b_path (str): path to 0/1 raster
        target_path (str): path to created raster of A&B

    Returns:
        None.
    """
    def _sub(base_a_array, base_b_array):
        return base_a_array > base_b_array

    geoprocessing.raster_calculator(
        [(base_a_path, 1), (base_b_path, 1)], _sub, target_path, gdal.GDT_Byte,
        None)


def main():
    """Entry point."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    task_graph = taskgraph.TaskGraph(OUTPUT_DIR, 4, 15.0)

    # create forest masks
    restoration_mask_task = task_graph.add_task(
        func=create_mask,
        args=(LULC_RESTORATION_PATH, FOREST_LULC_CODES, FOREST_MASK_RESTORATION_PATH),
        target_path_list=[FOREST_MASK_RESTORATION_PATH],
        task_name=f'create_mask: {LULC_RESTORATION_PATH}')
    esa_mask_task = task_graph.add_task(
        func=create_mask,
        args=(LULC_ESA_PATH, FOREST_LULC_CODES, FOREST_MASK_ESA_PATH),
        target_path_list=[FOREST_MASK_ESA_PATH],
        task_name=f'create_mask: {LULC_ESA_PATH}')
    task_graph.add_task(
        func=sub_rasters,
        args=(FOREST_MASK_RESTORATION_PATH, FOREST_MASK_ESA_PATH, NEW_FOREST_MASK_ESA_TO_RESTORATION_PATH),
        target_path_list=[NEW_FOREST_MASK_ESA_TO_RESTORATION_PATH],
        dependent_task_list=[restoration_mask_task, esa_mask_task],
        task_name=f'and_rasters: {FOREST_MASK_RESTORATION_PATH}')

    # Build ESA carbon map since the change is just static and covert to co2
    task_graph.add_task(
        func=build_ipcc_carbon,
        args=(LULC_RESTORATION_PATH, CARBON_TABLE_PATH, CARBON_ZONES_PATH, FOREST_LULC_CODES, IPCC_CARBON_RESTORATION_PATH),
        target_path_list=[IPCC_CARBON_RESTORATION_PATH],
        task_name=f'build_ipcc_carbon: {LULC_RESTORATION_PATH}')
    task_graph.add_task(
        func=build_ipcc_carbon,
        args=(LULC_ESA_PATH, CARBON_TABLE_PATH, CARBON_ZONES_PATH, FOREST_LULC_CODES, IPCC_CARBON_ESA_PATH),
        target_path_list=[IPCC_CARBON_ESA_PATH],
        task_name=f'build_ipcc_carbon: {LULC_ESA_PATH}')

    # task_graph.add_task(
    #     func=regression_carbon_model,
    #     args=('./models/hansen_model_2022_07_14.dat', FOREST_MASK_RESTORATION_PATH),
    #     kwargs={'predictor_raster_dir': 'processed_rasters', 'model_result_path': REGRESSION_CARBON_RESTORATION_PATH},
    #     target_path_list=[REGRESSION_CARBON_RESTORATION_PATH],
    #     dependent_task_list=[restoration_mask_task],
    #     task_name=f'regression model {REGRESSION_CARBON_RESTORATION_PATH}')
    # task_graph.add_task(
    #     func=regression_carbon_model,
    #     args=('./models/hansen_model_2022_07_14.dat', FOREST_MASK_ESA_PATH),
    #     kwargs={'predictor_raster_dir': 'processed_rasters', 'model_result_path': REGRESSION_CARBON_ESA_PATH},
    #     target_path_list=[REGRESSION_CARBON_ESA_PATH],
    #     dependent_task_list=[esa_mask_task],
    #     task_name=f'regression model {REGRESSION_CARBON_ESA_PATH}')

    task_graph.join()
    #CALL python .\run_model.py .\models\hansen_model_2022_07_14.dat ./processed_rasters/fc_stack_hansen_forest_cover2016_compressed.tif --predictor_raster_dir ./processed_rasters


if __name__ == '__main__':
    main()
