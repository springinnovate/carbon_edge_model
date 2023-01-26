"""
This script will run a 0 step carbon model like this:
* take 90m forest mask
* coarsen to 900m
* uncoarsen to 90m
* run regression model on it
* sum the result
"""

import argparse
import os
import tempfile
import multiprocessing
import pickle
import shutil
from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal
import ecoshard
import pandas
import numpy

import gaussian_filter_rasters
from run_model import regression_carbon_model
from run_model import _pre_warp_rasters
from run_model import ECKERT_PIXEL_SIZE
from run_model import GLOBAL_BOUNDING_BOX_TUPLE
from run_model import WORLD_ECKERT_IV_WKT
from run_model import ZSTD_CREATION_TUPLE

import logging
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = (
    '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
    ' [%(funcName)s:%(lineno)d] %(message)s')
logging.basicConfig(
    level=LOG_LEVEL,
    filename='base_coarse.log',
    format=LOG_FORMAT)
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

gdal.SetCacheMax(2**24)
OUTPUT_DIR = f"./output_{GLOBAL_BOUNDING_BOX_TUPLE[0]}"

# Base data
CARBON_MODEL_PATH = './models/hansen_model_2022_07_14.dat'
INPUT_RASTERS = {
    'LULC_RESTORATION_PATH': "./ipcc_carbon_data/restoration_limited_md5_372bdfd9ffaf810b5f68ddeb4704f48f.tif",
    'LULC_ESA_PATH': "./ipcc_carbon_data/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif",
}
CARBON_ZONES_PATH = "./ipcc_carbon_data/carbon_zones_md5_aa16830f64d1ef66ebdf2552fb8a9c0d.gpkg"
CARBON_TABLE_PATH = "./ipcc_carbon_data/IPCC_carbon_table_md5_a91f7ade46871575861005764d85cfa7.csv"
FOREST_LULC_CODES = (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170)

COARSEN_FACTOR = 10
AREA_REPORT_STEPS = numpy.arange(1, 36, 1) * 100000000000

# Forest masks created by script
FOREST_MASK_RESTORATION_PATH = f'{OUTPUT_DIR}/forest_mask_restoration_limited.tif'
FOREST_MASK_ESA_PATH = f'{OUTPUT_DIR}/forest_mask_esa.tif'
COARSE_FOREST_MASK_ESA_PATH = f'{OUTPUT_DIR}/coarsened_forest_mask_esa.tif'
NEW_FOREST_MASK_ESA_TO_RESTORATION_PATH = f'{OUTPUT_DIR}/new_forest_mask_esa_to_restoration.tif'

# Convolved new forest mask
NEW_FOREST_MASK_COVERAGE_PATH = f'{OUTPUT_DIR}/new_forest_mask_coverage.tif'

# marginal value rasters
IPCC_MARGINAL_VALUE_PATH = f'{OUTPUT_DIR}/marginal_value_ipcc.tif'
REGRESSION_MARGINAL_VALUE_PATH = f'{OUTPUT_DIR}/marginal_value_regression.tif'

COARSE_IPCC_MARGINAL_VALUE_PATH = f'{OUTPUT_DIR}/coarsened_marginal_value_ipcc.tif'
COARSE_REGRESSION_MARGINAL_VALUE_PATH = f'{OUTPUT_DIR}/coarsened_marginal_value_regression.tif'

# IPCC based carbon maps
IPCC_CARBON_RESTORATION_PATH = f'{OUTPUT_DIR}/ipcc_carbon_restoration_limited.tif'
IPCC_CARBON_ESA_PATH = f'{OUTPUT_DIR}/ipcc_carbon_esa.tif'

IPCC_OPTIMIZATION_OUTPUT_DIR = f'{OUTPUT_DIR}/ipcc_optimization'
IPCC_AREA_PATH = f'{OUTPUT_DIR}/ipcc_area.tif'

MASKED_IPCC_CARBON_RESTORATION_PATH = f'{OUTPUT_DIR}/masked_ipcc_carbon_restoration_limited.tif'
MASKED_IPCC_CARBON_ESA_PATH = f'{OUTPUT_DIR}/masked_ipcc_carbon_esa.tif'

# Regression based carbon maps:
REGRESSION_CARBON_RESTORATION_PATH = f'{OUTPUT_DIR}/regression_carbon_restoration.tif'
REGRESSION_CARBON_ESA_PATH = f'{OUTPUT_DIR}/regression_carbon_esa.tif'

# Intermediate regression marginal value:
REGRESSION_PER_PIXEL_DISTANCE_CONTRIBUTION_PATH = f'{OUTPUT_DIR}/regression_per_pixel_carbon_distance_weight.tif'

REGRESSION_OPTIMIZATION_OUTPUT_DIR = f'{OUTPUT_DIR}/regression_optimization'
REGRESSION_AREA_PATH = f'{OUTPUT_DIR}/regression_area.tif'

PREDICTOR_RASTER_DIR = './processed_rasters'
PRE_WARP_DIR = os.path.join(PREDICTOR_RASTER_DIR, f'pre_warped_{GLOBAL_BOUNDING_BOX_TUPLE[0]}')


def sum_raster(raster_path):
    """Return the sum of non-nodata value pixels in ``raster_path``."""
    running_sum = 0.0
    nodata = geoprocessing.get_raster_info(raster_path)['nodata'][0]
    for _, block_array in geoprocessing.iterblocks((raster_path, 1)):
        if nodata is not None:
            valid_array = block_array != nodata
        else:
            valid_array = slice(-1)
        running_sum += numpy.sum(block_array[valid_array])
    return running_sum


def main():
    coarse_raster_path = "./output_global/coarsened_forest_mask_esa.tif"
    fine_raster_path = "./output_global/90m_forest_mask_esa.tif"
    new_carbon_path = "./output_global/900m_to_90m_regression_esa.tif"
    geoprocessing.warp_raster(
        coarse_raster_path, ECKERT_PIXEL_SIZE,
        fine_raster_path, 'near',
        target_bb=GLOBAL_BOUNDING_BOX_TUPLE[1],
        target_projection_wkt=WORLD_ECKERT_IV_WKT,
        n_threads=multiprocessing.cpu_count(),
        working_dir=PRE_WARP_DIR,
        raster_driver_creation_tuple=ZSTD_CREATION_TUPLE)

    task_graph = taskgraph.TaskGraph('.', multiprocessing.cpu_count(), 15.0)
    regression_carbon_model(
        CARBON_MODEL_PATH, GLOBAL_BOUNDING_BOX_TUPLE,
        fine_raster_path, PREDICTOR_RASTER_DIR,
        pre_warp_dir=PRE_WARP_DIR,
        target_result_path=new_carbon_path,
        external_task_graph=task_graph,
        clean_workspace=False)
    task_graph.join()

    sum_task = task_graph.add_task(
        func=sum_raster,
        args=(new_carbon_path),
        store_result=True)

    LOGGER.info(f'sum of {new_carbon_path}: {sum_task.get()}')


if __name__ == '__main__':
    main()
