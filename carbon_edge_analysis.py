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
logging.getLogger('matplotlib').setLevel(logging.ERROR)
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = (
    '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
    ' [%(funcName)s:%(lineno)d] %(message)s')
logging.basicConfig(
    level=LOG_LEVEL,
    filename='carbon_edge_anal.log',
    format=LOG_FORMAT)

# set up logging to console
console = logging.StreamHandler()
console.setLevel(LOG_LEVEL)
# set a format which is simpler for console use
formatter = logging.Formatter(LOG_FORMAT)
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

logging.getLogger('taskgraph').setLevel(logging.DEBUG)
LOGGER = logging.getLogger(__name__)

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


def build_ipcc_carbon(lulc_path, lulc_table_path, zone_path, lulc_codes, target_carbon_path):
    """Calculate IPCC carbon.

    Args:
        lulc_path (str): path to raster with LULC codes found in lulc_table_path and lulc_codes.
        lulc_table_path (str): maps carbon zone (rows) to lulc codes (columns) to map carbon total.
        zone_path (str): path to vector with carbon zones in 'CODE' field.
        lulc_codes (tuple): only evaluate codes in this tuple
        target_carbon_path (str): created raster that contains carbon values

    Return:
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
    shutil.rmtree(working_dir, ignore_errors=True)


def create_mask(base_path, code_list, target_path):
    """Mask out base with values in code list, 0 otherwise.

    Args:
        base_path (str): path to an integer raster
        code_list (list): list of integer values to set to 1 in base
        target_path (str): path to created raster

    Return:
        None
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

    Return:
        None
    """
    def _sub(base_a_array, base_b_array):
        return base_a_array > base_b_array

    geoprocessing.raster_calculator(
        [(base_a_path, 1), (base_b_path, 1)], _sub, target_path, gdal.GDT_Byte,
        None)


def mask_raster(base_path, mask_path, target_path):
    """Only pass through base if mask is 1."""
    def _mask(base_array, mask_array):
        result = numpy.zeros(base_array.shape, dtype=base_array.type)
        mask = mask_array == 1
        result[mask] = base_array[mask]
        return result


def sub_and_mask(base_a_path, base_b_path, mask_path, target_path):
    """Subtract a and b and only keep mask.

    Args:
        base_a_path (str): path to raster
        base_b_path (str): path to raster
        mask_path (str): path to 0/1 raster, 1 indicates area to keep
        target_path (str): result.

    Return:
        None
    """
    def _mask_and_sub(base_a_array, base_b_array, mask_array):
        result = numpy.zeros(base_a_array.shape, dtype=base_a_array.dtype)
        valid_mask = mask_array == 1
        result[valid_mask] = base_a_array[valid_mask] - base_b_array[valid_mask]
        return result
    raster_info = geoprocessing.get_raster_info(base_a_path)
    geoprocessing.raster_calculator(
        [(base_a_path, 1), (base_b_path, 1), (mask_path, 1)], _mask_and_sub,
        target_path, raster_info['datatype'], 0)


def sub_and_divide(base_a_path, base_b_path, mask_path, target_path):
    """Subtract a from b then divide by mask. If mask is 0 then skip.

    Args:
        base_a_path (str): path to raster
        base_b_path (str): path to raster
        mask_path (str): path to 0/non-0 raster
        target_path (str): result

    Return:
        None
    """
    def _sub_and_divide(base_a_array, base_b_array, mask_array):
        result = numpy.zeros(base_a_array.shape, dtype=base_a_array.dtype)
        valid_mask = mask_array > 0
        result[valid_mask] = (
            base_a_array[valid_mask] - base_b_array[valid_mask]) / (
            mask_array[valid_mask])
        return result
    raster_info = geoprocessing.get_raster_info(base_a_path)
    geoprocessing.raster_calculator(
        [(base_a_path, 1), (base_b_path, 1), (mask_path, 1)],
        _sub_and_divide, target_path, raster_info['datatype'], 0)


def regression_marginal_value(base_path, gf_size, mask_path, target_path):
    """Calculate marginal value regression by convolving base and masking.

    Args:
        base_path (str): path to value raster
        gf_size (float): gaussian kernel size to filter distance
        mask_path (str): path to 0/1 mask raster
        target_path (str): path to marginal value target which is filtered
            base by gf_size then masked by the mask.

    Return:
        None
    """
    basename = os.path.basename(os.path.splitext(base_path)[0])
    base_filtered_path = os.path.join(
        os.path.dirname(target_path),
        f'{basename}_filtered.tif')
    os.makedirs(os.path.dirname(base_filtered_path), exist_ok=True)
    gaussian_filter_rasters.filter_raster(
        (base_path, 1), gf_size, base_filtered_path)

    base_nodata = geoprocessing.get_raster_info(
        base_filtered_path)['nodata'][0]
    def _mask_op(base_array, mask_array):
        return numpy.where(
            (mask_array == 1) & (base_array != base_nodata), base_array, 0.0)
    raster_info = geoprocessing.get_raster_info(base_path)
    geoprocessing.raster_calculator(
        [(base_filtered_path, 1), (mask_path, 1)],
        _mask_op, target_path, raster_info['datatype'], 0)


def make_area_raster(base_path, target_area_path):
    """Create raster full of pixel area based on base."""
    base_info = geoprocessing.get_raster_info(base_path)
    base_area = abs(numpy.prod(base_info['pixel_size']))
    gtiff_creation_tuple_options = ('GTIFF', (
        'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW', 'SPARSE_OK=TRUE',
        'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'NUM_THREADS=ALL_CPUS'))
    # creates a sparse empty raster with pixel sizes of nodata that are area
    geoprocessing.new_raster_from_base(
        base_path, target_area_path, gdal.GDT_Float32, [base_area],
        raster_driver_creation_tuple=gtiff_creation_tuple_options)


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

def sum_by_mask(raster_path, mask_path):
    """Return tuple of sum of non-nodata values in raster_path in and out of mask.

    Returns:
        (in sum, out sum)
    """
    in_running_sum = 0.0
    out_running_sum = 0.0
    nodata = geoprocessing.get_raster_info(raster_path)['nodata'][0]
    for ((_, block_array), (_, mask_array)) in \
            zip(geoprocessing.iterblocks((raster_path, 1)),
                geoprocessing.iterblocks((mask_path, 1))):
        if nodata is not None:
            valid_array = block_array != nodata
        else:
            valid_array = numpy.ones(block_array.shape, dtype=bool)
        in_running_sum += numpy.sum(block_array[valid_array & (mask_array == 1)])
        out_running_sum += numpy.sum(block_array[valid_array & (mask_array != 1)])
    return (in_running_sum, out_running_sum)


def add_masks(mask_a_path, mask_b_path, target_path):
    """Combine two masks as a logical OR.

    Args:
        mask_a_path, mask_b_path (str): path to mask rasters with 1 indicating mask
        target_path (str): path to combined mask raster.

    Return:
        None
    """
    raster_info = geoprocessing.get_raster_info(mask_a_path)
    nodata = raster_info['nodata'][0]

    def _add_masks(mask_a_array, mask_b_array):
        """Combine a and b."""
        valid_mask = (mask_a_array == 1) | (mask_b_array == 1)
        return valid_mask

    geoprocessing.raster_calculator(
        [(mask_a_path, 1), (mask_b_path, 1)],
        _add_masks, target_path, raster_info['datatype'], nodata)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='carbon edge analysis for paper')
    parser.add_argument(
        'n_workers', type=int, help='number of parallel task graph workers')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    task_graph = taskgraph.TaskGraph(OUTPUT_DIR, args.n_workers, 15.0)

    # project everything in same projection as carbon model
    aligned_dir = './aligned_carbon_edge'
    os.makedirs(aligned_dir, exist_ok=True)
    LOGGER.info(f'pre-warp the input rasters {INPUT_RASTERS}')
    for raster_id in INPUT_RASTERS:
        raster_path = INPUT_RASTERS[raster_id]
        aligned_path = os.path.join(aligned_dir, os.path.basename(raster_path))
        task_graph.add_task(
            func=geoprocessing.warp_raster,
            args=(raster_path, ECKERT_PIXEL_SIZE,
                  aligned_path, 'near'),
            kwargs={
                'target_bb': GLOBAL_BOUNDING_BOX_TUPLE[1],
                'target_projection_wkt': WORLD_ECKERT_IV_WKT,
                'working_dir': aligned_dir,
                'n_threads': multiprocessing.cpu_count()/len(INPUT_RASTERS)*2,
                'raster_driver_creation_tuple': ZSTD_CREATION_TUPLE},
            target_path_list=[aligned_path],
            task_name=f'warp {aligned_path}')
        INPUT_RASTERS[raster_id] = aligned_path

    _pre_warp_rasters(
        task_graph, GLOBAL_BOUNDING_BOX_TUPLE[1], CARBON_MODEL_PATH,
        PREDICTOR_RASTER_DIR, PRE_WARP_DIR)
    task_graph.join()
    LOGGER.debug('all done pre-warping')

    LULC_RESTORATION_PATH = INPUT_RASTERS['LULC_RESTORATION_PATH']
    LULC_ESA_PATH = INPUT_RASTERS['LULC_ESA_PATH']

    with open(CARBON_MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file).copy()

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
    sub_mask_task = task_graph.add_task(
        func=sub_rasters,
        args=(FOREST_MASK_RESTORATION_PATH, FOREST_MASK_ESA_PATH, NEW_FOREST_MASK_ESA_TO_RESTORATION_PATH),
        target_path_list=[NEW_FOREST_MASK_ESA_TO_RESTORATION_PATH],
        dependent_task_list=[restoration_mask_task, esa_mask_task],
        task_name=f'and_rasters: {FOREST_MASK_RESTORATION_PATH}')

    # convolve the mask same as carbon kernel NEW_FOREST_MASK_COVERAGE_PATH
    new_forest_mask_coverage_task = task_graph.add_task(
        func=gaussian_filter_rasters.filter_raster,
        args=((NEW_FOREST_MASK_ESA_TO_RESTORATION_PATH, 1), model['gf_size'],
              NEW_FOREST_MASK_COVERAGE_PATH),
        dependent_task_list=[sub_mask_task],
        target_path_list=[NEW_FOREST_MASK_COVERAGE_PATH],
        task_name=f'gaussian filter {NEW_FOREST_MASK_COVERAGE_PATH}')

    # Build ESA carbon map since the change is just static and covert to co2
    ipcc_restoration_carbon_task = task_graph.add_task(
        func=build_ipcc_carbon,
        args=(LULC_RESTORATION_PATH, CARBON_TABLE_PATH, CARBON_ZONES_PATH, FOREST_LULC_CODES, IPCC_CARBON_RESTORATION_PATH),
        target_path_list=[IPCC_CARBON_RESTORATION_PATH],
        task_name=f'build_ipcc_carbon: {LULC_RESTORATION_PATH}')
    ipcc_esa_carbon_task = task_graph.add_task(
        func=build_ipcc_carbon,
        args=(LULC_ESA_PATH, CARBON_TABLE_PATH, CARBON_ZONES_PATH, FOREST_LULC_CODES, IPCC_CARBON_ESA_PATH),
        target_path_list=[IPCC_CARBON_ESA_PATH],
        task_name=f'build_ipcc_carbon: {LULC_ESA_PATH}')

    # IPCC_CARBON_RESTORATION_PATH-IPCC_CARBON_ESA_PATH and masked to new forest
    ipcc_marginal_value_task = task_graph.add_task(
        func=sub_and_mask,
        args=(
            IPCC_CARBON_RESTORATION_PATH, IPCC_CARBON_ESA_PATH,
            NEW_FOREST_MASK_ESA_TO_RESTORATION_PATH, IPCC_MARGINAL_VALUE_PATH),
        target_path_list=[IPCC_MARGINAL_VALUE_PATH],
        dependent_task_list=[
            ipcc_restoration_carbon_task, ipcc_esa_carbon_task],
        task_name=f'create IPCC marginal value {IPCC_MARGINAL_VALUE_PATH}')

    task_graph.join()

    regression_carbon_model(
        CARBON_MODEL_PATH, GLOBAL_BOUNDING_BOX_TUPLE,
        FOREST_MASK_RESTORATION_PATH, PREDICTOR_RASTER_DIR,
        pre_warp_dir=PRE_WARP_DIR,
        target_result_path=REGRESSION_CARBON_RESTORATION_PATH,
        external_task_graph=task_graph,
        clean_workspace=False)

    regression_carbon_model(
        CARBON_MODEL_PATH, GLOBAL_BOUNDING_BOX_TUPLE,
        FOREST_MASK_ESA_PATH, PREDICTOR_RASTER_DIR,
        pre_warp_dir=PRE_WARP_DIR,
        target_result_path=REGRESSION_CARBON_ESA_PATH,
        external_task_graph=task_graph,
        clean_workspace=False)
    task_graph.join()

    # Calculate per-pixel weighted contribution REGRESSION_CARBON_RESTORATION_PATH-REGRESSION_CARBON_ESA_PATH/NEW_FOREST_MASK_COVERAGE_PATH
    weighted_regression_task = task_graph.add_task(
        func=sub_and_divide,
        args=(REGRESSION_CARBON_RESTORATION_PATH, REGRESSION_CARBON_ESA_PATH,
              NEW_FOREST_MASK_COVERAGE_PATH,
              REGRESSION_PER_PIXEL_DISTANCE_CONTRIBUTION_PATH),
        dependent_task_list=[
            new_forest_mask_coverage_task, ],
        target_path_list=[REGRESSION_PER_PIXEL_DISTANCE_CONTRIBUTION_PATH],
        task_name=f'per pixel weighted coverage {REGRESSION_PER_PIXEL_DISTANCE_CONTRIBUTION_PATH}')

    # convolve above and mask to new forest
    regression_marginal_value_task = task_graph.add_task(
        func=regression_marginal_value,
        args=(REGRESSION_PER_PIXEL_DISTANCE_CONTRIBUTION_PATH,
              model['gf_size'],
              NEW_FOREST_MASK_ESA_TO_RESTORATION_PATH,
              REGRESSION_MARGINAL_VALUE_PATH),
        dependent_task_list=[weighted_regression_task, sub_mask_task],
        target_path_list=[REGRESSION_MARGINAL_VALUE_PATH],
        task_name=f'regression marg value {REGRESSION_MARGINAL_VALUE_PATH}')

    regression_marginal_value_task.join()

    coarsen_forest_esa_mask_task = task_graph.add_task(
        func=ecoshard.convolve_layer,
        args=(FOREST_MASK_ESA_PATH, COARSEN_FACTOR, 'mode', COARSE_FOREST_MASK_ESA_PATH),
        dependent_task_list=[esa_mask_task],
        target_path_list=[COARSE_FOREST_MASK_ESA_PATH],
        task_name=f'coarsen {COARSE_FOREST_MASK_ESA_PATH}')

    # coarsen IPCC_MARGINAL_VALUE
    coarsen_task_map = {}
    for base_raster_path, target_coarse_path, base_task, task_id in [
            (IPCC_MARGINAL_VALUE_PATH, COARSE_IPCC_MARGINAL_VALUE_PATH, ipcc_marginal_value_task, 'ipcc'),
            (REGRESSION_MARGINAL_VALUE_PATH, COARSE_REGRESSION_MARGINAL_VALUE_PATH, regression_marginal_value_task, 'regression')]:
        coarsen_task_map[task_id] = task_graph.add_task(
            func=ecoshard.convolve_layer,
            args=(base_raster_path, COARSEN_FACTOR, 'sum', target_coarse_path),
            dependent_task_list=[base_task],
            target_path_list=[target_coarse_path],
            task_name=f'coarsen {target_coarse_path}')

    # run optimization on above on 350Mha
    for output_dir, marginal_value_path, area_path, out_prefix, in [
            (IPCC_OPTIMIZATION_OUTPUT_DIR, COARSE_IPCC_MARGINAL_VALUE_PATH, IPCC_AREA_PATH, 'ipcc'),
            (REGRESSION_OPTIMIZATION_OUTPUT_DIR, COARSE_REGRESSION_MARGINAL_VALUE_PATH, REGRESSION_AREA_PATH, 'regression')]:
        os.makedirs(output_dir, exist_ok=True)
        area_task = task_graph.add_task(
            func=make_area_raster,
            args=(marginal_value_path, area_path),
            target_path_list=[area_path],
            dependent_task_list=[coarsen_task_map[out_prefix]],
            task_name=f'make area raster {area_path}')

        # run optimization on above
        greedy_pixel_pick_task = task_graph.add_task(
            func=geoprocessing.greedy_pixel_pick_by_area,
            args=((marginal_value_path, 1), (area_path, 1),
                  AREA_REPORT_STEPS, output_dir),
            kwargs={'output_prefix': out_prefix, 'ffi_buffer_size': 2**20},
            dependent_task_list=[area_task],
            store_result=True,
            task_name=f'{out_prefix} optimization')

        if out_prefix == 'regression':
            raster_sum_list = []
            # the [1] is so we get the mask path list, [0] is the table path
            esa_base_sum_task = task_graph.add_task(
                func=sum_raster,
                args=(REGRESSION_CARBON_ESA_PATH,),
                store_result=True,
                task_name=f'sum regression carbon {REGRESSION_CARBON_ESA_PATH}')
            for new_forest_mask_path in greedy_pixel_pick_task.get()[1]:
                transient_run = False
                # combine result mask path with FOREST_MASK_ESA_PATH
                coarse_carbon_opt_forest_step_path = (
                    '%s_coarse_full_forest_mask%s' % os.path.splitext(new_forest_mask_path))
                full_forest_task = task_graph.add_task(
                    func=add_masks,
                    args=(new_forest_mask_path, COARSE_FOREST_MASK_ESA_PATH, coarse_carbon_opt_forest_step_path),
                    dependent_task_list=[coarsen_forest_esa_mask_task],
                    target_path_list=[coarse_carbon_opt_forest_step_path],
                    task_name=f'combine optimization mask with base ESA forest mask {coarse_carbon_opt_forest_step_path}')

                carbon_opt_forest_step_path = (
                    '%s_full_forest_mask%s' % os.path.splitext(new_forest_mask_path))
                uncoarsen_forest_mask_task = task_graph.add_task(
                    func=geoprocessing.warp_raster,
                    args=(coarse_carbon_opt_forest_step_path, ECKERT_PIXEL_SIZE,
                          carbon_opt_forest_step_path, 'near'),
                    kwargs={
                        'target_bb': GLOBAL_BOUNDING_BOX_TUPLE[1],
                        'target_projection_wkt': WORLD_ECKERT_IV_WKT,
                        'n_threads': multiprocessing.cpu_count(),
                        'working_dir': PRE_WARP_DIR,
                        'raster_driver_creation_tuple': ZSTD_CREATION_TUPLE
                    },
                    dependent_task_list=[full_forest_task],
                    target_path_list=[carbon_opt_forest_step_path],
                    task_name=f'uncoarsen {carbon_opt_forest_step_path}')

                uncoarsened_new_forest_mask_path = (
                    '%s_new_forest_mask%s' % os.path.splitext(new_forest_mask_path))
                uncoarsen_new_forest_mask_task = task_graph.add_task(
                    func=geoprocessing.warp_raster,
                    args=(new_forest_mask_path, ECKERT_PIXEL_SIZE,
                          uncoarsened_new_forest_mask_path, 'near'),
                    kwargs={
                        'target_bb': GLOBAL_BOUNDING_BOX_TUPLE[1],
                        'target_projection_wkt': WORLD_ECKERT_IV_WKT,
                        'n_threads': multiprocessing.cpu_count(),
                        'working_dir': PRE_WARP_DIR,
                        'raster_driver_creation_tuple': ZSTD_CREATION_TUPLE
                    },
                    dependent_task_list=[greedy_pixel_pick_task],
                    target_path_list=[uncoarsened_new_forest_mask_path],
                    task_name=f'uncoarsen {uncoarsened_new_forest_mask_path}')

                LOGGER.debug(f'{coarse_carbon_opt_forest_step_path} to this {carbon_opt_forest_step_path}')

                modeled_carbon_opt_step_path = (
                    '%s_regression%s' % os.path.splitext(new_forest_mask_path))
                task_graph.join()
                regression_carbon_model(
                    CARBON_MODEL_PATH, GLOBAL_BOUNDING_BOX_TUPLE,
                    carbon_opt_forest_step_path, PREDICTOR_RASTER_DIR,
                    pre_warp_dir=PRE_WARP_DIR,
                    target_result_path=modeled_carbon_opt_step_path,
                    external_task_graph=task_graph,
                    clean_workspace=False)
                LOGGER.debug(f'regression result should be in {modeled_carbon_opt_step_path}')
                # break out result into old and new forest
                sum_in_out_forest_carbon_density_by_mask_task = task_graph.add_task(
                    func=sum_by_mask,
                    args=(modeled_carbon_opt_step_path, uncoarsened_new_forest_mask_path),
                    dependent_task_list=[uncoarsen_forest_mask_task],
                    store_result=True,
                    transient_run=transient_run,
                    task_name=f'separate out old and new carbon for {modeled_carbon_opt_step_path}')

                # count number of total forest pixels
                count_forest_pixel_task = task_graph.add_task(
                    func=sum_raster,
                    args=(carbon_opt_forest_step_path,),
                    dependent_task_list=[uncoarsen_forest_mask_task],
                    transient_run=transient_run,
                    task_name=f'sum raster of {carbon_opt_forest_step_path}',
                    store_result=True)

                # count number of new forest pixels
                count_new_forest_pixel_task = task_graph.add_task(
                    func=sum_raster,
                    args=(uncoarsened_new_forest_mask_path,),
                    dependent_task_list=[uncoarsen_new_forest_mask_task],
                    transient_run=transient_run,
                    task_name=f'sum raster of {uncoarsen_new_forest_mask_task}',
                    store_result=True)

                raster_sum_list.append(
                    (os.path.basename(modeled_carbon_opt_step_path),
                     count_forest_pixel_task,
                     count_new_forest_pixel_task,
                     sum_in_out_forest_carbon_density_by_mask_task))
            task_graph.join()
            raster_info = geoprocessing.get_raster_info(carbon_opt_forest_step_path)
            LOGGER.debug('writing regression_optimization_carbon')
            with open('regression_optimization_carbon.csv', 'w') as opt_table:
                opt_table.write(
                    'file,'
                    'number of forest pixels,'
                    'number of old forest pixels,'
                    'number of new forest pixels,'
                    'sum of carbon density for all forest pixels,'
                    'sum of carbon density for old forest pixels,'
                    'sum of carbon density for new forest pixels,'
                    'carbon density per pixel for all forest,'
                    'carbon density per pixel in old forest,'
                    'carbon density per pixel in new forest,'
                    'carbon density per pixel in esa scenario,'
                    'area of pixel in m^2\n')
                for path, count_forest_pixel_task, count_new_forest_pixel_task, sum_in_out_forest_carbon_density_by_mask_task in raster_sum_list:
                    new_carbon_density_sum = sum_in_out_forest_carbon_density_by_mask_task.get()[0]
                    old_carbon_density_sum = sum_in_out_forest_carbon_density_by_mask_task.get()[1]
                    all_forest_pixel_count = count_forest_pixel_task.get()
                    new_forest_pixel_count = count_new_forest_pixel_task.get()
                    old_forest_pixel_count = all_forest_pixel_count - new_forest_pixel_count
                    opt_table.write(
                        f'{path},'
                        f'{all_forest_pixel_count},'
                        f'{old_forest_pixel_count},'
                        f'{new_forest_pixel_count},'
                        f'{old_carbon_density_sum+new_carbon_density_sum},'
                        f'{old_carbon_density_sum},'
                        f'{new_carbon_density_sum},'
                        f'{(new_carbon_density_sum+old_carbon_density_sum)/(all_forest_pixel_count)},'
                        f'{old_carbon_density_sum/old_forest_pixel_count},'
                        f'{new_carbon_density_sum/new_forest_pixel_count},'
                        f'{esa_base_sum_task.get()},'
                        f'{abs(numpy.prod(raster_info["pixel_size"]))}\n')

    task_graph.join()
    LOGGER.debug('all done')


if __name__ == '__main__':
    main()
