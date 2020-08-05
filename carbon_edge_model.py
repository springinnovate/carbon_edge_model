"""Carbon edge regression model."""
import argparse
import glob
import logging
import multiprocessing
import os
import subprocess
import sys

from osgeo import gdal
from osgeo import osr
import ecoshard
import pygeoprocessing
import numpy
import scipy.ndimage
import taskgraph

import mult_by_columns_library
import model_files

gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)

LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)

CONVOLUTION_PIXEL_DIST_LIST = [1, 2, 3, 5, 10, 20, 30, 50, 100]

# Landcover classification codes
# 1: cropland
# 2: urban
# 3: forest
# 4: other
MASK_TYPES = [
    ('is_cropland_10sec', (1,)),
    ('is_urban_10sec', (2,)),
    ('not_forest_10sec', (1, 2, 4)),
    ('forest_10sec', (3, ))]

MASK_NODATA = 2
MULT_BY_COLUMNS_NODATA = -1

ZERO_NODATA_SYMBOLS = {
    'bdod_10sec',
    'BDTICM_10sec',
    'BLDFIE_10sec',
    'CECSOL_10sec',
    'cfvo_10sec',
    'clay_10sec',
    'is_cropland_10sec',
    'is_urban_10sec',
    'ndvcec015_10sec',
    'nitrogen_10sec',
    'not_forest_10sec',
    'ocd_10sec',
    'OCDENS_10sec',
    'ocs_10sec',
    'OCSTHA_10sec',
    'phh2o_10sec',
    'population_2015_30sec',
    'population_2015_5min',
    'sand_10sec',
    'silt_10sec',
    'soc_10sec',
    }


def sub_pos_op(array_a, array_b):
    """Assume nodata value is negative and the same for a and b."""
    result = array_a.copy()
    mask = array_b > 0
    result[mask] -= array_b[mask]
    return result


def where_op(condition_array, if_true_array, else_array):
    """Select from `if true array` if condition true, `else array`."""
    result = numpy.copy(else_array)
    mask = condition_array == 1
    result[mask] = if_true_array[mask]
    return result


def raster_where(
        condition_raster_path, if_true_raster_path, else_raster_path,
        target_raster_path):
    """A raster version of the numpy.where function."""
    pygeoprocessing.raster_calculator(
        [(condition_raster_path, 1), (if_true_raster_path, 1),
         (else_raster_path, 1)], where_op, target_raster_path,
        gdal.GDT_Float32,
        pygeoprocessing.get_raster_info(if_true_raster_path)['nodata'][0])


def make_kernel_raster(pixel_radius, target_path):
    """Create kernel with given radius to `target_path`."""
    truncate = 4
    size = int(pixel_radius * 2 * truncate + 1)
    step_fn = numpy.zeros((size, size))
    step_fn[size//2, size//2] = 1
    kernel_array = scipy.ndimage.filters.gaussian_filter(
        step_fn, pixel_radius, order=0, mode='reflect', cval=0.0,
        truncate=truncate)
    pygeoprocessing.numpy_array_to_raster(
        kernel_array, -1, (1., -1.), (0.,  0.), None,
        target_path)


def _mask_vals_op(array, nodata, valid_1d_array, target_nodata):
    """Set values 1d array/array to nodata."""
    result = numpy.zeros(array.shape, dtype=numpy.uint8)
    if nodata is not None:
        nodata_mask = numpy.isclose(array, nodata)
    else:
        nodata_mask = numpy.zeros(array.shape, dtype=numpy.bool)
    valid_mask = numpy.in1d(array, valid_1d_array).reshape(result.shape)
    result[valid_mask & ~nodata_mask] = 1
    result[nodata_mask] = target_nodata
    return result


def mult_rasters_op(array_a, array_b, nodata_a, nodata_b, target_nodata):
    """Mult a*b and account for nodata."""
    result = numpy.empty(array_a.shape)
    result[:] = target_nodata
    valid_mask = (
        ~numpy.isclose(array_a, nodata_a) & ~numpy.isclose(array_b, nodata_b))
    result[valid_mask] = array_a[valid_mask] * array_b[valid_mask]
    return result


def mult_by_const_op(array, const, nodata, target_nodata):
    """Mult array by const."""
    result = numpy.empty(array.shape, dtype=numpy.float32)
    result[:] = target_nodata
    valid_mask = ~numpy.isclose(array, nodata)
    result[valid_mask] = array[valid_mask].astype(numpy.float32) * const
    return result


def mask_ranges(
        base_raster_path, mask_value_list, target_raster_path):
    """Mask all values in the given inclusive range to 1, the rest to 0.

    Args:
        base_raster_path (str): path to an integer raster
        mask_value_list (list): lits of integer codes to set output to 1 or 0
            if it is ccontained in the list.
        target_raster_path (str): path to output mask, will contain 1 or 0
            whether the base had a pixel value contained in the
            `mask_value_list` while accounting for `inverse`.

    Returns:
        None.

    """
    base_nodata = pygeoprocessing.get_raster_info(
        base_raster_path)['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1), (base_nodata, 'raw'),
         (mask_value_list, 'raw'),
         (MASK_NODATA, 'raw')], _mask_vals_op,
        target_raster_path, gdal.GDT_Byte, MASK_NODATA)


def fetch_data(data_dir, task_graph):
    """Download all the global data needed to run this analysis.

    Args:
        bounding_box (list): minx, miny, maxx, maxy list to clip to
        data_dir (str): path to directory to copy clipped rasters
            to
        task_graph (TaskGraph): taskgraph object to schedule work.

    Returns:
        None.

    """
    files_to_download = model_files.CARBON_EDGE_REGRESSION_MODEL_URL_LIST + [
        model_files.BACCINI_10s_2014_BIOMASS_URL,
        model_files.FOREST_REGRESSION_LASSO_TABLE_URL]

    LOGGER.debug(f'here are the files to download: {files_to_download}')

    try:
        os.makedirs(data_dir)
    except OSError:
        pass

    for file_url in files_to_download:
        target_file_path = os.path.join(
            data_dir, os.path.basename(file_url))
        _ = task_graph.add_task(
            func=ecoshard.download_url,
            args=(file_url, target_file_path),
            kwargs={'skip_if_target_exists': True},
            target_path_list=[target_file_path],
            task_name=f'download {file_url} to {data_dir}')

    task_graph.join()
    global BACCINI_10s_2014_BIOMASS_RASTER_PATH
    BACCINI_10s_2014_BIOMASS_RASTER_PATH = os.path.join(
        data_dir, os.path.basename(model_files.BACCINI_10s_2014_BIOMASS_URL))
    global FOREST_REGRESSION_LASSO_TABLE_PATH
    FOREST_REGRESSION_LASSO_TABLE_PATH = os.path.join(
        data_dir, os.path.basename(
            model_files.FOREST_REGRESSION_LASSO_TABLE_URL))
    task_graph.join()


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Carbon edge model')
    parser.add_argument(
        'landtype_mask_raster_path', help=(
            'Path to landtype raster where codes correspond to:\n'
            '\t1: cropland\n\t2: urban\n\t3: forest\n\t4: other'))
    parser.add_argument(
        '--workspace_dir', default='carbon_model_workspace', help=(
            'Path to workspace dir, the carbon stock file will be named '
            '"c_stocks_[landtype_mask_raster_path]. Default is '
            '`carbon_model_workspace`"'))
    parser.add_argument(
        '--co2', action='store_true', help=(
            'If specified, output is in units of C02 rather than C and the '
            'output file will be prefixed by c02 '
            'co2_stocks_[landtype_mask_raster_path]'))

    args = parser.parse_args()

    if args.co2:
        c_prefix = 'co2'
    else:
        c_prefix = 'c'

    workspace_dir = args.workspace_dir
    churn_dir = os.path.join(workspace_dir, 'churn')
    data_dir = os.path.join(workspace_dir, 'data')

    for dir_path in [workspace_dir, churn_dir, data_dir]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    # 1) Download data
    task_graph = taskgraph.TaskGraph(
        churn_dir, multiprocessing.cpu_count(), 5.0)
    LOGGER.info("Step 0: Download data")
    fetch_data(data_dir, task_graph)

    # 2) TODO: align all downloaded data
    base_raster_data_path_list = glob.glob(os.path.join(data_dir, '*.tif'))
    landtype_basename = os.path.basename(
        os.path.splitext(args.landtype_mask_raster_path)[0])
    aligned_data_dir = os.path.join(
        args.workspace_dir, f'{landtype_basename}_aligned_data')
    try:
        os.makedirs(aligned_data_dir)
    except OSError:
        pass
    aligned_raster_path_list = [
        os.path.join(aligned_data_dir, os.path.basename(path))
        for path in base_raster_data_path_list]
    base_raster_info = pygeoprocessing.get_raster_info(
        args.landtype_mask_raster_path)
    for base_raster_path, aligned_raster_path in zip(
            base_raster_data_path_list, aligned_raster_path_list):
        _ = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(
                base_raster_path, base_raster_info['pixel_size'],
                aligned_raster_path, 'near'),
            kwargs={
                'target_bb': base_raster_info['bounding_box'],
                'target_projection_wkt': base_raster_info['projection_wkt'],
                'working_dir': aligned_data_dir,
                'n_threads': 2,
                },
            target_path_list=[aligned_raster_path],
            task_name=f'align {base_raster_path} data')
    LOGGER.info('wait for data to align')
    task_graph.join()

    # FOREST REGRESSION

    # 1) Make convolutions with custom kernel of 1, 2, 3, 5, 10, 20, 30, 50,
    #    and 100 pixels for not_forest (see forest lulc codes), is_cropland
    #    (classes 10-40), and is_urban (class 190) for LULC maps

    LOGGER.info("Forest Regression step 1")
    mask_path_task_map = {}
    for mask_type, lulc_codes in MASK_TYPES:
        lulc_mask_raster_path = os.path.join(
            aligned_data_dir, f'mask_of_{mask_type}.tif')
        mask_task = task_graph.add_task(
            func=mask_ranges,
            args=(
                args.landtype_mask_raster_path, lulc_codes,
                lulc_mask_raster_path),
            target_path_list=[lulc_mask_raster_path],
            task_name=f'make {mask_type}')
        mask_path_task_map[mask_type] = (lulc_mask_raster_path, mask_task)
        LOGGER.debug(
            f'this is the scenario lulc mask target: '
            f'{lulc_mask_raster_path}')

    kernel_raster_path_map = {}
    for pixel_radius in reversed(sorted(CONVOLUTION_PIXEL_DIST_LIST)):
        kernel_raster_path = os.path.join(
            churn_dir, f'{pixel_radius}_kernel.tif')
        kernel_task = task_graph.add_task(
            func=make_kernel_raster,
            args=(pixel_radius, kernel_raster_path),
            target_path_list=[kernel_raster_path],
            task_name=f'make kernel of radius {pixel_radius}')
        kernel_raster_path_map[pixel_radius] = kernel_raster_path
        convolution_task_list = []
        for mask_type, (scenario_mask_path, mask_task) in \
                mask_path_task_map.items():
            LOGGER.debug(
                f'this is the scenario mask about to convolve: '
                f'{scenario_mask_path} {mask_task}')
            convolution_mask_raster_path = os.path.join(
                aligned_data_dir,
                f'mask_{mask_type}_gs{pixel_radius}.tif')
            convolution_task = task_graph.add_task(
                func=pygeoprocessing.convolve_2d,
                args=(
                    (scenario_mask_path, 1), (kernel_raster_path, 1),
                    convolution_mask_raster_path),
                dependent_task_list=[mask_task, kernel_task],
                target_path_list=[convolution_mask_raster_path],
                task_name=f'convolve {pixel_radius} {mask_type}')
            convolution_task_list.append(convolution_task)
    LOGGER.info('wait for convolution to complete')
    task_graph.join()

    # 2) Evalute the forest regression for each scenario
    LOGGER.info("Forest Regression step 2")

    mult_by_columns_workspace = os.path.join(
        aligned_data_dir, 'mult_by_columns_workspace')
    try:
        os.makedirs(mult_by_columns_workspace)
    except OSError:
        pass
    task_graph.join()

    base_projection = osr.SpatialReference()
    base_projection.ImportFromWkt(base_raster_info['projection_wkt'])

    # convert m^2 to to hectares and biomass to C
    conversion_factor = base_raster_info['pixel_size'][0]**2 * (1/10000) * 0.47
    if args.co2:
        conversion_factor *= (15.9992*2+12.011)/12.011  # C into CO2

    if not base_projection.IsProjected():
        # It's in lat/lng, make an estimate of degree pixel size in m^2
        conversion_factor *= 111120**2

    forest_carbon_stocks_raster_path = os.path.join(
        churn_dir, f'{landtype_basename}_forest_{c_prefix}.tif')

    mult_by_columns_library.mult_by_columns(
        FOREST_REGRESSION_LASSO_TABLE_PATH, aligned_data_dir,
        mult_by_columns_workspace,
        'lulc_esa_smoothed_2014_10sec', landtype_basename,
        base_raster_info['pixel_size'],
        forest_carbon_stocks_raster_path,
        task_graph, zero_nodata_symbols=ZERO_NODATA_SYMBOLS,
        target_nodata=MULT_BY_COLUMNS_NODATA,
        conversion_factor=conversion_factor)

    # NON-FOREST BIOMASS
    LOGGER.info(f'convert baccini non forest into {c_prefix}')
    baccini_aligned_raster_path = os.path.join(
        aligned_data_dir,
        os.path.basename(BACCINI_10s_2014_BIOMASS_RASTER_PATH))
    baccini_nodata = pygeoprocessing.get_raster_info(
        baccini_aligned_raster_path)['nodata'][0]
    baccini_c_raster_path = os.path.join(
        churn_dir, f'baccini_{c_prefix}_{landtype_basename}.tif')
    task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(baccini_aligned_raster_path, 1),
             (conversion_factor, 'raw'), (baccini_nodata, 'raw'),
             (MULT_BY_COLUMNS_NODATA, 'raw')],
            mult_by_const_op, baccini_c_raster_path, gdal.GDT_Float32,
            MULT_BY_COLUMNS_NODATA),
        target_path_list=[baccini_c_raster_path],
        task_name=f'convert baccini biomass density to {c_prefix}')

    task_graph.join()

    # combine both the non-forest and forest into one map for each
    # scenario based on their masks
    total_carbon_stocks_raster_path = os.path.join(
        workspace_dir, f'{c_prefix}_stocks_{landtype_basename}.tif')

    task_graph.add_task(
        func=raster_where,
        args=(
            mask_path_task_map['forest_10sec'][0],
            forest_carbon_stocks_raster_path,
            baccini_c_raster_path,
            total_carbon_stocks_raster_path),
        target_path_list=[
            total_carbon_stocks_raster_path],
        task_name=f'combine forest/nonforest')

    task_graph.close()
    task_graph.join()


if __name__ == '__main__':
    main()
