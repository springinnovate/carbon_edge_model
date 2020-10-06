"""Carbon edge regression model."""
import argparse
import logging
import multiprocessing
import os
import pickle
import sys

from osgeo import gdal
from osgeo import osr
import pygeoprocessing
import pygeoprocessing.multiprocessing
import numpy
import scipy.ndimage
import taskgraph

from carbon_model_data import BASE_DATA_DIR
from carbon_model_data import BACCINI_10s_2014_BIOMASS_URI
import carbon_model_data

gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)

LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)

MAX_CARBON = 368  # 99th percentile of baccini

MODEL_URI = (
    'gs://ecoshard-root/global_carbon_regression/models/'
    'carbon_model_lsvr_poly_2_90000_pts.mod')

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


def _carbon_op(*args):
    """Evaluate carbon model.

    Args:
        args (list): a list length 3*n+2 long where the first 3*n elements
            are (array, nodata, nodata_replace) tuples, these arrays
            are replaced or nodata'd then passed to the 3*n+2th argument
            which is the carbon model, 3*n+1th is nodata.

    Returns:
        Evaluation of array on given arg model.
    """
    result = numpy.empty(args[0].shape, dtype=numpy.float32)
    n = (len(args)-2) // 3
    result[:] = args[3*n]  # assign target nodata
    model = args[3*n+1]
    valid_mask = numpy.ones(args[0].shape, dtype=numpy.bool)
    it = iter(args[0:3*n])
    for index, array in enumerate(it):
        nodata = next(it)
        nodata_replace = next(it)
        if nodata is not None:
            nodata_mask = numpy.isclose(array, nodata)
            if nodata_replace is not None:
                array[nodata_mask] = nodata_replace
            else:
                valid_mask &= ~nodata_mask

    if numpy.count_nonzero(valid_mask) > 0:
        # .predict will crash if there's an empty list passed to it
        array_arg_list = numpy.array(
            [array[valid_mask] for array in args[0:3*n:3]])
        result[valid_mask] = model.predict(array_arg_list.transpose())
        # clamp just in case
        result[valid_mask & (result > MAX_CARBON)] = MAX_CARBON
        result[valid_mask & (result < 0)] = 0
    return result


def sub_pos_op(array_a, array_b):
    """Assume nodata value is negative and the same for a and b."""
    result = array_a.copy()
    mask = array_b > 0
    result[mask] -= array_b[mask]
    return result


def where_op(
        condition_array, if_true_array, else_array, upper_threshold, nodata):
    """Select from `if true array` if condition true, `else array`."""
    result = numpy.copy(else_array).astype(numpy.float32)
    mask = condition_array == 1
    result[mask] = if_true_array[mask]
    invalid_mask = (
        numpy.isnan(result) | numpy.isinf(result) | (result < 0) |
        (result > upper_threshold))
    result[invalid_mask] = nodata
    return result


def raster_where(
        condition_raster_path, if_true_raster_path, else_raster_path,
        upper_threshold, target_raster_path):
    """A raster version of the numpy.where function."""
    nodata = pygeoprocessing.get_raster_info(if_true_raster_path)['nodata'][0]
    LOGGER.debug(
        f'selecting {if_true_raster_path} if {condition_raster_path} is 1 '
        f'else {else_raster_path}, upper upper_threshold {upper_threshold}, '
        f'target is {target_raster_path}')
    pygeoprocessing.raster_calculator(
        [(condition_raster_path, 1), (if_true_raster_path, 1),
         (else_raster_path, 1), (upper_threshold, 'raw'), (nodata, 'raw')],
        where_op, target_raster_path, gdal.GDT_Float32, nodata)


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


def warp_and_gaussian_filter_data(
        landcover_type_raster_path, base_data_dir,
        target_data_dir, task_graph):
    """Clip required data to fit landcover and apply gaussian filtering.

    Args:
        landcover_type_raster_path (str): path to landcover raster used to
            infer target raster size and projection.
        base_data_dir (str): location of base model data files expected by
            carbon_model_data.CARBON_EDGE_MODEL_DATA_NODATA.
        target_data_dir (str): path to directory to contain all warped and
            filtered files required to run the model on a particular scenario.
        task_graph (TaskGraph): object used for concurrent execution and
            avoided reexecution.

    Returns:
        None.

    """
    # Expected data is given by `carbon_model_data`.
    base_raster_data_path_list = [
        os.path.join(base_data_dir, filename)
        for filename, _, _ in
        carbon_model_data.CARBON_EDGE_MODEL_DATA_NODATA] + [
        os.path.join(BASE_DATA_DIR, os.path.basename(
            BACCINI_10s_2014_BIOMASS_URI))]
    # sanity check:
    missing_raster_list = []
    for path in base_raster_data_path_list:
        if not os.path.exists(path):
            missing_raster_list.append(path)
    if missing_raster_list:
        raise ValueError(
            f'Expected the following files that did not exist: '
            f'{missing_raster_list}')

    base_raster_info = pygeoprocessing.get_raster_info(
        landcover_type_raster_path)
    aligned_raster_path_list = [
        os.path.join(target_data_dir, os.path.basename(path))
        for path in base_raster_data_path_list]
    # TODO: if 'warp' bounding box is the same as target, then hardlink
    for base_raster_path, target_aligned_raster_path in zip(
            base_raster_data_path_list, aligned_raster_path_list):
        if carbon_model_data.same_coverage(
                base_raster_path, landcover_type_raster_path):
            LOGGER.info(
                f'{base_raster_path} and {landcover_type_raster_path} are '
                f'aligned already, hardlinking to '
                f'{target_aligned_raster_path}')
            if os.path.exists(target_aligned_raster_path):
                LOGGER.warn(
                    f'{target_aligned_raster_path} already exists, removing '
                    'so we can hard link')
                os.remove(target_aligned_raster_path)
            os.link(base_raster_path, target_aligned_raster_path)
            continue

        _ = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(
                base_raster_path, base_raster_info['pixel_size'],
                target_aligned_raster_path, 'near'),
            kwargs={
                'target_bb': base_raster_info['bounding_box'],
                'target_projection_wkt': base_raster_info['projection_wkt'],
                'working_dir': target_data_dir,
                },
            target_path_list=[target_aligned_raster_path],
            task_name=f'align {base_raster_path} data')
    LOGGER.info('wait for data to align')
    task_graph.join()

    mask_path_task_map = {}
    LOGGER.info('separate out landcover masks')
    for mask_type, lulc_codes in MASK_TYPES:
        lulc_mask_raster_path = os.path.join(
            target_data_dir, f'mask_of_{mask_type}.tif')
        mask_task = task_graph.add_task(
            func=mask_ranges,
            args=(
                landcover_type_raster_path, lulc_codes,
                lulc_mask_raster_path),
            target_path_list=[lulc_mask_raster_path],
            hash_algorithm='md5',
            copy_duplicate_artifact=True,
            hardlink_allowed=True,
            task_name=f'make {mask_type}')
        mask_path_task_map[mask_type] = (lulc_mask_raster_path, mask_task)

    LOGGER.info('create gaussian filter of landcover types')
    convolution_file_paths = carbon_model_data.create_convolutions(
        landcover_type_raster_path,
        carbon_model_data.EXPECTED_MAX_EDGE_EFFECT_KM_LIST,
        target_data_dir, task_graph)
    LOGGER.info('wait for convolution to complete')
    task_graph.join()
    return convolution_file_paths


def evaluate_model_with_landcover(
        carbon_model, landcover_type_raster_path, convolution_file_paths,
        workspace_dir, data_dir,
        n_workers, file_suffix, max_biomass=368.0):
    """Evaluate the model over a landcover raster.

    Args:
        carbon_model (scikit.learn.model): a trained model expecting vector
            input for output
        landcover_type_raster_path (str): path to ESA style landcover raster
        convolution_file_paths (list): list of convolutions raster paths in
            the same order as expected by the model.
        workspace_dir (str): path to general workspace dir
        data_dir (str): path to directory containing base data refered to
            by CARBON_EDGE_MODEL_DATA_NODATA.
        n_workers (int): number of workers to allocate to raster calculator
        file_suffix (str): append this to target rasters
        max_biomass (float): threshold modeled biomass to this value

    Returns:
        (str) Path to created biomass_per_ha raster.

    """
    landtype_basename = os.path.basename(
        os.path.splitext(landcover_type_raster_path)[0])
    base_raster_info = pygeoprocessing.get_raster_info(
        landcover_type_raster_path)
    base_projection = osr.SpatialReference()
    base_projection.ImportFromWkt(base_raster_info['projection_wkt'])

    churn_dir = os.path.join(workspace_dir, 'churn')
    try:
        os.makedirs(churn_dir)
    except OSError:
        pass

    raster_info_tuple_list = [
        ((os.path.join(data_dir, filename), 1),
         (nodata, 'raw'),
         (nodata_replace, 'raw'))
        for filename, nodata, nodata_replace in
        carbon_model_data.CARBON_EDGE_MODEL_DATA_NODATA] + \
        [((path, 1), (nodata, 'raw'), (nodata_replace, 'raw'))
         for path, nodata, nodata_replace in convolution_file_paths]

    target_nodata = -1
    raster_path_band_list = [
        item for sublist in raster_info_tuple_list for item in sublist] + \
        [(target_nodata, 'raw'), (carbon_model, 'raw')]

    # This raster is the predicted forest biomass
    forest_carbon_stocks_raster_path = os.path.join(
        churn_dir,
        f'{landtype_basename}_forest_biomass_per_ha{file_suffix}.tif')
    LOGGER.info(
        f'scheduling the regression model on {raster_path_band_list} to '
        f'target {forest_carbon_stocks_raster_path}')
    if n_workers == -1:
        pygeoprocessing.raster_calculator(
            raster_path_band_list, _carbon_op,
            forest_carbon_stocks_raster_path, gdal.GDT_Float32, target_nodata)
    else:
        pygeoprocessing.multiprocessing.raster_calculator(
            raster_path_band_list, _carbon_op,
            forest_carbon_stocks_raster_path, gdal.GDT_Float32, target_nodata,
            n_workers=n_workers)

    # NON-FOREST BIOMASS
    LOGGER.info(f'convert baccini non forest into biomass_per_ha')
    baccini_aligned_raster_path = os.path.join(
        data_dir, os.path.basename(BACCINI_10s_2014_BIOMASS_URI))

    # combine both the non-forest and forest into one map for each
    # scenario based on their masks
    total_biomass_stocks_raster_path = os.path.join(
        workspace_dir,
        f'biomass_per_ha_stocks_{landtype_basename}{file_suffix}.tif')

    forest_mask_path = os.path.join(
        data_dir, f'mask_of_forest_10sec.tif')

    LOGGER.debug(
        f'selecting {forest_carbon_stocks_raster_path} '
        f'if {forest_mask_path} is 1 '
        f'else {baccini_aligned_raster_path}, upper '
        f'upper_threshold {max_biomass}\n'
        f'result in {total_biomass_stocks_raster_path}')

    raster_where(
        forest_mask_path,
        forest_carbon_stocks_raster_path,
        baccini_aligned_raster_path, max_biomass,
        total_biomass_stocks_raster_path)

    return total_biomass_stocks_raster_path


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Carbon edge model')
    parser.add_argument(
        '--landcover_type_raster_path', help=(
            'Path to landtype raster where codes correspond to:\n'
            '\t1: cropland\n\t2: urban\n\t3: forest\n\t4: other'))
    parser.add_argument(
        '--workspace_dir', default='carbon_model_workspace', help=(
            'Path to workspace dir, the carbon stock file will be named '
            '"c_stocks_[landcover_type_raster_path]. Default is '
            '`carbon_model_workspace`"'))
    parser.add_argument(
        '--local_model_path', help='point to local model rather than default')
    parser.add_argument(
        '--n_workers', type=int, default=multiprocessing.cpu_count(), help=(
            'number of cpu workers to allocate'))
    parser.add_argument(
        '--file_suffix', default='',
        help='add this to the end of output files')

    args = parser.parse_args()
    workspace_dir = args.workspace_dir
    churn_dir = os.path.join(workspace_dir, 'churn')

    for dir_path in [workspace_dir, churn_dir]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    task_graph = taskgraph.TaskGraph(BASE_DATA_DIR, args.n_workers, 5.0)

    LOGGER.info("download data")
    carbon_model_data.fetch_data(BASE_DATA_DIR, task_graph)

    model_dir = os.path.join(BASE_DATA_DIR, 'models')
    try:
        os.makedirs(model_dir)
    except OSError:
        pass

    if not args.local_model_path:
        model_path = os.path.join(model_dir, os.path.basename(MODEL_URI))
        _ = task_graph.add_task(
            func=carbon_model_data.download_gs,
            args=(MODEL_URI, model_path),
            target_path_list=[model_path],
            task_name=f'download model {MODEL_URI} to {model_path}')
        task_graph.join()
    else:
        model_path = args.local_model_path

    LOGGER.info("prep data")
    convolution_file_paths = warp_and_gaussian_filter_data(
        args.landcover_type_raster_path, BASE_DATA_DIR, churn_dir, task_graph)
    task_graph.join()

    LOGGER.info('evaulate carbon model')

    with open(model_path, 'rb') as model_file:
        carbon_model = pickle.load(model_file)
    evaluate_model_with_landcover(
        carbon_model, args.landcover_type_raster_path, convolution_file_paths,
        workspace_dir, churn_dir, args.n_workers, args.file_suffix)

    task_graph.close()
    task_graph.join()


if __name__ == '__main__':
    main()
