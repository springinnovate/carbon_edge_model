"""One-time script used to generate base global data for the model."""
import glob
import os
import logging
import multiprocessing
import re

import ecoshard
import numpy
from ecoshard import geoprocessing
import scipy
import taskgraph

from osgeo import gdal


logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)

logging.getLogger('matplotlib').setLevel(logging.WARN)
logging.getLogger('fiona').setLevel(logging.WARN)

WORLD_ECKERT_IV_WKT = """PROJCRS["unknown",
    BASEGEOGCRS["GCS_unknown",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1]],
            ID["EPSG",6326]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["unnamed",
        METHOD["Eckert IV"],
        PARAMETER["Longitude of natural origin",0,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8802]],
        PARAMETER["False easting",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8806]],
        PARAMETER["False northing",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8807]]],
    CS[Cartesian,2],
        AXIS["(E)",east,
            ORDER[1],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]],
        AXIS["(N)",north,
            ORDER[2],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]]]"""

PIXEL_SIZE = (300, -300)
ALIGNED_WORKSPACE = 'aligned_rasters'
os.makedirs(ALIGNED_WORKSPACE, exist_ok=True)
MASK_DIR = os.path.join(ALIGNED_WORKSPACE, 'mask_dir')
os.makedirs(MASK_DIR, exist_ok=True)
CHURN_DIR = os.path.join(ALIGNED_WORKSPACE, 'churn_dir')
os.makedirs(CHURN_DIR, exist_ok=True)

GLOBAL_ECKERT_IV_BB = [-16921202.923, -8460601.461, 16921797.077, 8461398.539]

CROPLAND_LULC_CODES = tuple(range(10, 41))
URBAN_LULC_CODES = (190,)
FOREST_CODES = (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170)

MASK_TYPES = [
    ('cropland', CROPLAND_LULC_CODES),
    ('urban', URBAN_LULC_CODES),
    ('forest', FOREST_CODES)]

EXPECTED_MAX_EDGE_EFFECT_KM_LIST = [1.0, 3.0, 10.0, 30.0]

MASK_NODATA = 127

LOGGER = logging.getLogger(__name__)


def same_coverage(raster_a_path, raster_b_path):
    """Return true if raster a and b have same pixels and bounding box."""
    raster_a_info = pygeoprocessing.get_raster_info(raster_a_path)
    raster_b_info = pygeoprocessing.get_raster_info(raster_b_path)
    if raster_a_info['raster_size'] != raster_b_info['raster_size']:
        return False
    if not numpy.isclose(
            raster_a_info['bounding_box'],
            raster_b_info['bounding_box']).all():
        return False
    return True


def _reclassify_vals_op(array, array_nodata, mask_values):
    """Set values 1d array/array to nodata unless `inverse` then opposite.

    Args:
        array (numpy.ndarray): base integer array containing either nodata or
            possible value in mask values
        array_nodata (int): nodata value for array
        mask_values (tuple/list): values to set to 1 in ``array``.

    Returns:
        values in ``array`` set to 1 where in mask_values, 0 otherwise, or
            nodata.

    """
    result = numpy.zeros(array.shape, dtype=numpy.uint8)
    if array_nodata is not None:
        result[numpy.isclose(array, array_nodata)] = MASK_NODATA
    mask_array = numpy.in1d(array, mask_values).reshape(result.shape)
    result[mask_array] = 1
    return result


def create_mask(base_raster_path, mask_values, target_raster_path):
    """Create a mask of base raster where in `mask_values` it's 1, else 0."""
    # reclassify clipped file as the output file
    nodata = pygeoprocessing.get_raster_info(base_raster_path)['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1), (nodata, 'raw'), (mask_values, 'raw')],
        _reclassify_vals_op, target_raster_path, gdal.GDT_Byte, None)


def make_kernel_raster(pixel_radius, target_path):
    """Create kernel with given radius to `target_path`."""
    truncate = 2
    size = int(pixel_radius * 2 * truncate + 1)
    step_fn = numpy.zeros((size, size))
    step_fn[size//2, size//2] = 1
    kernel_array = scipy.ndimage.filters.gaussian_filter(
        step_fn, pixel_radius, order=0, mode='constant', cval=0.0,
        truncate=truncate)
    pygeoprocessing.numpy_array_to_raster(
        kernel_array, -1, (1., -1.), (0.,  0.), None,
        target_path)


def create_convolutions(
        landcover_type_raster_path, expected_max_edge_effect_km_list,
        target_data_dir, task_graph):
    """Create forest convolution mask at `expected_max_edge_effect_km`.

    Args:
        landcover_type_raster_path (path): path to raster containing 4 codes
            representing:
                1: cropland
                2: urban
                3: forest
                4: other

        excepcted_max_edge_effect_km_list (list): list of floats of
            expected edge effect in km.
        target_data_dir (path): path to directory to write resulting files
        task_graph (TaskGraph): object used to schedule work and avoid
            reexecution.

    Returns:
        List of convolution file paths created by this function
    """
    churn_dir = os.path.join(target_data_dir, 'convolution_kernels')
    LOGGER.debug(f'making convolution kernel churn dir at {churn_dir}')
    try:
        os.makedirs(churn_dir)
    except OSError:
        pass
    pixel_size = pygeoprocessing.get_raster_info(
        landcover_type_raster_path)['pixel_size']

    # this is calculated as 111km per degree
    convolution_raster_list = []
    for expected_max_edge_effect_km in expected_max_edge_effect_km_list:
        pixel_radius = (pixel_size[0] * 111 / expected_max_edge_effect_km)**-1
        kernel_raster_path = os.path.join(
            churn_dir, f'kernel_{pixel_radius}.tif')
        kernel_task = task_graph.add_task(
            func=make_kernel_raster,
            args=(pixel_radius, kernel_raster_path),
            target_path_list=[kernel_raster_path],
            task_name=f'make kernel of radius {pixel_radius}')

        for mask_id, mask_code in MASK_TYPES:
            mask_raster_path = os.path.join(
                target_data_dir, f'{mask_id}_mask.tif')
            create_mask_task = task_graph.add_task(
                func=create_mask,
                args=(landcover_type_raster_path,
                      (mask_code,), mask_raster_path),
                target_path_list=[mask_raster_path],
                task_name=f'create {mask_id} mask')

            mask_gf_path = (
                f'{os.path.splitext(mask_raster_path)[0]}_gf_'
                f'{expected_max_edge_effect_km}.tif')
            LOGGER.debug(f'making convoluion for {mask_gf_path}')
            convolution_task = task_graph.add_task(
                func=pygeoprocessing.convolve_2d,
                args=(
                    (mask_raster_path, 1), (kernel_raster_path, 1),
                    mask_gf_path),
                dependent_task_list=[create_mask_task, kernel_task],
                target_path_list=[mask_gf_path],
                task_name=f'create guassian filter of {mask_id} at {mask_gf_path}')
            convolution_raster_list.append(((mask_gf_path, None, None)))
    task_graph.join()
    LOGGER.debug(f'all done convolutoin list - {convolution_raster_list}')
    return convolution_raster_list


def main():
    """Entry point."""
    LOGGER.info('align everything')
    task_graph = taskgraph.TaskGraph('.', multiprocessing.cpu_count(), 5.0)
    for raster_path in glob.glob('raw_rasters/*.tif'):
        LOGGER.debug(f'process {raster_path}')
        target_raster_path = os.path.join(
            ALIGNED_WORKSPACE, os.path.basename(raster_path))
        task_graph.add_task(
            func=geoprocessing.warp_raster,
            args=(
                raster_path, PIXEL_SIZE, target_raster_path,
                'near'),
            kwargs={
                'target_bb': GLOBAL_ECKERT_IV_BB,
                'target_projection_wkt': WORLD_ECKERT_IV_WKT,
                },
            target_path_list=[target_raster_path],
            task_name=f'warp {os.path.basename(raster_path)}')

    kernel_raster_path_list = []
    for expected_max_edge_effect_km in EXPECTED_MAX_EDGE_EFFECT_KM_LIST:
        pixel_radius = 100*expected_max_edge_effect_km/PIXEL_SIZE[0]
        kernel_raster_path = os.path.join(
            CHURN_DIR, f'kernel_{pixel_radius}.tif')
        kernel_task = task_graph.add_task(
            func=make_kernel_raster,
            args=(pixel_radius, kernel_raster_path),
            target_path_list=[kernel_raster_path],
            task_name=f'make kernel of radius {pixel_radius}')
        kernel_raster_path_list.append(
            (kernel_raster_path, expected_max_edge_effect_km))

    task_graph.join()

    LOGGER.info('process by year')
    # make urban, forest, and crop masks for ESACCI-LC-L4-LCCS-Map-300m-P1Y-20\d\d-v2.0.7_smooth_compressed
    landcover_raster_path_list = [
        path for path in glob.glob(os.path.join(ALIGNED_WORKSPACE, '*.tif'))
        if re.match(r'.*ESACCI-LC-L4-LCCS-Map-300m-P1Y-20\d\d-v2.0.7_smooth_compressed.tif', path)]
    for landcover_raster_path in landcover_raster_path_list:
        basename = re.match(
            r'.*(ESACCI-LC-L4-LCCS-Map-300m-P1Y-20\d\d).*',
            landcover_raster_path).group(1)
        for mask_id, lucodes in MASK_TYPES:
            LOGGER.info(f'mask {mask_id} on {landcover_raster_path}')
            mask_raster_path = os.path.join(
                MASK_DIR, f'{basename}_{mask_id}.tif')
            mask_task = task_graph.add_task(
                func=_mask_raster,
                args=(landcover_raster_path, mask_raster_path),
                target_path_list=[mask_raster_path],
                task_id=f'mask out {mask_id}')

            for kernel_raster_path, expected_max_edge_effect_km in \
                    kernel_raster_path_list:
                mask_gf_path = os.path.join(
                    MASK_DIR,
                    f'{basename}_{mask_id}_gf_{expected_max_edge_effect_km}.tif')
                LOGGER.debug(f'making convoluion for {mask_gf_path}')
                convolution_task = task_graph.add_task(
                    func=geoprocessing.convolve_2d,
                    args=(
                        (mask_raster_path, 1), (kernel_raster_path, 1),
                        mask_gf_path),
                    dependent_task_list=[mask_task, kernel_task],
                    target_path_list=[mask_gf_path],
                    task_name=f'create guassian filter of {mask_id} at {mask_gf_path}')

    task_graph.join()
    task_graph.close()


def _mask_raster(base_raster_path, integer_codes, target_raster_path):
    """Mask any integer codes in base to 1."""
    def _reclassify_op(array):
        """Set values 1d array/array to nodata unless `inverse` then opposite."""
        result = numpy.in1d(array, integer_codes).reshape(array.shape)
        return result

    geoprocessing.raster_calculator(
        [(base_raster_path, 1)], _reclassify_op, target_raster_path,
        gdal.GDT_Byte, None)


if __name__ == '__main__':
    main()
