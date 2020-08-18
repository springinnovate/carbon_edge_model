"""One-time script used to generate base global data for the model."""
import os
import logging
import subprocess

import numpy
import pygeoprocessing
import retrying
import scipy

from osgeo import gdal

BASE_URL = 'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs'
BASE_URI = 'gs://ecoshard-root/global_carbon_regression/inputs'

BACCINI_10s_2014_BIOMASS_URI = (
    'gs://ecoshard-root/global_carbon_regression/'
    'baccini_10s_2014_md5_5956a9d06d4dffc89517cefb0f6bb008.tif')

ESA_LULC_URI = (
    'gs://ecoshard-root/global_carbon_regression/'
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif')

CARBON_EDGE_MODEL_DATA_NODATA = [
    ('accessibility_to_cities_2015_30sec.tif', -9999, None),
    ('ACDWRB_10sec.tif', 255, None),
    ('altitude_10sec.tif', None, None),
    ('AWCh1_10sec.tif', 255, None),
    ('AWCh2_10sec.tif', 255, None),
    ('AWCh3_10sec.tif', 255, None),
    ('AWCtS_10sec.tif', 255, None),
    ('bdod_10sec.tif', 0, None),
    ('BDRICM_10sec.tif', 255, None),
    ('BDRLOG_10sec.tif', 255, None),
    ('BDTICM_10sec.tif', 9999, None),
    ('bio_01_30sec.tif', -9999, None),
    ('bio_02_30sec.tif', -9999, None),
    ('bio_03_30sec.tif', -9999, None),
    ('bio_04_30sec.tif', -9999, None),
    ('bio_05_30sec.tif', -9999, None),
    ('bio_06_30sec.tif', -9999, None),
    ('bio_07_30sec.tif', -9999, None),
    ('bio_08_30sec.tif', -9999, None),
    ('bio_09_30sec.tif', -9999, None),
    ('bio_10_30sec.tif', -9999, None),
    ('bio_11_30sec.tif', -9999, None),
    ('bio_12_30sec.tif', -9999, None),
    ('bio_13_30sec.tif', -9999, None),
    ('bio_14_30sec.tif', -9999, None),
    ('bio_15_30sec.tif', -9999, None),
    ('bio_16_30sec.tif', -9999, None),
    ('bio_17_30sec.tif', -9999, None),
    ('bio_18_30sec.tif', -9999, None),
    ('bio_19_30sec.tif', -9999, None),
    ('BLDFIE_10sec.tif', 9999, None),
    ('cfvo_10sec.tif', -9999, None),
    ('clay_10sec.tif', -9999, None),
    ('CLYPPT_10sec.tif', 255, None),
    ('CRFVOL_10sec.tif', 255, None),
    ('hillshade_10sec.tif', 181, None),
    ('HISTPR_10sec.tif', 255, None),
    ('livestock_Bf_2010_5min.tif', -9999, 0.0),
    ('livestock_Ch_2010_5min.tif', -1.7e308, 0.0),
    ('livestock_Ct_2010_5min.tif', -1.7e308, 0.0),
    ('livestock_Dk_2010_5min.tif', -1.7e308, 0.0),
    ('livestock_Gt_2010_5min.tif', -1.7e308, 0.0),
    ('livestock_Ho_2010_5min.tif', -1.7e308, 0.0),
    ('livestock_Pg_2010_5min.tif', -1.7e308, 0.0),
    ('livestock_Sh_2010_5min.tif', -1.7e308, 0.0),
    ('ndvcec015_10sec.tif', 0, None),
    ('night_lights_10sec.tif', None, None),
    ('night_lights_5min.tif', None, None),
    ('nitrogen_10sec.tif', -9999, None),
    ('ocd_10sec.tif', -9999, None),
    ('OCDENS_10sec.tif', 9999, None),
    ('ocs_10sec.tif', -9999, None),
    ('OCSTHA_10sec.tif', 9999, None),
    ('phh2o_10sec.tif', -9999, None),
    ('PHIHOX_10sec.tif', 255, None),
    ('PHIKCL_10sec.tif', 255, None),
    ('population_2015_30sec.tif', 3.4028235e+38, 0.0),
    ('population_2015_5min.tif', 3.4028235e+38, 0.0),
    ('sand_10sec.tif', -9999, None),
    ('silt_10sec.tif', -9999, None),
    ('slope_10sec.tif', None, None),
    ('soc_10sec.tif', -9999, None),
    ('tri_10sec.tif', 0, None),
    ('wind_speed_10sec.tif', -999, None),
]

CROPLAND_LULC_CODES = range(10, 41)
URBAN_LULC_CODES = (190,)
FOREST_CODES = (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170)

MASK_TYPES = [
    ('cropland', CROPLAND_LULC_CODES),
    ('urban', URBAN_LULC_CODES),
    ('forest', FOREST_CODES)]
OTHER_TYPE = 4
MASK_NODATA = 127

LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)


def _reclassify_vals_op(array, mask_values):
    """Set values 1d array/array to nodata unless `inverse` then opposite."""
    result = numpy.empty(array.shape, dtype=numpy.uint8)
    result[:] = MASK_NODATA  # default is '4 -- other'
    for mask_id, code_list in mask_values:
        mask_array = numpy.in1d(array, code_list).reshape(result.shape)
        result[mask_array] = mask_id
    return result


def create_mask(base_raster_path, mask_values, target_raster_path):
    """Create a mask of base raster where in `mask_values` it's 1, else 0."""
    # reclassify clipped file as the output file
    pygeoprocessing.multiprocessing.raster_calculator(
        [(base_raster_path, 1), (mask_values, 'raw')],
        _reclassify_vals_op, target_raster_path, gdal.GDT_Byte, None)


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


def create_convolutions(
        task_graph, esa_lulc_raster_path, expected_max_edge_effect_km,
        data_dir):
    """Create forest convolution mask at `expected_max_edge_effect_km`.

    Args:
        excepcted_max_edge_effect_km (float): expected edge effect in km.

    Returns:
        None
    """
    # 2a) make 3km gaussian kernel
    # pixel_length is in degrees and we want about a 30km decay so do that:
    # (deg/pixel  * km/deg * 1/30km)^-1
    # ~111km / degree

    churn_dir = os.path.join(data_dir, 'convolution_kernels')
    try:
        os.makedirs(churn_dir)
    except OSError:
        pass
    pixel_size = pygeoprocessing.get_raster_info(
        esa_lulc_raster_path)['pixel_size']
    # this is calculated as 111km per degree
    pixel_radius = (pixel_size[0] * 111 / expected_max_edge_effect_km)**-1
    convolution_raster_list = []
    for mask_id, mask_code_list in MASK_TYPES:
        mask_raster_path = os.path.join(data_dir, f'{mask_id}_mask.tif')
        create_mask_task = task_graph.add_task(
            func=create_mask,
            args=(esa_lulc_raster_path, mask_code_list, mask_raster_path),
            target_path_list=[mask_raster_path],
            task_name=f'create {mask_id} mask')

        kernel_raster_path = os.path.join(
            churn_dir, f'kernel_{pixel_radius}.tif')
        kernel_task = task_graph.add_task(
            func=make_kernel_raster,
            args=(pixel_radius, kernel_raster_path),
            target_path_list=[kernel_raster_path],
            task_name=f'make kernel of radius {pixel_radius}')

        mask_gf_path = (
            f'{os.path.splitext(mask_raster_path)[0]}_gf_'
            f'{expected_max_edge_effect_km}.tif')

        convolution_task = task_graph.add_task(
            func=pygeoprocessing.convolve_2d,
            args=(
                (mask_raster_path, 1), (kernel_raster_path, 1),
                mask_gf_path),
            dependent_task_list=[create_mask_task, kernel_task],
            target_path_list=[mask_gf_path],
            task_name=f'create guassian filter of {mask_id}')
        convolution_raster_list.append(((mask_gf_path, None, None)))
    task_graph.join()
    return convolution_raster_list


@retrying.retry(wait_exponential_multiplier=100, wait_exponential_max=5000)
def download_gs(base_uri, target_path, skip_if_target_exists=False):
    """Download base to target."""
    try:
        if not(skip_if_target_exists and os.path.exists(target_path)):
            subprocess.run(
                f'/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil cp '
                f'{base_uri} {target_path}',
                check=True, shell=True)
    except Exception:
        LOGGER.exception(
            f'exception during download of {base_uri} to {target_path}')
        raise


def fetch_data(data_dir, task_graph):
    """Download all the global data needed to run this analysis.

    Args:
        bounding_box (list): minx, miny, maxx, maxy list to clip to
        data_dir (str): path to directory to copy clipped rasters
            to
        task_graph (TaskGraph): taskgraph object to schedule work.

    Returns:
        List of (file_path, nodata, nodata_replacement) tuples.

    """
    try:
        os.makedirs(data_dir)
    except OSError:
        pass

    downloaded_file_list = []
    for file_uri, nodata, nodata_replacement in \
            CARBON_EDGE_MODEL_DATA_NODATA + [
                (BACCINI_10s_2014_BIOMASS_URI, None, None),
                (ESA_LULC_URI, None, None)]:
        target_file_path = os.path.join(
            data_dir, os.path.basename(file_uri))
        _ = task_graph.add_task(
            func=download_gs,
            args=(file_uri, target_file_path),
            kwargs={'skip_if_target_exists': True},
            target_path_list=[target_file_path],
            task_name=f'download {file_uri} to {data_dir}')
        if file_uri not in [BACCINI_10s_2014_BIOMASS_URI, ESA_LULC_URI]:
            downloaded_file_list.append(
                (target_file_path, nodata, nodata_replacement))

    task_graph.join()
    return downloaded_file_list
