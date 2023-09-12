"""One-time script used to generate base global data for the model."""
import os
import logging
import multiprocessing
import subprocess

import ecoshard
import numpy
import pygeoprocessing
import pygeoprocessing.multiprocessing
import retrying
import scipy
import taskgraph

from osgeo import gdal

BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'model_base_data')
BASE_URL = (
    'https://storage.googleapis.com/ecoshard-root/'
    'global_carbon_regression/inputs')

BASE_URL = (
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression_2/'
    'inputs/')

BACCINI_10s_2014_BIOMASS_URL = (
    'https://storage.googleapis.com/ecoshard/ecoshard-root/global_carbon_regression/'
    'baccini_10s_2014_md5_5956a9d06d4dffc89517cefb0f6bb008.tif')

ESA_LULC_URL = (
    'https://storage.googleapis.com/ecoshard/ecoshard-root/global_carbon_regression/'
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif')

CARBON_EDGE_MODEL_DATA_NODATA = [
    ('accessibility_to_cities_2015_30sec_compressed_wgs84__md5_a6a8ffcb6c1025c131f7663b80b3c9a7.tif', -9999),
    ('altitude_10sec_compressed_wgs84__md5_bfa771b1aef1b18e48962c315e5ba5fc.tif', None),
    ('bio_01_30sec_compressed_wgs84__md5_3f851546237e282124eb97b479c779f4.tif', -9999),
    ('bio_02_30sec_compressed_wgs84__md5_7ad508baff5bbd8b2e7991451938a5a7.tif', -9999),
    ('bio_03_30sec_compressed_wgs84__md5_a2de2d38c1f8b51f9d24f7a3a1e5f142.tif', -9999),
    ('bio_04_30sec_compressed_wgs84__md5_94cfca6af74ffe52316a02b454ba151b.tif', -9999),
    ('bio_05_30sec_compressed_wgs84__md5_bdd225e46613405c80a7ebf7e3b77249.tif', -9999),
    ('bio_06_30sec_compressed_wgs84__md5_ef252a4335eafb7fe7b4dc696d5a70e3.tif', -9999),
    ('bio_07_30sec_compressed_wgs84__md5_1db9a6cdce4b3bd26d79559acd2bc525.tif', -9999),
    ('bio_08_30sec_compressed_wgs84__md5_baf898dd624cfc9415092d7f37ae44ff.tif', -9999),
    ('bio_09_30sec_compressed_wgs84__md5_180c820aae826529bfc824b458165eee.tif', -9999),
    ('bio_10_30sec_compressed_wgs84__md5_d720d781970e165a40a1934adf69c80e.tif', -9999),
    ('bio_11_30sec_compressed_wgs84__md5_f48a251c54582c22d9eb5d2158618bbe.tif', -9999),
    ('bio_12_30sec_compressed_wgs84__md5_23cb55c3acc544e5a941df795fcb2024.tif', -9999),
    ('bio_13_30sec_compressed_wgs84__md5_b004ebe58d50841859ea485c06f55bf6.tif', -9999),
    ('bio_14_30sec_compressed_wgs84__md5_7cb680af66ff6c676441a382519f0dc2.tif', -9999),
    ('bio_15_30sec_compressed_wgs84__md5_edc8e5af802448651534b7a0bd7113ac.tif', -9999),
    ('bio_16_30sec_compressed_wgs84__md5_a9e737a926f1f916746d8ce429c06fad.tif', -9999),
    ('bio_17_30sec_compressed_wgs84__md5_0bc4db0e10829cd4027b91b7bbfc560f.tif', -9999),
    ('bio_18_30sec_compressed_wgs84__md5_76cf3d38eb72286ba3d5de5a48bfadd4.tif', -9999),
    ('bio_19_30sec_compressed_wgs84__md5_a91b8b766ed45cb60f97e25bcac0f5d2.tif', -9999),
    ('cec_0-5cm_mean_compressed_wgs84__md5_b3b4285906c65db596a014d0c8a927dd.tif', None),
    ('cec_0-5cm_uncertainty_compressed_wgs84__md5_f0f4eb245fd2cc4d5a12bd5f37189b53.tif', None),
    ('cec_5-15cm_mean_compressed_wgs84__md5_55c4d960ca9006ba22c6d761d552c82f.tif', None),
    ('cec_5-15cm_uncertainty_compressed_wgs84__md5_880eac199a7992f61da6c35c56576202.tif', None),
    ('cfvo_0-5cm_mean_compressed_wgs84__md5_7abefac8143a706b66a1b7743ae3cba1.tif', None),
    ('cfvo_0-5cm_uncertainty_compressed_wgs84__md5_3d6b883fba1d26a6473f4219009298bb.tif', None),
    ('cfvo_5-15cm_mean_compressed_wgs84__md5_ae36d799053697a167d114ae7821f5da.tif', None),
    ('cfvo_5-15cm_uncertainty_compressed_wgs84__md5_1f2749cd35adc8eb1c86a67cbe42aebf.tif', None),
    ('clay_0-5cm_mean_compressed_wgs84__md5_9da9d4017b691bc75c407773269e2aa3.tif', None),
    ('clay_0-5cm_uncertainty_compressed_wgs84__md5_f38eb273cb55147c11b48226400ae79a.tif', None),
    ('clay_5-15cm_mean_compressed_wgs84__md5_c136adb39b7e1910949b749fcc16943e.tif', None),
    ('clay_5-15cm_uncertainty_compressed_wgs84__md5_0acc36c723aa35b3478f95f708372cc7.tif', None),
    ('hillshade_10sec_compressed_wgs84__md5_192a760d053db91fc9e32df199358b54.tif', None),
    ('night_lights_10sec_compressed_wgs84__md5_54e040d93463a2918a82019a0d2757a3.tif', None),
    ('night_lights_5min_compressed_wgs84__md5_e36f1044d45374c335240777a2b94426.tif', None),
    ('nitrogen_0-5cm_mean_compressed_wgs84__md5_6adecc8d790ccca6057a902e2ddd0472.tif', None),
    ('nitrogen_0-5cm_uncertainty_compressed_wgs84__md5_4425b4bd9eeba0ad8a1092d9c3e62187.tif', None),
    ('nitrogen_10sec_compressed_wgs84__md5_1aed297ef68f15049bbd987f9e98d03d.tif', None),
    ('nitrogen_5-15cm_mean_compressed_wgs84__md5_9487bc9d293effeb4565e256ed6e0393.tif', None),
    ('nitrogen_5-15cm_uncertainty_compressed_wgs84__md5_2de5e9d6c3e078756a59ac90e3850b2b.tif', None),
    ('phh2o_0-5cm_mean_compressed_wgs84__md5_00ab8e945d4f7fbbd0bddec1cb8f620f.tif', None),
    ('phh2o_0-5cm_uncertainty_compressed_wgs84__md5_8090910adde390949004f30089c3ae49.tif', None),
    ('phh2o_5-15cm_mean_compressed_wgs84__md5_9b187a088ecb955642b9a86d56f969ad.tif', None),
    ('phh2o_5-15cm_uncertainty_compressed_wgs84__md5_6809da4b13ebbc747750691afb01a119.tif', None),
    ('sand_0-5cm_mean_compressed_wgs84__md5_6c73d897cdef7fde657386af201a368d.tif', None),
    ('sand_0-5cm_uncertainty_compressed_wgs84__md5_efd87fd2062e8276148154c4a59c9b25.tif', None),
    ('sand_5-15cm_uncertainty_compressed_wgs84__md5_03bc79e2bfd770a82c6d15e36a65fb5c.tif', None),
    ('silt_0-5cm_mean_compressed_wgs84__md5_1d141933d8d109df25c73bd1dcb9d67c.tif', None),
    ('silt_0-5cm_uncertainty_compressed_wgs84__md5_ac5ec50cbc3b9396cf11e4e431b508a9.tif', None),
    ('silt_5-15cm_mean_compressed_wgs84__md5_d0abb0769ebd015fdc12b50b20f8c51e.tif', None),
    ('silt_5-15cm_uncertainty_compressed_wgs84__md5_cc125c85815db0d1f66b315014907047.tif', None),
    ('slope_10sec_compressed_wgs84__md5_e2bdd42cb724893ce8b08c6680d1eeaf.tif', None),
    ('soc_0-5cm_mean_compressed_wgs84__md5_b5be42d9d0ecafaaad7cc592dcfe829b.tif', None),
    ('soc_0-5cm_uncertainty_compressed_wgs84__md5_33c1a8c3100db465c761a9d7f4e86bb9.tif', None),
    ('soc_5-15cm_mean_compressed_wgs84__md5_4c489f6132cc76c6d634181c25d22d19.tif', None),
    ('tri_10sec_compressed_wgs84__md5_258ad3123f05bc140eadd6246f6a078e.tif', None),
    ('wind_speed_10sec_compressed_wgs84__md5_7c5acc948ac0ff492f3d148ffc277908.tif', None),
    ('population_2015_5min_compressed_md5_4c267e3cb681689acc08020fd64e023d.tif', 3.4028235e+38, 0.0),
]

MASK_TYPES = [
    ('cropland', 1),
    ('urban', 2),
    ('forest', 3)]
OTHER_TYPE = 4

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


@retrying.retry(wait_exponential_multiplier=100, wait_exponential_max=5000)
def download_gs(base_uri, target_path, skip_if_target_exists=False):
    """Download base to target."""
    try:
        if not(skip_if_target_exists and os.path.exists(target_path)):
            subprocess.run(
                f'/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil -m cp '
                f'{base_uri} {target_path}',
                check=True, shell=True)
    except Exception:
        LOGGER.exception(
            f'exception during download of {base_uri} to {target_path}')
        raise
