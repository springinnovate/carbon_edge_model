"""One-time script used to generate base global data for the model."""
import os
import logging
import multiprocessing
import subprocess

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
BASE_URI = 'gs://ecoshard-root/global_carbon_regression/inputs'

BACCINI_10s_2014_BIOMASS_URI = (
    'gs://ecoshard-root/global_carbon_regression/'
    'baccini_10s_2014_md5_5956a9d06d4dffc89517cefb0f6bb008.tif')

ESA_LULC_URI = (
    'gs://ecoshard-root/global_carbon_regression/'
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif')

CARBON_EDGE_MODEL_DATA_NODATA = [
    ('accessibility_to_cities_2015_30sec_compressed_md5_c8b0cede8a8f6b0f004c8b97586ea61a.tif', -9999, None),
    ('altitude_10sec_compressed_md5_5f2c8b4e26ec969819134109181c3744.tif', None, None),
    ('bio_01_30sec_compressed_md5_6f0ba86674e14d3e2a11d9f66282df51.tif', -9999, None),
    ('bio_02_30sec_compressed_md5_4a7139ff1bcde6a384cc3824d93e3aeb.tif', -9999, None),
    ('bio_03_30sec_compressed_md5_b0d5cd27de607125451efa648cae58a7.tif', -9999, None),
    ('bio_04_30sec_compressed_md5_9cbe6c4a4c22ae3fda829a68ebb5c3ab.tif', -9999, None),
    ('bio_05_30sec_compressed_md5_f3e26f183e4add02cac1c984775618c3.tif', -9999, None),
    ('bio_06_30sec_compressed_md5_0b2ab91b48920df38ca4455b46313797.tif', -9999, None),
    ('bio_07_30sec_compressed_md5_6e3b749fb1ae93d7283a73b24a76e02c.tif', -9999, None),
    ('bio_08_30sec_compressed_md5_dacfe3568b4510d371b0dd4b719400a0.tif', -9999, None),
    ('bio_09_30sec_compressed_md5_40120ac7b65703b6f93eec2ba771be99.tif', -9999, None),
    ('bio_10_30sec_compressed_md5_cfc0444c884753ae9c01237dcdbddf67.tif', -9999, None),
    ('bio_11_30sec_compressed_md5_92f12d876ee52439fe403096764c7519.tif', -9999, None),
    ('bio_12_30sec_compressed_md5_1466feb920dd5defcbbe7afa6a713966.tif', -9999, None),
    ('bio_13_30sec_compressed_md5_c25eba18f88adb7576e4221293b79d46.tif', -9999, None),
    ('bio_14_30sec_compressed_md5_81716ca53ef4308c06d9334cbdd34fc2.tif', -9999, None),
    ('bio_15_30sec_compressed_md5_eff27479e3a40a134dc794c0f755ce85.tif', -9999, None),
    ('bio_16_30sec_compressed_md5_5617089f98f296129e27223c203778aa.tif', -9999, None),
    ('bio_17_30sec_compressed_md5_152fe6a9be238c8e125dbb304f1406fe.tif', -9999, None),
    ('bio_18_30sec_compressed_md5_13910428a50f5a3a2a0c49d4e35f68ff.tif', -9999, None),
    ('bio_19_30sec_compressed_md5_34b16f0cc7d11e1c20c1d74280008f76.tif', -9999, None),
    ('hillshade_10sec_compressed_md5_0973aa325db643290320ce8a2afbdf49.tif', 181, None),
    ('livestock_Bf_2010_5min_compressed_md5_9291ed6ebb8fa0784caaf756ff49e6a1.tif', -9999, 0.0),
    ('livestock_Ch_2010_5min_compressed_md5_7b9436a725ae19e78adca5f1a253ef68.tif', -1.7e308, 0.0),
    ('livestock_Ct_2010_5min_compressed_md5_a1baa8737123585a1e2852e4ad27ab3b.tif', -1.7e308, 0.0),
    ('livestock_Dk_2010_5min_compressed_md5_9e5d7ed72481011963d18a4cab6c59e8.tif', -1.7e308, 0.0),
    ('livestock_Gt_2010_5min_compressed_md5_ffb9ff487f044dc3c799e510053f68b7.tif', -1.7e308, 0.0),
    ('livestock_Ho_2010_5min_compressed_md5_b4454fc97c1fc6f9937e4488b8a03482.tif', -1.7e308, 0.0),
    ('night_lights_10sec_compressed_md5_0d66b62beb113326848a49ebab369105.tif', None, None),
    ('night_lights_5min_compressed_md5_f69d0392bd9cd5537417f3656dfa126d.tif', None, None),
    ('nitrogen_10sec_compressed_md5_6994f7cbb2974ab2c6b07f0941e2d2ad.tif', -9999, None),
    ('slope_10sec_compressed_md5_939da641aaa7f72bdd143c64a81cbad6.tif', None, None),
    ('tri_10sec_compressed_md5_021caeb308060476b216c1bba57514d7.tif', 0, None),
    ('wind_speed_10sec_compressed_md5_ec54562e1a6d307e532b767989f48a13.tif', -999, None),
    ('population_2015_30sec_compressed_md5_676c2ff75cebe0a4fcd090dfecc7a037.tif', 3.4028235e+38, 0.0),
    ('population_2015_5min_compressed_md5_4c267e3cb681689acc08020fd64e023d.tif', 3.4028235e+38, 0.0),
    ('bdod_0-5cm_uncertainty_compressed_md5_37f9af4d5cb1babc47642f90e3fbaf9a.tif', 65535, None),
    ('bdod_5-15cm_mean_compressed_md5_eabee41137f65c5400c16012fd31c686.tif', -32768, None),
    ('bdod_5-15cm_uncertainty_compressed_md5_a3ae7668759e85482b4a3930b1bc4818.tif', 65535, None),
    ('cec_0-5cm_mean_compressed_md5_fcb258ec64c03d494f6f37811e1953e7.tif', -32768, None),
    ('cec_0-5cm_uncertainty_compressed_md5_da49cc29b7e92932636bef2fcb59f2bc.tif', 65535, None),
    ('cec_5-15cm_mean_compressed_md5_2237766c8236006be2ae6b533c18ce1b.tif', -32768, None),
    ('cec_5-15cm_uncertainty_compressed_md5_d5bbaf58ccce257fa9b6c848ebeb1438.tif', 65535, None),
    ('cfvo_0-5cm_mean_compressed_md5_559e5694539eebc1c1812d097f51f264.tif', -32768, None),
    ('cfvo_0-5cm_uncertainty_compressed_md5_3ceda87f2ff831a5bca7dcddbba8a0ec.tif', 65535, None),
    ('cfvo_5-15cm_mean_compressed_md5_2d0ca616540fac16f337111d161044c7.tif', -32768, None),
    ('cfvo_5-15cm_uncertainty_compressed_md5_c4708eda6aae30dbda28957419a4aeef.tif', 65535, None),
    ('clay_0-5cm_mean_compressed_md5_8811e315c128b13d19b91eedd38f3289.tif', -32768, None),
    ('clay_0-5cm_uncertainty_compressed_md5_d2d5413cad4be779f67633f055ec7edd.tif', 65535, None),
    ('clay_5-15cm_mean_compressed_md5_f3034943f4c27c2e34cd8b1c3c9eae12.tif', -32768, None),
    ('clay_5-15cm_uncertainty_compressed_md5_2bc6390a2f1be9a148364daf1d74cd1e.tif', 65535, None),
    ('nitrogen_0-5cm_mean_compressed_md5_982afcaa250504dda1c74a9305bd4dfb.tif', -32768, None),
    ('nitrogen_0-5cm_uncertainty_compressed_md5_0a764e8e11c095a6198cbe0d57ff17e9.tif', 65535, None),
    ('nitrogen_5-15cm_mean_compressed_md5_40cc7a4f8dc6e3f20477b26e61a7fe14.tif', -32768, None),
    ('nitrogen_5-15cm_uncertainty_compressed_md5_0a20d836d1d3e10ee369e100b864267b.tif', 65535, None),
    ('phh2o_0-5cm_mean_compressed_md5_cf6d71bd6fb983f0b95da9a4a42daa5f.tif', -32768, None),
    ('phh2o_0-5cm_uncertainty_compressed_md5_53cbff25993947a31426ad9eeeac89a8.tif', 65535, None),
    ('phh2o_5-15cm_mean_compressed_md5_a4791fe139d07654cd51539a3ef4cee0.tif', -32768, None),
    ('phh2o_5-15cm_uncertainty_compressed_md5_b44c67d80604a9cb678a907961f13b6b.tif', 65535, None),
    ('sand_0-5cm_mean_compressed_md5_39ce08e191c75f63fd0b876d41b54688.tif', -32768, None),
    ('sand_0-5cm_uncertainty_compressed_md5_e7034724303f6dce155ad29ad450102e.tif', 65535, None),
    ('sand_5-15cm_uncertainty_compressed_md5_4ecbda7ddf5504e5cc6a86ab3986239d.tif', 65535, None),
    ('silt_0-5cm_mean_compressed_md5_2d035d45e08cc5a442bbbffcc1ebc0c7.tif', -32768, None),
    ('silt_0-5cm_uncertainty_compressed_md5_bd64b35062130e80bdf76ebde9f91f58.tif', 65535, None),
    ('silt_5-15cm_mean_compressed_md5_8393319c8345da12c664d39b67a0afa4.tif', -32768, None),
    ('silt_5-15cm_uncertainty_compressed_md5_8277d8b498e593cb352be7bf43592be6.tif', 65535, None),
    ('soc_0-5cm_mean_compressed_md5_076f6fa676ab399577e5881d4aa3784e.tif', -32768, None),
    ('soc_0-5cm_uncertainty_compressed_md5_e7f40d8b08e2ad9b128e916b93b346b5.tif', -32768, None),
    ('soc_5-15cm_mean_compressed_md5_6bf0892a2bbe8ee00f84fafd84302f29.tif', -32768, None),
    ('soc_5-15cm_uncertainty_compressed_md5_153b60fa1115c7e99210a6d81cd70fa5.tif', -32768, None),
    # These were in the 1.0 version
    # ('AWCh1_10sec.tif', 255, None),
    # ('AWCh2_10sec.tif', 255, None),
    # ('AWCh3_10sec.tif', 255, None),
    # ('AWCtS_10sec.tif', 255, None),
    # ('bdod_10sec.tif', 0, None),  # use new
    # ('BDRICM_10sec.tif', 255, None),
    # ('BDTICM_10sec.tif', 9999, None),
    # ('BLDFIE_10sec.tif', 9999, None),
    # ('cfvo_10sec.tif', -9999, None),
    # ('clay_10sec.tif', -9999, None),
    # ('CLYPPT_10sec.tif', 255, None),
    # ('CRFVOL_10sec.tif', 255, None),
    # ('HISTPR_10sec.tif', 255, None),
    # ('ndvcec015_10sec.tif', 0, None),
    # ('ocd_10sec.tif', -9999, None),
    # ('OCDENS_10sec.tif', 9999, None),
    # ('ocs_10sec.tif', -9999, None),
    # ('OCSTHA_10sec.tif', 9999, None),
    # ('phh2o_10sec.tif', -9999, None),
    # ('PHIHOX_10sec.tif', 255, None),
    # ('PHIKCL_10sec.tif', 255, None),
    # ('sand_10sec.tif', -9999, None),
    # ('silt_10sec.tif', -9999, None),
    # ('soc_10sec.tif', -9999, None),
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


def create_aligned_base_data(
        alignment_raster_path, target_data_dir,
        n_workers=multiprocessing.cpu_count()):
    """Create aligned base data.

    Create the base data that are aligned to the given raster path.

    Args:
        alignment_raster_path (str): base raster to align inputs to.
        target_data_dir (str): path to directory to dump aligned rasters.
        n_workers (str): how many warps to allow to run in parallel.

    Returns:
        None
    """
    LOGGER.info(
        f"align data to {alignment_raster_path}, place in {target_data_dir}")
    task_graph = taskgraph.TaskGraph(
        target_data_dir, n_workers, 15.0)
    # Expected data is given by `carbon_model_data`.
    base_raster_data_path_list = [
        os.path.join(BASE_DATA_DIR, filename)
        for filename, _, _ in CARBON_EDGE_MODEL_DATA_NODATA]

    # sanity check:
    missing_raster_list = []
    for path in base_raster_data_path_list:
        if not os.path.exists(path):
            missing_raster_list.append(path)
    if missing_raster_list:
        raise ValueError(
            f'Expected the following files that did not exist: '
            f'{missing_raster_list}')

    alignment_raster_info = pygeoprocessing.get_raster_info(
        alignment_raster_path)
    aligned_raster_path_list = [
        os.path.join(target_data_dir, os.path.basename(path))
        for path in base_raster_data_path_list]
    for base_raster_path, target_aligned_raster_path in zip(
            base_raster_data_path_list, aligned_raster_path_list):
        if same_coverage(base_raster_path, alignment_raster_path):
            LOGGER.info(
                f'{base_raster_path} and {alignment_raster_path} are aligned '
                f'already, hardlinking to {target_aligned_raster_path}')
            if os.path.exists(target_aligned_raster_path):
                os.remove(target_aligned_raster_path)
            os.link(base_raster_path, target_aligned_raster_path)
            continue

        _ = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(
                base_raster_path, alignment_raster_info['pixel_size'],
                target_aligned_raster_path, 'near'),
            kwargs={
                'target_bb': alignment_raster_info['bounding_box'],
                'target_projection_wkt': (
                    alignment_raster_info['projection_wkt']),
                'working_dir': target_data_dir,
                },
            target_path_list=[target_aligned_raster_path],
            task_name=f'align {base_raster_path} data')
    LOGGER.info('wait for data to align')
    task_graph.join()
    task_graph.close()
    task_graph = None


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


def fetch_data(target_data_dir, task_graph):
    """Download all the global data needed to run this analysis.

    Args:
        target_data_dir (str): path to directory to copy clipped rasters
            to
        task_graph (TaskGraph): taskgraph object to schedule work.

    Returns:
        List of (file_path, nodata, nodata_replacement) tuples.

    """
    try:
        os.makedirs(target_data_dir)
    except OSError:
        pass

    downloaded_file_list = []
    for file_id, nodata, nodata_replacement in \
            CARBON_EDGE_MODEL_DATA_NODATA + [
                (BACCINI_10s_2014_BIOMASS_URI, None, None),
                (ESA_LULC_URI, None, None)]:
        if file_id.startswith('gs://'):
            file_uri = file_id
        else:
            file_uri = os.path.join(BASE_URI, file_id)
        target_file_path = os.path.join(
            target_data_dir, os.path.basename(file_uri))
        LOGGER.info(
            f'download_gs for {file_uri}, exists? '
            f'{os.path.exists(target_file_path)}')
        _ = task_graph.add_task(
            func=download_gs,
            args=(file_uri, target_file_path),
            kwargs={'skip_if_target_exists': True},
            target_path_list=[target_file_path],
            task_name=f'download {file_uri} to {target_data_dir}')
        if file_uri not in [BACCINI_10s_2014_BIOMASS_URI, ESA_LULC_URI]:
            downloaded_file_list.append(
                (target_file_path, nodata, nodata_replacement))

    LOGGER.info('waiting for downloads to complete')
    task_graph.join()
    LOGGER.info('returning downloaded result')
    return downloaded_file_list
