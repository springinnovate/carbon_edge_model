"""Script to download everything needed to train the models."""
import argparse
import os
import collections
import multiprocessing
import pickle
import time
import logging

from osgeo import gdal
from osgeo import osr
from utils import esa_to_carbon_model_landcover_types
import ecoshard
import pygeoprocessing
import numpy
import scipy
import taskgraph
import torch

gdal.SetCacheMax(2**27)
torch.set_num_threads(multiprocessing.cpu_count())
torch.set_num_interop_threads(multiprocessing.cpu_count())
logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.ERROR)
LOGGER = logging.getLogger(__name__)

BOUNDING_BOX = [-64, -4, -55, 3]

WORKSPACE_DIR = f"workspace{'_'.join([str(v) for v in BOUNDING_BOX])}/ecoshards"
ALIGN_DIR = os.path.join(WORKSPACE_DIR, 'align')
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
for dir_path in [WORKSPACE_DIR, ALIGN_DIR, CHURN_DIR]:
    os.makedirs(dir_path, exist_ok=True)
MODEL_PATH = os.path.join(WORKSPACE_DIR, 'model.dat')
RASTER_LOOKUP_PATH = os.path.join(WORKSPACE_DIR, 'raster_lookup.dat')

URL_PREFIX = (
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression_2/'
    'inputs/')

RESPONSE_RASTER_FILENAME = 'baccini_carbon_data_2003_2014_compressed_md5_11d1455ee8f091bf4be12c4f7ff9451b.tif'

MASK_TYPES = [
    ('cropland', esa_to_carbon_model_landcover_types.CROPLAND_LULC_CODES),
    ('urban', esa_to_carbon_model_landcover_types.URBAN_LULC_CODES),
    ('forest', esa_to_carbon_model_landcover_types.FOREST_CODES)]

EXPECTED_MAX_EDGE_EFFECT_KM_LIST = [1.0, 3.0, 10.0]

CELL_SIZE = (0.004, -0.004)  # in degrees
PROJECTION_WKT = osr.SRS_WKT_WGS84_LAT_LONG
SAMPLE_RATE = 0.001

MAX_TIME_INDEX = 11

TIME_PREDICTOR_LIST = [
    #('baccini_carbon_error_compressed_wgs84__md5_77ea391e63c137b80727a00e4945642f.tif', None),
]

LULC_TIME_LIST = [
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2003-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2004-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2005-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2006-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2007-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2008-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2009-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2010-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2011-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2012-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2013-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif', None)]

PREDICTOR_LIST = [
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
]


def download_data(task_graph):
    """Download the whole data stack."""
    # First download the response raster to align all the rest
    response_url = URL_PREFIX + RESPONSE_RASTER_FILENAME
    response_path = os.path.join(WORKSPACE_DIR, RESPONSE_RASTER_FILENAME)
    LOGGER.debug(f'download {response_url} to {response_path}')
    download_task = task_graph.add_task(
        func=ecoshard.download_url,
        args=(response_url, response_path),
        target_path_list=[response_path],
        task_name=f'download {response_path}')
    aligned_path = os.path.join(ALIGN_DIR, RESPONSE_RASTER_FILENAME)
    align_task = task_graph.add_task(
        func=pygeoprocessing.warp_raster,
        args=(response_path, CELL_SIZE, aligned_path, 'near'),
        kwargs={
            'target_bb': BOUNDING_BOX,
            'target_projection_wkt': PROJECTION_WKT,
            'working_dir': WORKSPACE_DIR},
        dependent_task_list=[download_task],
        target_path_list=[aligned_path],
        task_name=f'align {aligned_path}')
    raster_lookup = collections.defaultdict(list)
    raster_lookup['response'] = aligned_path

    # download the rest and align to response
    download_project_list = []
    for raster_list, raster_type in [
            (LULC_TIME_LIST, 'lulc_time_list'),
            (TIME_PREDICTOR_LIST, 'time_predictor'),
            (PREDICTOR_LIST, 'predictor')]:
        for payload in raster_list:
            if isinstance(payload, list):
                # list of timesteps, keep the list structure
                raster_lookup[raster_type].append([])
                for filename, nodata in payload:
                    aligned_path = os.path.join(ALIGN_DIR, filename)
                    raster_lookup[raster_type][-1].append(
                        (aligned_path, nodata))
                    download_project_list.append(filename)
            elif isinstance(payload, tuple):
                # it's a path/nodata tuple
                filename, nodata = payload
                aligned_path = os.path.join(ALIGN_DIR, filename)
                download_project_list.append(filename)
                raster_lookup[raster_type].append((aligned_path, nodata))

        for filename in download_project_list:
            url = URL_PREFIX + filename
            ecoshard_path = os.path.join(WORKSPACE_DIR, filename)
            download_task = task_graph.add_task(
                func=ecoshard.download_url,
                args=(url, ecoshard_path),
                target_path_list=[ecoshard_path],
                task_name=f'download {ecoshard_path}')
            aligned_path = os.path.join(ALIGN_DIR, filename)
            _ = task_graph.add_task(
                func=pygeoprocessing.warp_raster,
                args=(ecoshard_path, CELL_SIZE, aligned_path, 'near'),
                kwargs={
                    'target_bb': BOUNDING_BOX,
                    'target_projection_wkt': PROJECTION_WKT,
                    'working_dir': WORKSPACE_DIR},
                dependent_task_list=[download_task],
                target_path_list=[aligned_path],
                task_name=f'align {aligned_path}')

    return raster_lookup


def sample_data(time_domain_mask_list, predictor_lookup):
    """Sample data stack.

    All input rasters are aligned.

    Args:
        response_path (str): path to response raster
        predictor_lookup (dict): dictionary with keys 'predictor' and
            'time_predictor'. 'time_predictor' are either a tuple of rasters
            or a single multiband raster with indexes that conform to the
            bands in ``response_path``.


    """
    predictor_band_nodata_list = []
    raster_list = []
    # simple lookup to map predictor band/nodata to a list
    for predictor_path, nodata in predictor_lookup['predictor']:
        predictor_raster = gdal.OpenEx(predictor_path, gdal.OF_RASTER)
        raster_list.append(predictor_raster)
        predictor_band = predictor_raster.GetRasterBand(1)

        if nodata is None:
            nodata = predictor_band.GetNoDataValue()
        predictor_band_nodata_list.append((predictor_band, nodata))

    # create a dictionary that maps time index to list of predictor band/nodata
    # values, will be used to create a stack of data with the previous
    # collection and one per timestep on this one
    time_predictor_lookup = collections.defaultdict(list)
    for payload in predictor_lookup['time_predictor']:
        # time predictors could either be a tuple of rasters or a single
        # raster with multiple bands
        if isinstance(payload, tuple):
            time_predictor_path, nodata = payload
            time_predictor_raster = gdal.OpenEx(
                time_predictor_path, gdal.OF_RASTER)
            raster_list.append(time_predictor_raster)
            for index in range(time_predictor_raster.RasterCount):
                time_predictor_band = time_predictor_raster.GetRasterBand(
                    index+1)
                if nodata is None:
                    nodata = time_predictor_band.GetNoDataValue()
                time_predictor_lookup[index].append(
                    (time_predictor_band, nodata))
        elif isinstance(payload, list):
            for index, (time_predictor_path, nodata) in enumerate(
                    payload):
                time_predictor_raster = gdal.OpenEx(
                    time_predictor_path, gdal.OF_RASTER)
                raster_list.append(time_predictor_raster)
                time_predictor_band = time_predictor_raster.GetRasterBand(1)
                if nodata is None:
                    nodata = time_predictor_band.GetNoDataValue()
                time_predictor_lookup[index].append((
                    time_predictor_band, nodata))
        else:
            raise ValueError(
                f'expected str or tuple but got {payload}')
    mask_band_list = []
    for time_domain_mask_raster_path in time_domain_mask_list:
        mask_raster = gdal.OpenEx(time_domain_mask_raster_path, gdal.OF_RASTER)
        raster_list.append(mask_raster)
        mask_band = mask_raster.GetRasterBand(1)
        mask_band_list.append(mask_band)

    # build up an array of predictor stack
    response_raster = gdal.OpenEx(predictor_lookup['response'], gdal.OF_RASTER)
    raster_list.append(response_raster)
    # if response_raster.RasterCount != len(time_predictor_lookup):
    #     raise ValueError(
    #         f'expected {response_raster.RasterCount} time elements but only '
    #         f'got {len(time_predictor_lookup)}')

    y_list = []
    x_vector = None
    i = 0
    last_time = time.time()
    total_pixels = predictor_raster.RasterXSize * predictor_raster.RasterYSize
    for offset_dict in pygeoprocessing.iterblocks(
            (predictor_lookup['response'], 1),
            offset_only=True, largest_block=2**20):
        LOGGER.debug(f"{offset_dict['win_xsize']} {offset_dict['win_ysize']}")
        if time.time() - last_time > 5.0:
            n_pixels_processed = offset_dict['xoff']+offset_dict['yoff']*predictor_raster.RasterXSize
            LOGGER.info(f"processed {100*n_pixels_processed/total_pixels:.3f}% so far ({n_pixels_processed}) (x/y {offset_dict['xoff']}/{offset_dict['yoff']}) y_list size {len(y_list)}")
            last_time = time.time()
        predictor_stack = []  # N elements long
        valid_array = numpy.ones(
            (offset_dict['win_ysize'], offset_dict['win_xsize']),
            dtype=bool)
        # load all the regular predictors
        for predictor_band, predictor_nodata in predictor_band_nodata_list:
            predictor_array = predictor_band.ReadAsArray(**offset_dict)
            if predictor_nodata is not None:
                valid_array &= predictor_array != predictor_nodata
            predictor_stack.append(predictor_array)

        if not numpy.any(valid_array):
            continue

        # load the time based predictors
        for index, time_predictor_band_nodata_list in \
                time_predictor_lookup.items():
            if index > MAX_TIME_INDEX:
                break
            mask_array = mask_band_list[index].ReadAsArray(**offset_dict)
            valid_time_array = valid_array & (mask_array == 1)
            predictor_time_stack = []
            predictor_time_stack.extend(predictor_stack)
            for predictor_band, predictor_nodata in \
                    time_predictor_band_nodata_list:
                predictor_array = predictor_band.ReadAsArray(**offset_dict)
                if predictor_nodata is not None:
                    valid_time_array &= predictor_array != predictor_nodata
                predictor_time_stack.append(predictor_array)

            # load the time based responses
            response_band = response_raster.GetRasterBand(index+1)
            response_nodata = response_band.GetNoDataValue()
            response_array = response_band.ReadAsArray(**offset_dict)
            if response_nodata is not None:
                valid_time_array &= response_array != response_nodata

            if not numpy.any(valid_time_array):
                break

            sample_mask = numpy.random.rand(
                numpy.count_nonzero(valid_time_array)) < SAMPLE_RATE

            # all of response_time_stack and response_array are valid, clip and add to set
            local_x_list = []
            # each element in array should correspond with an element in y
            for array in predictor_time_stack:
                local_x_list.append((array[valid_time_array])[sample_mask])
            if x_vector is None:
                x_vector = numpy.array(local_x_list)
            else:
                local_x_vector = numpy.array(local_x_list)
                x_vector = numpy.append(x_vector, local_x_vector, axis=1)
            y_list.extend(
                list((response_array[valid_time_array])[sample_mask]))

        i += 1
    y_vector = numpy.array(y_list)
    LOGGER.debug(f'got all done {x_vector.shape} {y_vector.shape}')
    return (x_vector.T).astype(numpy.float32), (y_vector.astype(numpy.float32))


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


def _create_lulc_mask(lulc_raster_path, mask_codes, target_mask_raster_path):
    """Create a mask raster given an lulc and mask code."""
    numpy_mask_codes = numpy.array(mask_codes)
    pygeoprocessing.raster_calculator(
        [(lulc_raster_path, 1)],
        lambda array: numpy.isin(array, numpy_mask_codes),
        target_mask_raster_path, gdal.GDT_Byte, None)


def mask_lulc(task_graph, lulc_raster_path):
    """Create all the masks and convolutions off of lulc_raster_path."""
    # this is calculated as 111km per degree
    convolution_raster_list = []
    for expected_max_edge_effect_km in EXPECTED_MAX_EDGE_EFFECT_KM_LIST:
        pixel_radius = (CELL_SIZE[0] * 111 / expected_max_edge_effect_km)**-1
        kernel_raster_path = os.path.join(
            CHURN_DIR, f'kernel_{pixel_radius}.tif')
        kernel_task = task_graph.add_task(
            func=make_kernel_raster,
            args=(pixel_radius, kernel_raster_path),
            target_path_list=[kernel_raster_path],
            task_name=f'make kernel of radius {pixel_radius}')

        for mask_id, mask_codes in MASK_TYPES:
            mask_raster_path = os.path.join(
                CHURN_DIR, f'{os.path.basename(os.path.splitext(lulc_raster_path)[0])}_{mask_id}_mask.tif')
            create_mask_task = task_graph.add_task(
                func=_create_lulc_mask,
                args=(lulc_raster_path, mask_codes, mask_raster_path),
                target_path_list=[mask_raster_path],
                task_name=f'create {mask_id} mask')
            if mask_id == 'forest':
                forest_mask_raster_path = mask_raster_path
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
            convolution_raster_list.append(((mask_gf_path, None)))
    task_graph.join()
    LOGGER.debug(f'all done convolution list - {convolution_raster_list}')
    return forest_mask_raster_path, convolution_raster_list


def train(x_vector, y_vector, target_model_path):
    LOGGER.debug(f'{x_vector.shape} {y_vector.shape}')
    # Use the nn package to define our model and loss function.
    N = 300
    model = torch.nn.Sequential(
        torch.nn.Linear(x_vector.shape[1], N),
        torch.nn.Sigmoid(),
        torch.nn.Linear(N, N),
        torch.nn.Sigmoid(),
        torch.nn.Linear(N, N),
        torch.nn.Sigmoid(),
        torch.nn.Linear(N, N),
        torch.nn.Sigmoid(),
        torch.nn.Linear(N, 1),
        torch.nn.Flatten(0, 1)
    )
    loss_fn = torch.nn.MSELoss(reduction='mean')

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use RMSprop; the optim package contains many other
    # optimization algorithms. The first argument to the RMSprop constructor tells the
    # optimizer which Tensors it should update.
    learning_rate = 1e-3
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    last_loss = None

    iter_count = 0
    while True:
        iter_count += 1
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x_vector)

        # Compute and print loss.
        loss = loss_fn(y_pred, y_vector)
        if iter_count % 300 == 0:
            if last_loss is not None:
                if loss.item() - last_loss > 0:
                    learning_rate *= 0.95
                else:
                    learning_rate *= 1.05
                total_loss = last_loss-loss.item()
                loss_rate = (total_loss)/last_loss
                if (total_loss < 10 and loss_rate > 0) or iter_count > 4000:
                    break
                print(iter_count, loss.item(), total_loss, loss_rate)
            last_loss = loss.item()

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
    torch.save(model, target_model_path)


def align_predictors(
        task_graph, lulc_raster_base_path, predictor_list, workspace_dir):
    """Align all the predictors to lulc."""
    lulc_raster_info = pygeoprocessing.get_raster_info(lulc_raster_base_path)
    aligned_dir = os.path.join(workspace_dir, 'aligned')
    os.makedirs(aligned_dir, exist_ok=True)
    aligned_predictor_list = []
    for predictor_raster_path, nodata in predictor_list:
        if nodata is None:
            nodata = pygeoprocessing.get_raster_info(
                predictor_raster_path)['nodata'][0]
        aligned_predictor_raster_path = os.path.join(
            aligned_dir, os.path.basename(predictor_raster_path))
        task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(
                predictor_raster_path, lulc_raster_info['pixel_size'],
                aligned_predictor_raster_path, 'near'),
            kwargs={
                'target_bb': lulc_raster_info['bounding_box'],
                'target_projection_wkt': lulc_raster_info['projection_wkt']},
            target_path_list=[aligned_predictor_raster_path],
            task_name=f'align {aligned_predictor_raster_path}')
        aligned_predictor_list.append((aligned_predictor_raster_path, nodata))
    return aligned_predictor_list


def model_predict(
            model, lulc_raster_path, forest_mask_raster_path,
            aligned_predictor_list, predicted_biomass_raster_path):
    """Predict biomass given predictors."""
    pygeoprocessing.new_raster_from_base(
        lulc_raster_path, predicted_biomass_raster_path, gdal.GDT_Float32,
        [-1])
    predicted_biomass_raster = gdal.OpenEx(
        predicted_biomass_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    predicted_biomass_band = predicted_biomass_raster.GetRasterBand(1)

    predictor_band_nodata_list = []
    raster_list = []
    # simple lookup to map predictor band/nodata to a list
    for predictor_path, nodata in aligned_predictor_list:
        predictor_raster = gdal.OpenEx(predictor_path, gdal.OF_RASTER)
        raster_list.append(predictor_raster)
        predictor_band = predictor_raster.GetRasterBand(1)

        if nodata is None:
            nodata = predictor_band.GetNoDataValue()
        predictor_band_nodata_list.append((predictor_band, nodata))
    forest_raster = gdal.OpenEx(forest_mask_raster_path, gdal.OF_RASTER)
    forest_band = forest_raster.GetRasterBand(1)

    for offset_dict in pygeoprocessing.iterblocks(
            (lulc_raster_path, 1), offset_only=True):
        forest_array = forest_band.ReadAsArray(**offset_dict)
        valid_mask = (forest_array == 1)
        x_vector = None
        array_list = []
        for band, nodata in predictor_band_nodata_list:
            array = band.ReadAsArray(**offset_dict)
            if nodata is None:
                nodata = band.GetNoDataValue()
            if nodata is not None:
                valid_mask &= array != nodata
            array_list.append(array)
        if not numpy.any(valid_mask):
            continue
        for array in array_list:
            if x_vector is None:
                x_vector = array[valid_mask].astype(numpy.float32)
                x_vector = numpy.reshape(x_vector, (-1, x_vector.size))
            else:
                valid_array = array[valid_mask].astype(numpy.float32)
                valid_array = numpy.reshape(valid_array, (-1, valid_array.size))
                x_vector = numpy.append(x_vector, valid_array, axis=0)
        y_vector = model(torch.from_numpy(x_vector.T))
        result = numpy.full(forest_array.shape, -1)
        result[valid_mask] = y_vector.detach().numpy()
        predicted_biomass_band.WriteArray(
            result,
            xoff=offset_dict['xoff'],
            yoff=offset_dict['yoff'])
    predicted_biomass_band = None
    predicted_biomass_raster = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='People Travel Coverage')
    parser.add_argument('lulc_raster_input', help='Path to lulc raster to model')
    args = parser.parse_args()

    task_graph = taskgraph.TaskGraph('.', multiprocessing.cpu_count(), 15.0)
    if not os.path.exists(MODEL_PATH) or not os.path.exists(RASTER_LOOKUP_PATH):
        raster_lookup = download_data(task_graph)
        task_graph.join()
        # raster lookup has 'predictor' and 'time_predictor' lists
        LOGGER.debug('running sample data')

        time_domain_convolution_raster_list = []
        forest_mask_raster_path_list = []
        for lulc_path, _ in raster_lookup['lulc_time_list']:
            forest_mask_raster_path, convolution_raster_list = mask_lulc(
                task_graph, lulc_path)
            time_domain_convolution_raster_list.append(convolution_raster_list)
            forest_mask_raster_path_list.append(forest_mask_raster_path)
            # convolution_raster_list is all the convolutions for a given timestep
        for time_domain_list in zip(*time_domain_convolution_raster_list):
            raster_lookup['time_predictor'].append(list(time_domain_list))
        task_graph.join()
        sample_data_task = task_graph.add_task(
            func=sample_data,
            args=(forest_mask_raster_path_list, raster_lookup),
            store_result=True,
            task_name='sample data')
        x_vector, y_vector = sample_data_task.get()
        task_graph.add_task(
            func=train,
            args=(
                torch.from_numpy(x_vector), torch.from_numpy(y_vector),
                MODEL_PATH),
            target_path_list=[MODEL_PATH],
            task_name='train')
        with open(RASTER_LOOKUP_PATH, 'wb') as raster_lookup_file:
            pickle.dump(raster_lookup, raster_lookup_file)

    with open(RASTER_LOOKUP_PATH, 'rb') as raster_lookup_file:
        raster_lookup = pickle.load(raster_lookup_file)
    local_workspace = os.path.join(
        WORKSPACE_DIR,
        os.path.basename(os.path.splitext(args.lulc_raster_input)[0]))

    local_info = pygeoprocessing.get_raster_info(args.lulc_raster_input)
    aligned_predictor_list = align_predictors(
        task_graph, args.lulc_raster_input, raster_lookup['predictor'],
        local_workspace)
    forest_mask_raster_path, convolution_raster_list = mask_lulc(
        task_graph, args.lulc_raster_input)
    model = torch.load(MODEL_PATH)
    model.eval()
    predicted_biomass_raster_path = os.path.join(
        local_workspace,
        f'modeled_biomass_{os.path.basename(args.lulc_raster_input)}')
    model_predict(
        model, args.lulc_raster_input, forest_mask_raster_path,
        aligned_predictor_list+convolution_raster_list,
        predicted_biomass_raster_path)
    LOGGER.debug('all done')
    task_graph.close()
# keep it special
