"""Script to download everything needed to train the models."""
from datetime import datetime
import itertools
import argparse
import os
import collections
import multiprocessing
import pickle
import time
import logging
import threading

from osgeo import gdal
from osgeo import osr
from osgeo import ogr
from utils import esa_to_carbon_model_landcover_types
import ecoshard
import pygeoprocessing
import numpy
import scipy
import taskgraph
import torch
from torch import nn
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.metrics import r2_score
torch.autograd.set_detect_anomaly(True)

gdal.SetCacheMax(2**27)
torch.set_num_threads(multiprocessing.cpu_count())
torch.set_num_interop_threads(multiprocessing.cpu_count())
logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)

#BOUNDING_BOX = [-64, -4, -55, 3]
BOUNDING_BOX = [-179, -60, 179, 60]

WORKSPACE_DIR = 'workspace'
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
ALIGN_DIR = os.path.join(WORKSPACE_DIR, f'align{"_".join([str(v) for v in BOUNDING_BOX])}')
CHURN_DIR = os.path.join(WORKSPACE_DIR, f'churn{"_".join([str(v) for v in BOUNDING_BOX])}')
for dir_path in [WORKSPACE_DIR, ECOSHARD_DIR, ALIGN_DIR, CHURN_DIR]:
    os.makedirs(dir_path, exist_ok=True)
RASTER_LOOKUP_PATH = os.path.join(WORKSPACE_DIR, f'raster_lookup{"_".join([str(v) for v in BOUNDING_BOX])}.dat')

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
SAMPLE_RATE = 0.0001

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
    ('global_fa_compressed_overviews_md5_c4cd993cc28aadbab573d0f0b15cfcd0.tif', None),
]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, M):
        l1 = 100
        #do = .5
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(M, l1),
            torch.nn.ReLU(),
            torch.nn.Linear(l1, l1),
            torch.nn.ReLU(),
            torch.nn.Linear(l1, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def download_data(task_graph, bounding_box):
    """Download the whole data stack."""
    # First download the response raster to align all the rest
    LOGGER.info(f'download data and clip to {bounding_box}')
    response_url = URL_PREFIX + RESPONSE_RASTER_FILENAME
    response_path = os.path.join(ECOSHARD_DIR, RESPONSE_RASTER_FILENAME)
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
            ecoshard_path = os.path.join(ECOSHARD_DIR, filename)
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


def sample_data(
        time_domain_mask_list, predictor_lookup, sample_rate, edge_index,
        sample_point_vector_path):
    """Sample data stack.

    All input rasters are aligned.

    Args:
        response_path (str): path to response raster
        predictor_lookup (dict): dictionary with keys 'predictor' and
            'time_predictor'. 'time_predictor' are either a tuple of rasters
            or a single multiband raster with indexes that conform to the
            bands in ``response_path``.
        edge_index (int): this is the edge raster in the predictor stack
            that should be used to randomly select samples from.


    """
    raster_info = pygeoprocessing.get_raster_info(
        predictor_lookup['predictor'][0][0])
    inv_gt = gdal.InvGeoTransform(raster_info['geotransform'])

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_info['projection_wkt'])
    raster_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    gpkg_driver = gdal.GetDriverByName('GPKG')
    sample_point_vector = gpkg_driver.Create(
        sample_point_vector_path, 0, 0, 0, gdal.GDT_Unknown)
    sample_point_layer = sample_point_vector.CreateLayer(
        'sample_points', raster_srs, ogr.wkbPoint)
    sample_point_layer.StartTransaction()

    LOGGER.info(f'building sample data')
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
            for predictor_index, (predictor_band, predictor_nodata) in \
                    enumerate(time_predictor_band_nodata_list):
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
                numpy.count_nonzero(valid_time_array)) < sample_rate

            X2D, Y2D = numpy.meshgrid(
                range(valid_time_array.shape[1]),
                range(valid_time_array.shape[0]))

            for i, j in zip(
                    (X2D[valid_time_array])[sample_mask],
                    (Y2D[valid_time_array])[sample_mask]):

                sample_point = ogr.Feature(sample_point_layer.GetLayerDefn())
                sample_geom = ogr.Geometry(ogr.wkbPoint)
                x, y = gdal.ApplyGeoTransform(
                    inv_gt, i+0.5+offset_dict['xoff'],
                    j+0.5+offset_dict['yoff'])
                sample_geom.AddPoint(x, y)
                sample_point.SetGeometry(sample_geom)
                sample_point_layer.CreateFeature(sample_point)

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
    sample_point_layer.CommitTransaction()
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


def mask_lulc(task_graph, lulc_raster_path, workspace_dir):
    """Create all the masks and convolutions off of lulc_raster_path."""
    # this is calculated as 111km per degree
    convolution_raster_list = []
    edge_effect_index = None
    current_raster_index = -1

    for expected_max_edge_effect_km in EXPECTED_MAX_EDGE_EFFECT_KM_LIST:
        pixel_radius = (CELL_SIZE[0] * 111 / expected_max_edge_effect_km)**-1
        kernel_raster_path = os.path.join(
            workspace_dir, f'kernel_{pixel_radius}.tif')
        if not os.path.exists(kernel_raster_path):
            kernel_task = task_graph.add_task(
                func=make_kernel_raster,
                args=(pixel_radius, kernel_raster_path),
                target_path_list=[kernel_raster_path],
                task_name=f'make kernel of radius {pixel_radius}')
            kernel_task.join()

    for mask_id, mask_codes in MASK_TYPES:
        mask_raster_path = os.path.join(
            workspace_dir, f'{os.path.basename(os.path.splitext(lulc_raster_path)[0])}_{mask_id}_mask.tif')
        create_mask_task = task_graph.add_task(
            func=_create_lulc_mask,
            args=(lulc_raster_path, mask_codes, mask_raster_path),
            target_path_list=[mask_raster_path],
            task_name=f'create {mask_id} mask')
        if mask_id == 'forest':
            forest_mask_raster_path = mask_raster_path
            if edge_effect_index is None:
                LOGGER.debug(f'CURRENT EDGE INDEX {current_raster_index}')
                edge_effect_index = current_raster_index

        for expected_max_edge_effect_km in EXPECTED_MAX_EDGE_EFFECT_KM_LIST:
            current_raster_index += 1

            pixel_radius = (CELL_SIZE[0] * 111 / expected_max_edge_effect_km)**-1
            kernel_raster_path = os.path.join(
                workspace_dir, f'kernel_{pixel_radius}.tif')
            mask_gf_path = (
                f'{os.path.splitext(mask_raster_path)[0]}_gf_'
                f'{expected_max_edge_effect_km}.tif')
            LOGGER.debug(f'making convoluion for {mask_gf_path}')

            convolution_task = task_graph.add_task(
                func=pygeoprocessing.convolve_2d,
                args=(
                    (mask_raster_path, 1), (kernel_raster_path, 1),
                    mask_gf_path),
                dependent_task_list=[create_mask_task],
                target_path_list=[mask_gf_path],
                task_name=f'create gaussian filter of {mask_id} at {mask_gf_path}')
            convolution_raster_list.append(((mask_gf_path, None)))
    task_graph.join()
    LOGGER.debug(f'all done convolution list - {convolution_raster_list}')
    return forest_mask_raster_path, convolution_raster_list, edge_effect_index


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
        result[valid_mask] = (y_vector.detach().numpy()).flatten()
        predicted_biomass_band.WriteArray(
            result,
            xoff=offset_dict['xoff'],
            yoff=offset_dict['yoff'])
    predicted_biomass_band = None
    predicted_biomass_raster = None


def prep_data(task_graph, raster_lookup_path):
    """Download and convolve global data."""
    raster_lookup = download_data(task_graph, BOUNDING_BOX)
    task_graph.join()
    # raster lookup has 'predictor' and 'time_predictor' lists
    time_domain_convolution_raster_list = []
    forest_mask_raster_path_list = []
    for lulc_path, _ in raster_lookup['lulc_time_list']:
        LOGGER.debug(f'mask {lulc_path}')
        forest_mask_raster_path, convolution_raster_list, edge_effect_index = mask_lulc(
            task_graph, lulc_path, CHURN_DIR)
        time_domain_convolution_raster_list.append(convolution_raster_list)
        forest_mask_raster_path_list.append(forest_mask_raster_path)
    for time_domain_list in zip(*time_domain_convolution_raster_list):
        raster_lookup['time_predictor'].append(list(time_domain_list))
    raster_lookup['edge_effect_index'] = (
        edge_effect_index+len(raster_lookup['predictor']))
    with open(raster_lookup_path, 'wb') as raster_lookup_file:
        pickle.dump(
            (forest_mask_raster_path_list, raster_lookup),
            raster_lookup_file)


def main():
    parser = argparse.ArgumentParser(description='download data')
    parser.add_argument('n_epochs', type=int, help='number of iterations to run trainer')
    parser.add_argument('batch_size', type=int, help='number of iterations to run trainer')
    parser.add_argument('learning_rate', type=float, help='learning rate of initial epoch')
    parser.add_argument('--last_epoch', help='last epoch to pick up at')
    args = parser.parse_args()
    task_graph = taskgraph.TaskGraph('.', multiprocessing.cpu_count(), 5.0)

    if not os.path.exists(RASTER_LOOKUP_PATH):
        LOGGER.info('prep data...')
        prep_data(task_graph, RASTER_LOOKUP_PATH)
        task_graph.join()
    with open(RASTER_LOOKUP_PATH, 'rb') as raster_lookup_file:
        forest_mask_raster_path_list, raster_lookup = pickle.load(
            raster_lookup_file)
    LOGGER.info('sample data...')
    sample_data_task = task_graph.add_task(
        func=sample_data,
        args=(forest_mask_raster_path_list, raster_lookup, SAMPLE_RATE,
              raster_lookup['edge_effect_index'], 'sample.gpkg'),
        store_result=True,
        target_path_list=['sample.gpkg'],
        task_name='sample data')

    task_graph.join()
    task_graph.close()
    task_graph = None

    LOGGER.info('get x/y training vector...')
    x_vector, y_vector = sample_data_task.get()

    y_vector = numpy.expand_dims(y_vector, axis=1)
    x_tensor = torch.from_numpy(x_vector)
    y_tensor = torch.from_numpy(y_vector)

    LOGGER.debug(f'{x_tensor.shape} {y_tensor.shape}')
    ds = torch.utils.data.TensorDataset(x_tensor, y_tensor, )

    n_predictors = x_vector.shape[1]
    loss_fn = torch.nn.L1Loss(reduction='sum')
    train_cifar(
        args.learning_rate, loss_fn, n_predictors, x_vector.shape[0], ds,
        args.n_epochs, args.batch_size, checkpoint_epoch=args.last_epoch)

    LOGGER.debug('all done')


def _sub_op(array_a, array_b, nodata_a, nodata_b):
    result = numpy.full(array_a.shape, nodata_a, dtype=numpy.float32)
    valid_mask = (array_a != nodata_a) & (array_b != nodata_b)
    result[valid_mask] = array_a[valid_mask] - array_b[valid_mask]
    return result


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1-abs(ss_res / ss_tot)
    return r2


def train_cifar(
        learning_rate, loss_fn, n_predictors, n_samples, ds, n_epochs,
        batch_size, checkpoint_epoch=None):
    model = NeuralNetwork(n_predictors)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=learning_rate, momentum=0.9)

    last_epoch = 0
    if checkpoint_epoch:
        model_path = f'model_{checkpoint_epoch}.dat'
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    train_indexes, test_indexes = train_test_split(
        list(range(n_samples)), test_size=.2)
    train_loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indexes))
    val_loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_indexes))
    n_train_samples = len(train_indexes)
    n_test_samples = len(test_indexes)
    print(f'{n_train_samples} samples to train {n_test_samples} samples to validate')

    loss_csv_path = f'{datetime.now().strftime("%H_%M_%S")}.csv'
    with open(loss_csv_path, 'w') as csv_file:
        csv_file.write('epoch,train,val,r2\n')

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, (predictor_t, response_t) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(predictor_t)
            loss = loss_fn(outputs, response_t)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 100 == 0:
                LOGGER.debug(f'[{epoch+1+last_epoch}/{n_epochs+last_epoch}, {i+1}/{n_train_samples/batch_size}]')

        print(f'pre validation max output val: {torch.max(outputs)} min: {torch.min(outputs)}')
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        r2_sum = 0.0
        for i, (predictor_t, response_t) in enumerate(val_loader, 0):
            with torch.no_grad():
                outputs = model(predictor_t)
                val_loss += loss_fn(outputs, response_t).item()
                r2_sum += r2_loss(outputs, response_t)
                r2 = r2_score(outputs, response_t)
                val_steps += 1

        print("[%d] \n training loss: %.3f \n validation loss: %.3f" % (
            epoch + 1 + last_epoch,
            running_loss/n_train_samples, val_loss/n_test_samples))
        r2 = r2_sum / val_steps
        print(f'r^2: {r2}')
        print(f'max output val: {torch.max(outputs)} min: {torch.min(outputs)}')

        with open(loss_csv_path, 'a') as csv_file:
            csv_file.write(f'{epoch+1+last_epoch},{running_loss/n_train_samples},{val_loss/n_test_samples},{r2}\n')

        LOGGER.info('save model')
        model_path = f'model_{epoch+1+last_epoch}.dat'

        torch.save({
            'epoch': epoch+1+last_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_path)

        if r2 > 10:
            LOGGER.debug('r2 too big, quitting')
            break
        LOGGER.info('run again')
    return model_path


if __name__ == '__main__':
    main()
