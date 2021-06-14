"""Script to download everything needed to train the models."""
import os
import collections
import multiprocessing
import time
import logging

from osgeo import gdal
from osgeo import osr
import ecoshard
import pygeoprocessing
import numpy
import rtree
import taskgraph

gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

WORKSPACE_DIR = 'workspace/ecoshards'
os.makedirs(WORKSPACE_DIR, exist_ok=True)
ALIGN_DIR = os.path.join(WORKSPACE_DIR, 'align')
os.makedirs(ALIGN_DIR, exist_ok=True)

URL_PREFIX = 'https://storage.googleapis.com/ecoshard-root/global_carbon_regression_2/inputs/'

RESPONSE_RASTER_FILENAME = 'baccini_carbon_data_2003_2014_compressed_md5_11d1455ee8f091bf4be12c4f7ff9451b.tif'

0.002777777777777777884,-0.002777777777777777884

CELL_SIZE = (0.004, -0.004)  # in degrees
PROJECTION_WKT = osr.SRS_WKT_WGS84_LAT_LONG
BOUNDING_BOX = [-179, -56, 179, 60]

TIME_PREDICTOR_LIST = [
    ('baccini_carbon_error_compressed_wgs84__md5_77ea391e63c137b80727a00e4945642f.tif', None),
    [('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2003-v2.0.7_smooth_compressed.tif', None),
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
     ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif', None)],
]

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


def download_data():
    """Download the whole data stack."""
    task_graph = taskgraph.TaskGraph('.', multiprocessing.cpu_count(), 15.0)

    # First download the response raster to align all the rest
    response_url = URL_PREFIX + RESPONSE_RASTER_FILENAME
    response_path = os.path.join(WORKSPACE_DIR, RESPONSE_RASTER_FILENAME)
    LOGGER.debug(f'download {response_url} to {response_path}')
    response_task = task_graph.add_task(
        func=ecoshard.download_url,
        args=(response_url, response_path),
        target_path_list=[response_path],
        task_name=f'download {response_path}')
    response_task.join()
    LOGGER.info(f'downloaded {response_path}')
    response_info = pygeoprocessing.get_raster_info(response_path)
    raster_lookup = collections.defaultdict(list)

    # download the rest and align to response
    download_project_list = []
    for raster_list, raster_type in [
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
            LOGGER.info(url)
            ecoshard_path = os.path.join(WORKSPACE_DIR, filename)
            download_task = task_graph.add_task(
                func=ecoshard.download_url,
                args=(url, ecoshard_path),
                target_path_list=[ecoshard_path],
                task_name=f'download {ecoshard_path}')
            aligned_path = os.path.join(ALIGN_DIR, filename)
            align_task = task_graph.add_task(
                func=pygeoprocessing.warp_raster,
                args=(ecoshard_path, CELL_SIZE, aligned_path, 'near'),
                kwargs={
                    'target_bb': BOUNDING_BOX,
                    'target_projection_wkt': PROJECTION_WKT,
                    'working_dir': WORKSPACE_DIR},
                dependent_task_list=[download_task],
                target_path_list=[aligned_path],
                task_name=f'align {aligned_path}')
    task_graph.join()
    task_graph.close()

    return raster_lookup


def sample_data(response_path, predictor_lookup):
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
    # simple lookup to map predictor band/nodata to a list
    for predictor_path, nodata in predictor_lookup['predictor']:
        predictor_raster = gdal.OpenEx(predictor_path, gdal.OF_RASTER)
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
                time_predictor_band = time_predictor_raster.GetRasterBand(1)
                if nodata is None:
                    nodata = time_predictor_band.GetNoDataValue()
                time_predictor_lookup[index].append((
                    time_predictor_band, nodata))
        else:
            raise ValueError(
                f'expected str or tuple but got {payload}')

    # build up an array of predictor stack
    response_raster = gdal.OpenEx(response_path, gdal.OF_RASTER)
    if response_raster.RasterCount != len(time_predictor_lookup):
        raise ValueError(
            f'expected {response_raster.RasterCount} time elements but only '
            f'got {len(time_predictor_lookup)}')

    x_vector = []
    y_vector = []

    for offset_dict in pygeoprocessing.iterblocks(
            (response_path, 1), offset_only=True):
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
            LOGGER.info(f'nodata at {offset_dict}')
            continue

        # load the time based predictors
        for index, time_predictor_band_nodata_list in \
                time_predictor_lookup.items():
            valid_time_array = numpy.copy(valid_array)
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
                LOGGER.info(f'nodata at {offset_dict}')
                break

            # all of response_time_stack and response_array are valid, clip and add to set
            local_x_vector = []
            # this is wrong, local_x_vector should be transposed
            for array in predictor_time_stack:
                local_x_vector.append(array[valid_time_array])
            x_vector.push(numpy.array(local_x_vector).T)
            y_vector.extend(list(response_array[valid_time_array]))
            if len(local_x_vector) != len(y_vector):
                raise ValueError('local x vector not same length as y vector')
        break

    LOGGER.debug(f'got all done {len(x_vector)} {len(y_vector)}')


if __name__ == '__main__':
    raster_lookup = download_data()
    sample_data(raster_lookup, 1000)
