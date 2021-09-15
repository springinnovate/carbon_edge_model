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
import glob

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)

logging.getLogger('matplotlib').setLevel(logging.WARN)
logging.getLogger('fiona').setLevel(logging.WARN)


from shapely.prepared import prep
import shapely
import geopandas
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
from utils import esa_to_carbon_model_landcover_types
import ecoshard
from ecoshard import geoprocessing
import numpy
import scipy
import taskgraph
import matplotlib.pyplot as plt

gdal.SetCacheMax(2**27)

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

EXPECTED_MAX_EDGE_EFFECT_KM_LIST = [0.5]  # changed given guidance from reviewer [1.0, 3.0, 10.0]

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


def sample_data(raster_path_list, gdf_points, target_bb_wgs84):
    """Sample raster paths given the points.

    Args:
        raster_path_list (list): path to a set of rasters
        gdf_points (geopandas Frame): points in lat/lng to sample

    Return:
        a geopandas frame with columns defined by the basenames of the
        rasters in ``raster_path_list`` and geometry by ``gdf_points``
        so long as the ``gdf_points`` lies in the bounding box of the rasters.
    """
    LOGGER.debug(f'target_bb_wgs84 {target_bb_wgs84}')

    # sample each raster by its block range so long as its within the
    # bounding box, this is complicated but it saves us from randomly reading
    # all across the raster
    last_time = time.time()
    for raster_path in raster_path_list:
        raster_info = geoprocessing.get_raster_info(raster_path)
        basename = os.path.basename(os.path.splitext(raster_path)[0])
        gdf_points[basename] = raster_info['nodata'][0]
        gt = raster_info['geotransform']
        inv_gt = gdal.InvGeoTransform(gt)
        raster = gdal.OpenEx(raster_path)
        band = raster.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        LOGGER.debug(f'processing {basename}')
        n_total = raster_info['raster_size'][0]*raster_info['raster_size'][1]
        n_processed = 0
        for offset_dict in geoprocessing.iterblocks(
                (raster_path, 1), offset_only=True, largest_block=2**20):
            if time.time()-last_time > 5:
                LOGGER.debug(
                    f'{n_processed/n_total*100:.2f}% complete for {basename} {n_processed} {n_total}')
                last_time = time.time()
            n_processed += offset_dict['win_xsize']*offset_dict['win_ysize']
            local_bb = (
                gdal.ApplyGeoTransform(
                    gt, offset_dict['xoff'], offset_dict['yoff']) +
                gdal.ApplyGeoTransform(
                    gt, offset_dict['xoff']+offset_dict['win_xsize'],
                    offset_dict['yoff']+offset_dict['win_ysize']))

            local_bb_wgs84 = geoprocessing.transform_bounding_box(
                local_bb,
                raster_info['projection_wkt'], osr.SRS_WKT_WGS84_LAT_LONG)

            local_box_wgs84 = shapely.geometry.box(
                local_bb_wgs84[0],
                local_bb_wgs84[1],
                local_bb_wgs84[2],
                local_bb_wgs84[3])

            # intersect local bb with target_bb
            intersect_box_wgs84 = local_box_wgs84.intersection(target_bb_wgs84)

            if intersect_box_wgs84.area == 0:
                continue

            gdf_intersect_box = geopandas.GeoDataFrame(
                geometry=[intersect_box_wgs84])

            # select points out of gdf_points that intersect with local bb
            local_points = geopandas.sjoin(
                gdf_points, gdf_intersect_box, op='intersects')
            if not local_points.index.is_unique:
                local_points = local_points.loc[~local_points.index.duplicated()]
            assert(local_points.index.is_unique)
            #LOGGER.debug(f'unique? {local_points.index.is_unique}')

            if len(local_points) == 0:
                continue

            local_coords = numpy.array([
                gdal.ApplyGeoTransform(inv_gt, point.x, point.y)
                for point in local_points['geometry']], dtype=int)
            local_coords = (local_coords - [
                offset_dict['xoff'], offset_dict['yoff']]).T

            #LOGGER.debug(f'local_coords: {local_coords}')
            local_points[basename] = (
                band.ReadAsArray(**offset_dict).T)[
                    local_coords[0, :], local_coords[1, :]]

            gdf_points.loc[local_points.index] = local_points
        if nodata is not None:
            LOGGER.debug(f'removing ndoata {nodata} from {basename}')
            gdf_points = gdf_points[gdf_points[basename] != nodata]

    return gdf_points


def __sample_data_old(
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
                # THE LAST ELEMENT IS THE FLOW ACCUMULATION THAT I WANT LOGGED
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


def prep_data(
        task_graph, predictor_raster_path_list, response_raster_path_list,
        raster_lookup_path):
    """Align global data."""

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


def generate_sample_points(
        raster_path_list, sample_polygon_path, bounding_box,
        holdback_prop, n_points, country_filter_list=None):
    """Create random sample points that are in bounds of the rasters.

    Args:
        raster_path_list (list): list of raster paths which are in WGS84
            projection.
        holdback_prop (float): between 0..1 representing what proportion of
            the window should be used for holdback, creates two sets
                * base sample
                * holdback sample

            any points that lie within a holdback_prop's buffer around
            the window are thrown out.
        n_points (int): number of samples.

    Return:
        GeoSeries of sample and holdback points
    """
    # include the vector bounding box information to make a global list
    print('read file')
    df = geopandas.read_file(sample_polygon_path)

    if country_filter_list:
        df = df[df['iso3'].isin(country_filter_list)]

    geom = df['geometry'].intersection(bounding_box)
    print('union')

    # TODO: add the raster bounds in here

    final_geom = geom.unary_union
    print('prep')
    final_geom_prep = prep(final_geom)
    x_min, y_min, x_max, y_max = final_geom.bounds

    x = numpy.random.uniform(x_min, x_max, n_points)
    y = numpy.random.uniform(y_min, y_max, n_points)

    box_width = holdback_prop*(x_max-x_min)
    box_height = holdback_prop*(y_max-y_min)

    holdback_box_edge = min(box_width, box_height)

    print('filter by allowed area')
    gdf_points = geopandas.GeoSeries(filter(
        final_geom_prep.contains, geopandas.points_from_xy(x, y)))

    for point in gdf_points:
        holdback_bounds = shapely.geometry.box(
            point.x-holdback_box_edge, point.y-holdback_box_edge,
            point.x+holdback_box_edge, point.y+holdback_box_edge,
            )
        if final_geom_prep.contains(holdback_bounds):
            break
        LOGGER.warn(f'skipping point {point} as a holdback bound')

    filtered_gdf = geopandas.GeoDataFrame(geometry=geopandas.GeoSeries(
        filter(lambda x: not holdback_bounds.contains(x), gdf_points)))
    filtered_gdf['holdback'] = False

    holdback_box = shapely.geometry.box(
        point.x-holdback_box_edge*0.5, point.y-holdback_box_edge*0.5,
        point.x+holdback_box_edge*0.5, point.y+holdback_box_edge*0.5,)

    holdback_points = geopandas.GeoDataFrame(geometry=geopandas.GeoSeries(
        filter(holdback_box.contains, gdf_points)))
    LOGGER.debug(f'holdbackpoints: {holdback_points}')
    holdback_points['holdback'] = True
    filtered_gdf = filtered_gdf.append(holdback_points)
    return filtered_gdf


def main():
    parser = argparse.ArgumentParser(
        description='create spatial samples of data on a global scale')
    parser.add_argument('--sample_rasters', type=str, nargs='+', help='path/pattern to list of rasters to sample', required=True)
    parser.add_argument('--holdback_prop', type=float, help='path/pattern to list of response rasters', required=True)
    parser.add_argument('--n_samples', type=int, help='number of point samples', required=True)
    parser.add_argument('--target_gpkg_path', type=str, help='name of target gpkg point samplefile', required=True)
    parser.add_argument('--iso_names', type=str, nargs='+', help='set of countries to allow, default is all')

    args = parser.parse_args()

    raster_path_set = set()

    for pattern in args.sample_rasters:
        file_path_list = list(glob.glob(pattern))
        if not file_path_list:
            raise FileNotFoundError(f"{pattern} doesn't match any files")
        for file_path in file_path_list:
            if (geoprocessing.get_gis_type(file_path) !=
                    geoprocessing.RASTER_TYPE):
                raise ValueError(
                    f'{file_path} found at {pattern} is not a raster')
        raster_path_set.update(file_path_list)

    raster_bounding_box_list = []
    basename_list = []
    nodata_list = []
    # find lat/lng bounding box
    for raster_path in raster_path_set:
        raster_info = geoprocessing.get_raster_info(raster_path)
        raster_bounding_box_list.append(
            geoprocessing.transform_bounding_box(
                raster_info['bounding_box'],
                raster_info['projection_wkt'], osr.SRS_WKT_WGS84_LAT_LONG))
        basename_list.append(
            os.path.basename(os.path.splitext(raster_path)[0]))
        nodata_list.append(raster_info['nodata'][0])
    target_bb_wgs84 = geoprocessing.merge_bounding_box_list(
        raster_bounding_box_list, 'intersection')
    target_box_wgs84 = shapely.geometry.box(
        target_bb_wgs84[0],
        target_bb_wgs84[1],
        target_bb_wgs84[2],
        target_bb_wgs84[3])

    sample_polygon_path = r"D:\repositories\critical-natural-capital-optimizations\data\countries_iso3_md5_6fb2431e911401992e6e56ddf0a9bcda.gpkg"

    global_sample_df = geopandas.GeoDataFrame()

    # used to scale how many points are sampled with how many are dropped for nodata
    oversample_rate = 2.0
    n_points_to_sample = oversample_rate * args.n_samples
    while True:
        filtered_gdf_points = generate_sample_points(
            raster_path_set, sample_polygon_path, target_box_wgs84,
            args.holdback_prop, n_points_to_sample, args.iso_names)

        LOGGER.info('sample data...')
        sample_df = sample_data(
            raster_path_set, filtered_gdf_points, target_box_wgs84)
        global_sample_df.append(sample_df)
        if len(global_sample_df) >= args.n_samples:
            break
        else:
            # sample more points but at a rate that's inversely proportional to
            # how many were dropped from the last sample
            oversample_rate *= n_points_to_sample / len(sample_df)
            n_points_to_sample = oversample_rate * (
                args.n_samples - len(global_sample_df))
            LOGGER.debug(f'sampling {n_points_to_sample} more points')

        sample_df.to_file(args.target_gpkg_path, driver="GPKG")

    print('plot')
    LOGGER.debug(f" all {filtered_gdf_points}")
    LOGGER.debug(f" non holdback {filtered_gdf_points[filtered_gdf_points['holdback'] == False]}")
    LOGGER.debug(f" holdback {filtered_gdf_points[filtered_gdf_points['holdback'] == True]}")
    fig, ax = plt.subplots(figsize=(12, 10))
    v = filtered_gdf_points[filtered_gdf_points['holdback']==False]
    v.plot(ax=ax, color='blue', markersize=2.5)

    w = filtered_gdf_points[filtered_gdf_points['holdback']==True]
    print(w)
    w.plot(ax=ax, color='green', markersize=2.5)
    plt.show()

    LOGGER.debug('all done')


if __name__ == '__main__':
    main()
