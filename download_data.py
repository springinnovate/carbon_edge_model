"""Script to download everything needed to train the models."""
import os
import collections
import multiprocessing
import time
import logging

from osgeo import gdal
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

TIME_PREDICTOR_LIST = [
    'baccini_carbon_error_compressed_wgs84__md5_77ea391e63c137b80727a00e4945642f.tif',
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2003-v2.0.7_smooth_compressed.tif',
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2004-v2.0.7_smooth_compressed.tif',
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2005-v2.0.7_smooth_compressed.tif',
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2006-v2.0.7_smooth_compressed.tif',
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2007-v2.0.7_smooth_compressed.tif',
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2008-v2.0.7_smooth_compressed.tif',
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2009-v2.0.7_smooth_compressed.tif',
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2010-v2.0.7_smooth_compressed.tif',
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2011-v2.0.7_smooth_compressed.tif',
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2012-v2.0.7_smooth_compressed.tif',
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2013-v2.0.7_smooth_compressed.tif',
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif',
]

PREDICTOR_LIST = [
    'accessibility_to_cities_2015_30sec_compressed_wgs84__md5_a6a8ffcb6c1025c131f7663b80b3c9a7.tif',
    'altitude_10sec_compressed_wgs84__md5_bfa771b1aef1b18e48962c315e5ba5fc.tif',
    'bio_01_30sec_compressed_wgs84__md5_3f851546237e282124eb97b479c779f4.tif',
    'bio_02_30sec_compressed_wgs84__md5_7ad508baff5bbd8b2e7991451938a5a7.tif',
    'bio_03_30sec_compressed_wgs84__md5_a2de2d38c1f8b51f9d24f7a3a1e5f142.tif',
    'bio_04_30sec_compressed_wgs84__md5_94cfca6af74ffe52316a02b454ba151b.tif',
    'bio_05_30sec_compressed_wgs84__md5_bdd225e46613405c80a7ebf7e3b77249.tif',
    'bio_06_30sec_compressed_wgs84__md5_ef252a4335eafb7fe7b4dc696d5a70e3.tif',
    'bio_07_30sec_compressed_wgs84__md5_1db9a6cdce4b3bd26d79559acd2bc525.tif',
    'bio_08_30sec_compressed_wgs84__md5_baf898dd624cfc9415092d7f37ae44ff.tif',
    'bio_09_30sec_compressed_wgs84__md5_180c820aae826529bfc824b458165eee.tif',
    'bio_10_30sec_compressed_wgs84__md5_d720d781970e165a40a1934adf69c80e.tif',
    'bio_11_30sec_compressed_wgs84__md5_f48a251c54582c22d9eb5d2158618bbe.tif',
    'bio_12_30sec_compressed_wgs84__md5_23cb55c3acc544e5a941df795fcb2024.tif',
    'bio_13_30sec_compressed_wgs84__md5_b004ebe58d50841859ea485c06f55bf6.tif',
    'bio_14_30sec_compressed_wgs84__md5_7cb680af66ff6c676441a382519f0dc2.tif',
    'bio_15_30sec_compressed_wgs84__md5_edc8e5af802448651534b7a0bd7113ac.tif',
    'bio_16_30sec_compressed_wgs84__md5_a9e737a926f1f916746d8ce429c06fad.tif',
    'bio_17_30sec_compressed_wgs84__md5_0bc4db0e10829cd4027b91b7bbfc560f.tif',
    'bio_18_30sec_compressed_wgs84__md5_76cf3d38eb72286ba3d5de5a48bfadd4.tif',
    'bio_19_30sec_compressed_wgs84__md5_a91b8b766ed45cb60f97e25bcac0f5d2.tif',
    'cec_0-5cm_mean_compressed_wgs84__md5_b3b4285906c65db596a014d0c8a927dd.tif',
    'cec_0-5cm_uncertainty_compressed_wgs84__md5_f0f4eb245fd2cc4d5a12bd5f37189b53.tif',
    'cec_5-15cm_mean_compressed_wgs84__md5_55c4d960ca9006ba22c6d761d552c82f.tif',
    'cec_5-15cm_uncertainty_compressed_wgs84__md5_880eac199a7992f61da6c35c56576202.tif',
    'cfvo_0-5cm_mean_compressed_wgs84__md5_7abefac8143a706b66a1b7743ae3cba1.tif',
    'cfvo_0-5cm_uncertainty_compressed_wgs84__md5_3d6b883fba1d26a6473f4219009298bb.tif',
    'cfvo_5-15cm_mean_compressed_wgs84__md5_ae36d799053697a167d114ae7821f5da.tif',
    'cfvo_5-15cm_uncertainty_compressed_wgs84__md5_1f2749cd35adc8eb1c86a67cbe42aebf.tif',
    'clay_0-5cm_mean_compressed_wgs84__md5_9da9d4017b691bc75c407773269e2aa3.tif',
    'clay_0-5cm_uncertainty_compressed_wgs84__md5_f38eb273cb55147c11b48226400ae79a.tif',
    'clay_5-15cm_mean_compressed_wgs84__md5_c136adb39b7e1910949b749fcc16943e.tif',
    'clay_5-15cm_uncertainty_compressed_wgs84__md5_0acc36c723aa35b3478f95f708372cc7.tif',
    'hillshade_10sec_compressed_wgs84__md5_192a760d053db91fc9e32df199358b54.tif',
    'night_lights_10sec_compressed_wgs84__md5_54e040d93463a2918a82019a0d2757a3.tif',
    'night_lights_5min_compressed_wgs84__md5_e36f1044d45374c335240777a2b94426.tif',
    'nitrogen_0-5cm_mean_compressed_wgs84__md5_6adecc8d790ccca6057a902e2ddd0472.tif',
    'nitrogen_0-5cm_uncertainty_compressed_wgs84__md5_4425b4bd9eeba0ad8a1092d9c3e62187.tif',
    'nitrogen_10sec_compressed_wgs84__md5_1aed297ef68f15049bbd987f9e98d03d.tif',
    'nitrogen_5-15cm_mean_compressed_wgs84__md5_9487bc9d293effeb4565e256ed6e0393.tif',
    'nitrogen_5-15cm_uncertainty_compressed_wgs84__md5_2de5e9d6c3e078756a59ac90e3850b2b.tif',
    'phh2o_0-5cm_mean_compressed_wgs84__md5_00ab8e945d4f7fbbd0bddec1cb8f620f.tif',
    'phh2o_0-5cm_uncertainty_compressed_wgs84__md5_8090910adde390949004f30089c3ae49.tif',
    'phh2o_5-15cm_mean_compressed_wgs84__md5_9b187a088ecb955642b9a86d56f969ad.tif',
    'phh2o_5-15cm_uncertainty_compressed_wgs84__md5_6809da4b13ebbc747750691afb01a119.tif',
    'sand_0-5cm_mean_compressed_wgs84__md5_6c73d897cdef7fde657386af201a368d.tif',
    'sand_0-5cm_uncertainty_compressed_wgs84__md5_efd87fd2062e8276148154c4a59c9b25.tif',
    'sand_5-15cm_uncertainty_compressed_wgs84__md5_03bc79e2bfd770a82c6d15e36a65fb5c.tif',
    'silt_0-5cm_mean_compressed_wgs84__md5_1d141933d8d109df25c73bd1dcb9d67c.tif',
    'silt_0-5cm_uncertainty_compressed_wgs84__md5_ac5ec50cbc3b9396cf11e4e431b508a9.tif',
    'silt_5-15cm_mean_compressed_wgs84__md5_d0abb0769ebd015fdc12b50b20f8c51e.tif',
    'silt_5-15cm_uncertainty_compressed_wgs84__md5_cc125c85815db0d1f66b315014907047.tif',
    'slope_10sec_compressed_wgs84__md5_e2bdd42cb724893ce8b08c6680d1eeaf.tif',
    'soc_0-5cm_mean_compressed_wgs84__md5_b5be42d9d0ecafaaad7cc592dcfe829b.tif',
    'soc_0-5cm_uncertainty_compressed_wgs84__md5_33c1a8c3100db465c761a9d7f4e86bb9.tif',
    'soc_5-15cm_mean_compressed_wgs84__md5_4c489f6132cc76c6d634181c25d22d19.tif',
    'tri_10sec_compressed_wgs84__md5_258ad3123f05bc140eadd6246f6a078e.tif',
    'wind_speed_10sec_compressed_wgs84__md5_7c5acc948ac0ff492f3d148ffc277908.tif',
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
    for raster_list, raster_type in [
            (TIME_PREDICTOR_LIST, 'time_predictor'),
            (PREDICTOR_LIST, 'predictor')]:
        for filename in raster_list:
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
                args=(
                    ecoshard_path, response_info['pixel_size'], aligned_path,
                    'near'),
                kwargs={
                    'target_bb': response_info['bounding_box'],
                    'target_projection_wkt': response_info['projection_wkt'],
                    'working_dir': WORKSPACE_DIR},
                dependent_task_list=[download_task],
                target_path_list=[aligned_path],
                task_name=f'align {aligned_path}')
            raster_lookup[raster_type].append(aligned_path)
    task_graph.join()
    task_graph.close()

    return raster_lookup


def sample_data(raster_lookup, n_points):
    """Sample data stack."""
    band_inv_gt_list = []
    for raster_path in raster_lookup['predictor']:
        raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
        band = raster.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        gt = raster.GetGeoTransform()
        inv_gt = gdal.InvGeoTransform(gt)
        band_inv_gt_list.append((raster_path, raster, band, nodata, gt, inv_gt))
        raster = None

    response_raster_path = raster_lookup['response'][0]
    response_info = pygeoprocessing.get_raster_info(response_raster_path)
    offset_list = list(pygeoprocessing.iterblocks(
        (response_raster_path, 1), offset_only=True, largest_block=0))
    lat_lng_bb_list = []
    LOGGER.info(f'creating {len(offset_list)} index boxes')
    for index, offset_dict in enumerate(offset_list):
        bb_lng_lat = [
            coord for coord in (
                gdal.ApplyGeoTransform(
                    response_info['geotransform'],
                    offset_dict['xoff'],
                    offset_dict['yoff']+offset_dict['win_ysize']) +
                gdal.ApplyGeoTransform(
                    response_info['geotransform'],
                    offset_dict['xoff']+offset_dict['win_xsize'],
                    offset_dict['yoff']))]
        lat_lng_bb_list.append((index, bb_lng_lat, None))
    LOGGER.info('creating the index all at once')
    baccini_memory_block_index = rtree.index.Index(lat_lng_bb_list)

    points_remaining = n_points
    lng_lat_vector = []
    X_vector = []
    y_vector = []
    LOGGER.info(f'build {n_points}')
    last_time = time.time()
    while points_remaining > 0:
        # from https://mathworld.wolfram.com/SpherePointPicking.html
        u = numpy.random.random((points_remaining,))
        v = numpy.random.random((points_remaining,))
        # pick between -180 and 180
        lng_arr = (2.0 * numpy.pi * u) * 180/numpy.pi - 180
        lat_arr = numpy.arccos(2*v-1) * 180/numpy.pi - 90
        valid_mask = numpy.abs(lat_arr) < 70
        window_index_to_point_list_map = collections.defaultdict(list)
        for lng, lat in zip(lng_arr[valid_mask], lat_arr[valid_mask]):
            if time.time() - last_time > 5.0:
                LOGGER.info(f'working ... {points_remaining} left')
                last_time = time.time()

            window_index = list(baccini_memory_block_index.intersection(
                (lng, lat, lng, lat)))[0]
            window_index_to_point_list_map[window_index].append((lng, lat))

        for window_index, point_list in window_index_to_point_list_map.items():
            if not point_list:
                LOGGER.info(f'not point list on index {window_index}')
                continue
            LOGGER.info(window_index)

            # raster_index_to_array_list is an xoff, yoff, array list
            # TODO: loop through each point in point list
            for lng, lat in point_list:
                # check each array/raster and ensure it's not nodata or if it
                # is, set to the valid value
                working_sample_list = []
                valid_working_list = True
                for index, (raster_path, _, band, nodata, gt, inv_gt) in enumerate(band_inv_gt_list):
                    x, y = [int(v) for v in (gdal.ApplyGeoTransform(inv_gt, lng, lat))]
                    if x < 0 or x >= band.XSize or y < 0 or y >= band.YSize:
                        # out of bounds
                        valid_working_list = False
                        break
                    val = band.ReadAsArray(x, y, 1, 1)[0, 0]
                    if nodata is None or val != nodata:
                        working_sample_list.append(val)
                    else:
                        # nodata value, skip
                        #LOGGER.info(f'nodata or oob on {raster_path} at {lng} {lat}')
                        valid_working_list = False
                        break

                if valid_working_list:
                    points_remaining -= 1
                    lng_lat_vector.append((lng, lat))
                    y_vector.append(working_sample_list[0])
                    # first element is dep -- don't include it
                    X_vector.append(working_sample_list[1:])
                    LOGGER.info(f'******* DID IT ON {lng} {lat}')


if __name__ == '__main__':
    raster_lookup = download_data()
    sample_data(raster_lookup, 1000)
