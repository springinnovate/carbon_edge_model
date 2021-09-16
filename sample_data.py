"""Script to download everything needed to train the models."""
import argparse
import glob
import logging
import os
import time

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)

logging.getLogger('matplotlib').setLevel(logging.WARN)
logging.getLogger('fiona').setLevel(logging.WARN)

from ecoshard import geoprocessing
from osgeo import gdal
from osgeo import osr
from shapely.prepared import prep
from utils import esa_to_carbon_model_landcover_types
import geopandas
import matplotlib.pyplot as plt
import numpy
import shapely

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


@profile
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

            local_points = gdf_points.cx[
                local_bb_wgs84[0]:local_bb_wgs84[2],
                local_bb_wgs84[1]:local_bb_wgs84[3],
                ]
            if len(local_points) == 0:
                continue

            local_coords = numpy.array([
                gdal.ApplyGeoTransform(inv_gt, point.x, point.y)
                for point in local_points['geometry']], dtype=int)
            min_x = min(local_coords[:, 0])
            min_y = min(local_coords[:, 1])
            max_x = max(local_coords[:, 0])
            max_y = max(local_coords[:, 1])
            local_window = {
                'xoff': min_x,
                'yoff': min_y,
                'win_xsize': max_x-min_x,
                'win_ysize': max_y-min_y,
            }
            local_coords = (local_coords - [
                local_window['xoff'], local_window['yoff']]).T

            raster_data = (
                band.ReadAsArray(**offset_dict).T)[
                    local_coords[0, :], local_coords[1, :]]
            gdf_points.loc[local_points.index, basename] = raster_data

        if nodata is not None:
            LOGGER.debug(f'removing ndoata {nodata} from {basename}')
            LOGGER.debug(f'before: {gdf_points}')
            gdf_points = gdf_points[gdf_points[basename] != nodata]
            LOGGER.debug(f'after: {gdf_points}')
        break

    return gdf_points


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
        LOGGER.warning(f'skipping point {point} as a holdback bound')

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
    filtered_gdf = filtered_gdf.append(holdback_points, ignore_index=True)
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
    n_points_to_sample = int(oversample_rate * args.n_samples)
    while True:
        LOGGER.debug(f'sampling {n_points_to_sample} points {len(global_sample_df)} of {args.n_samples} sampled so far')
        filtered_gdf_points = generate_sample_points(
            raster_path_set, sample_polygon_path, target_box_wgs84,
            args.holdback_prop, n_points_to_sample, args.iso_names)

        LOGGER.info('sample data...')
        sample_df = sample_data(
            raster_path_set, filtered_gdf_points, target_box_wgs84)
        if len(sample_df) == 0:
            LOGGER.warn('no valid points came out, trying again')
            continue
        global_sample_df.append(sample_df, ignore_index=True)
        if len(global_sample_df) >= args.n_samples:
            break
        else:
            # sample more points but at a rate that's inversely proportional to
            # how many were dropped from the last sample
            oversample_rate *= n_points_to_sample / len(sample_df)
            n_points_to_sample = int(oversample_rate * (
                args.n_samples - len(global_sample_df)))
            LOGGER.debug(f'sampling {n_points_to_sample} more points')
        return

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
