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
import geopandas
import matplotlib.pyplot as plt
import numpy
import shapely

gdal.SetCacheMax(2**27)


#@profile
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
    local_bb_transform_cache = {}
    for raster_path in sorted(raster_path_list):
        raster_info = geoprocessing.get_raster_info(raster_path)
        basename = os.path.basename(os.path.splitext(raster_path)[0])

        nodata = raster_info['nodata'][0]
        if nodata is None:
            nodata = 0
        gdf_points[basename] = nodata

        gt = raster_info['geotransform']
        inv_gt = gdal.InvGeoTransform(gt)
        raster = gdal.OpenEx(raster_path)
        band = raster.GetRasterBand(1)
        LOGGER.debug(f'processing {basename}')
        n_total = raster_info['raster_size'][0]*raster_info['raster_size'][1]
        n_processed = 0
        for offset_dict in geoprocessing.iterblocks(
                (raster_path, 1), offset_only=True, largest_block=2**28):
            if time.time()-last_time > 5:
                LOGGER.debug(
                    f'{n_processed/n_total*100:.2f}% complete for {basename} {n_processed} {n_total}')
                last_time = time.time()
            n_processed += offset_dict['win_xsize']*offset_dict['win_ysize']
            local_bb_key = (
                gt, offset_dict['xoff'], offset_dict['yoff'],
                offset_dict['win_xsize'], offset_dict['win_ysize'],
                raster_info['projection_wkt'])
            if local_bb_key not in local_bb_transform_cache:
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

                intersect_box_wgs84 = local_box_wgs84.intersection(
                    target_bb_wgs84)
                if intersect_box_wgs84.area == 0:
                    local_bb_transform_cache[local_bb_key] = None
                    continue

                local_points = gdf_points.cx[
                    intersect_box_wgs84.bounds[0]:intersect_box_wgs84.bounds[2],
                    intersect_box_wgs84.bounds[1]:intersect_box_wgs84.bounds[3],
                    ]

                if len(local_points) == 0:
                    local_bb_transform_cache[local_bb_key] = None
                    continue

                local_coords = numpy.array([
                    gdal.ApplyGeoTransform(inv_gt, point.x, point.y)
                    for point in local_points['geometry']], dtype=int)
                min_x = min(local_coords[:, 0])
                min_y = min(local_coords[:, 1])
                max_x = max(local_coords[:, 0])
                max_y = max(local_coords[:, 1])
                local_window = {
                    'xoff': int(min_x),
                    'yoff': int(min_y),
                    'win_xsize': int(max_x-min_x)+1,
                    'win_ysize': int(max_y-min_y)+1,
                }
                local_coords = (local_coords - [
                    local_window['xoff'], local_window['yoff']]).T

                local_bb_transform_cache[local_bb_key] = (
                    local_points.index, local_window, local_coords)

            payload = local_bb_transform_cache[local_bb_key]
            if payload is None:
                continue
            local_points_index, local_window, local_coords = payload

            raster_data = (
                band.ReadAsArray(**local_window).T)[
                    local_coords[0, :], local_coords[1, :]]
            # 0 out nodata
            if nodata is not None:
                raster_data[numpy.isclose(raster_data, nodata)] = 0.0
            #LOGGER.debug(f'{offset_dict} {local_window} {raster_data}')
            gdf_points.loc[local_points_index, basename] = raster_data

    return gdf_points


#@profile
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
    holdback_points['holdback'] = True
    filtered_gdf = filtered_gdf.append(holdback_points, ignore_index=True)
    return filtered_gdf


#@profile
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

    # used to scale how many points are sampled with how many are dropped for nodata
    filtered_gdf_points = generate_sample_points(
        raster_path_set, sample_polygon_path, target_box_wgs84,
        args.holdback_prop, args.n_samples, args.iso_names)

    LOGGER.info('sample data...')
    sample_df = sample_data(
        raster_path_set, filtered_gdf_points, target_box_wgs84)
    LOGGER.info(f'got {len(sample_df)} number of points')

    sample_df.to_file(args.target_gpkg_path, driver="GPKG")

    print('plot')
    LOGGER.debug(f" all {filtered_gdf_points}")
    LOGGER.debug(f" non holdback {filtered_gdf_points[filtered_gdf_points['holdback'] == False]}")
    LOGGER.debug(f" holdback {filtered_gdf_points[filtered_gdf_points['holdback'] == True]}")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(f'{len(filtered_gdf_points)} points')
    v = filtered_gdf_points[filtered_gdf_points['holdback']==False]
    v.plot(ax=ax, color='blue', markersize=2.5)

    w = filtered_gdf_points[filtered_gdf_points['holdback']==True]
    print(w)
    w.plot(ax=ax, color='green', markersize=2.5)
    plt.show()

    LOGGER.debug('all done')


if __name__ == '__main__':
    main()
