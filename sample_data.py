"""Sample raster data into geopandas structure."""
import argparse
import glob
import logging
import os
import time
import sys

from ecoshard import geoprocessing
from osgeo import gdal
from osgeo import osr
from shapely.prepared import prep
import ecoshard
import geopandas
import pandas
import matplotlib.pyplot as plt
import numpy
import shapely

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)

logging.getLogger('fiona').setLevel(logging.WARN)


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
        Any nodata values in the samples are set to 0.
    """
    LOGGER.debug(f'target_bb_wgs84 {target_bb_wgs84}')

    # sample each raster by its block range so long as its within the
    # bounding box, this is complicated but it saves us from randomly reading
    # all across the raster
    last_time = time.time()
    local_bb_transform_cache = {}
    for raster_path in sorted(raster_path_list):
        raster_info = geoprocessing.get_raster_info(raster_path)
        n_bands = len(raster_info['nodata'])
        for band_index in range(1, n_bands+1):
            basename = os.path.basename(os.path.splitext(raster_path)[0])
            if n_bands > 1:
                basename += f'_{band_index}'

            nodata = raster_info['nodata'][0]
            if nodata is None:
                nodata = 0
            # TODO: gdf_points[basename] = nodata
            gdf_points[basename] = -999

            gt = raster_info['geotransform']
            inv_gt = gdal.InvGeoTransform(gt)
            raster = gdal.OpenEx(raster_path)
            band = raster.GetRasterBand(band_index)
            LOGGER.debug(f'processing {basename}')
            n_total = raster_info['raster_size'][0]*raster_info['raster_size'][1]
            n_processed = 0
            for offset_dict in geoprocessing.iterblocks(
                    (raster_path, 1), offset_only=True, largest_block=2**29):
                if time.time()-last_time > 5:
                    LOGGER.debug(
                        f'{n_processed/n_total*100:.2f}% complete for {basename} {n_processed} {n_total}')
                    last_time = time.time()
                n_processed += offset_dict['win_xsize']*offset_dict['win_ysize']
                LOGGER.debug(offset_dict)
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
                    LOGGER.debug(f'{local_bb} vs {local_bb_wgs84} offset_dict {offset_dict} geotransform {gt}')

                    local_box_wgs84 = shapely.geometry.box(
                        local_bb_wgs84[0],
                        local_bb_wgs84[1],
                        local_bb_wgs84[2],
                        local_bb_wgs84[3])

                    intersect_box_wgs84 = local_box_wgs84.intersection(
                        target_bb_wgs84)
                    LOGGER.debug(intersect_box_wgs84.area)
                    if intersect_box_wgs84.area == 0:
                        local_bb_transform_cache[local_bb_key] = None
                        continue

                    local_points = gdf_points.cx[
                        intersect_box_wgs84.bounds[0]:intersect_box_wgs84.bounds[2],
                        intersect_box_wgs84.bounds[1]:intersect_box_wgs84.bounds[3],
                        ]
                    local_points = local_points.set_crs(
                        osr.SRS_WKT_WGS84_LAT_LONG)
                    local_points = local_points.to_crs(
                        raster_info['projection_wkt'])
                    LOGGER.debug(local_points)

                    if len(local_points) == 0:
                        local_bb_transform_cache[local_bb_key] = None
                        continue

                    local_coords = numpy.array([
                        gdal.ApplyGeoTransform(inv_gt, point.x, point.y)
                        for point in local_points['geometry']], dtype=int)
                    LOGGER.debug(local_coords)
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
                gdf_points.loc[local_points_index, basename] = raster_data
                #gdf_points = gdf_points.loc[local_points_index]

    return gdf_points


#@profile
def generate_sample_points(
        raster_path_list, sample_polygon_path, bounding_box,
        holdback_boxes, holdback_margin, n_points, country_filter_list=None):
    """Create random sample points that are in bounds of the rasters.

    Args:
        raster_path_list (list): list of raster paths which are in the same
            projection.
        sample_polygon_path (str): path to polygon vector that is used to
            limit the point selection or ``None``.
        bounding_box (4-float): minx, miny, maxx, maxy tuple of the total
            bounding box.
        holdback_boxes (list of shapely.Box): list of boxes to indicate
            holdback point regions
        holdback_margin (float): margin to holdback the points around the
            box so we don't have spatial correlation.
        n_points (int): number of samples.
        country_filter_list (list of str): list of country names to only
            select points in. If None, then selects everhwere.

    Return:
        GeoSeries of sample and holdback points
    """
    # include the vector bounding box information to make a global list
    sample_points = 0
    sample_point_list = []
    while sample_points < n_points:
        if sample_polygon_path is not None:
            LOGGER.debug(f'load {sample_polygon_path}')
            df = geopandas.read_file(sample_polygon_path)

            if country_filter_list:
                df = df[df['iso3'].isin(country_filter_list)]

            geom = df['geometry'].intersection(bounding_box)
            final_geom = geom.unary_union
            final_geom_prep = prep(final_geom)
            x_min, y_min, x_max, y_max = final_geom.bounds
        else:
            x_min, y_min, x_max, y_max = bounding_box.bounds
            final_geom_prep = prep(bounding_box)

        x = numpy.random.uniform(x_min, x_max, n_points)
        y = numpy.random.uniform(y_min, y_max, n_points)
        sample_point_list.append(geopandas.GeoSeries(filter(
            final_geom_prep.contains, geopandas.points_from_xy(x, y))))
        sample_points += sample_point_list[-1].size
        LOGGER.debug(f'{sample_points} so far')

    points_gdf = geopandas.GeoSeries(
        pandas.concat(sample_point_list, ignore_index=True),
        crs=sample_point_list[0].crs)

    # TODO: create bounds list
    # TODO: create shapely prepped objects for bounds list and holdback list
    # TODO: use the above in filtering points

    holdback_bounds = shapely.prepared.prep(shapely.ops.unary_union([
        box.buffer(holdback_margin) for box in holdback_boxes]))
    prep_holdback_box_list = shapely.prepared.prep(shapely.ops.unary_union(
        holdback_boxes))

    non_holdback_gdf = geopandas.GeoDataFrame(geometry=geopandas.GeoSeries(
        filter(lambda x: not holdback_bounds.contains(x), points_gdf)))
    non_holdback_gdf['holdback'] = False

    LOGGER.debug(f'non holdback points: {non_holdback_gdf.size}')

    holdback_gdf = geopandas.GeoDataFrame(geometry=geopandas.GeoSeries(
        filter(prep_holdback_box_list.contains, points_gdf)))
    holdback_gdf['holdback'] = True
    LOGGER.debug(f'holdback points: {holdback_gdf.size}')
    filtered_gdf = non_holdback_gdf.append(holdback_gdf, ignore_index=True)
    filtered_gdf = filtered_gdf.set_crs(osr.SRS_WKT_WGS84_LAT_LONG)
    return filtered_gdf


#@profile
def main():
    parser = argparse.ArgumentParser(
        description='create spatial samples of data on a global scale')
    parser.add_argument('--sample_rasters', type=str, nargs='+', help='path/pattern to list of rasters to sample', required=True)
    parser.add_argument('--holdback_centers', type=str, nargs='+', help='list of lat/lng bounding box centers to holdback', required=True)
    parser.add_argument('--holdback_margin', type=float, help='margin around the holdback box to ignore', required=True)
    parser.add_argument('--n_samples', type=int, help='number of point samples', required=True)
    parser.add_argument('--iso_names', type=str, nargs='+', help='set of countries to allow, default is all')
    parser.add_argument(
        '--sample_vector_path', type=str,
        help='path to a vector to limit sample points, if left off, samples to bounding box of rasters')

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

    holdback_boxes = []
    for lat_lng_tuple in args.holdback_centers:
        lat, lng = [
            float(x) for x in lat_lng_tuple.replace('(', '').replace(
                ')', '').split(',')]
        holdback_boxes.append(
            shapely.geometry.box(
                lng-args.holdback_margin, lat-args.holdback_margin,
                lng+args.holdback_margin, lat+args.holdback_margin))

    LOGGER.debug(holdback_boxes)

    # default bounding box list with some reasonable bounds
    raster_bounding_box_list = [(-179, -80, 179, 80)]
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

    LOGGER.debug(f'target box in wgs84: {target_box_wgs84}')

    LOGGER.info(f'generate {args.n_samples} sample points')
    filtered_gdf_points = generate_sample_points(
        raster_path_set, args.sample_vector_path, target_box_wgs84,
        holdback_boxes, args.holdback_margin, args.n_samples,
        args.iso_names)

    LOGGER.info('plot')
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

    LOGGER.info(f'sample data with {len(filtered_gdf_points)}...')
    sample_df = sample_data(
        raster_path_set, filtered_gdf_points, target_box_wgs84)

    target_gpkg_path = (
        f'sampled_points_'
        f'{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.gpkg')
    LOGGER.info(f'saving  {len(sample_df)} to {target_gpkg_path}')
    sample_df.to_file(target_gpkg_path, driver="GPKG")

    LOGGER.info(f'hashing {target_gpkg_path}')
    ecoshard.hash_file(
        target_gpkg_path, rename=True, hash_algorithm='md5', force=True,
        hash_length=6)

    LOGGER.debug('all done')


if __name__ == '__main__':
    main()
