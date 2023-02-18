"""Experiment:

Sample forest mask and carbon density and see if there's a relationship
between the resolution of the carbon density and the forest mask.
"""
import shutil
import tempfile
import concurrent
import logging
import os

from ecoshard import geoprocessing
from osgeo import gdal
from osgeo import osr
from shapely.prepared import prep
import matplotlib.pyplot as plt
import geopandas
import shapely
import numpy
import pandas
import scipy

WORKSPACE_DIR = 'carbon_patch_workspace'
os.makedirs(WORKSPACE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

gdal.SetCacheMax(2**27)


def generate_sample_points(
        sample_polygon_path, bounding_box,
        n_boxes, box_radius, country_filter_list=None):
    """Create random sample points that are in bounds of the rasters.

    Args:
        sample_polygon_path (str): path to polygon vector that is used to
            limit the point selection or ``None``.
        bounding_box (4-float): minx, miny, maxx, maxy tuple of the total
            bounding box.
        box_radius (float): the size of the box from the center to the
            edge in units of the input rasters.
        n_boxes (int): number of samples.
        country_filter_list (list of str): list of country names to only
            select points in. If None, then selects everhwere.

    Return:
        GeoSeries of sample and holdback points
    """
    # include the vector bounding box information to make a global list
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

    box_count = 0
    sample_point_list = []
    while box_count < n_boxes:
        x_sample = numpy.random.uniform(x_min, x_max, n_boxes)
        y_sample = numpy.random.uniform(y_min, y_max, n_boxes)
        sample_box_list = [
            shapely.geometry.box(
                x-box_radius,
                y-box_radius,
                x+box_radius,
                y+box_radius) for x, y in zip(x_sample, y_sample)]
        sample_point_list.append(geopandas.GeoSeries(filter(
            final_geom_prep.contains, sample_box_list)))
        box_count += sample_point_list[-1].size

    points_gdf = geopandas.GeoSeries(
        pandas.concat(sample_point_list, ignore_index=True),
        crs=sample_point_list[0].crs)

    return points_gdf


def pearson_correlation(raster_path_list, wgs84_box):
    """Return a list of raster slices in the wsg84_box."""
    # warp out a clip of the raster into wgs84 on both sides, then clip to
    # same bounding box
    local_dir = tempfile.mkdtemp(dir=WORKSPACE_DIR)
    for raster_index, (raster_path, _, pixel_length) in enumerate(raster_path_list):
        target_raster_path = f'{raster_index}.tif'
        geoprocessing.warp_raster(
            raster_path, [pixel_length, -pixel_length], target_raster_path,
            'near', target_bb=wgs84_box.bounds,
            target_projection_wkt=osr.SRS_WKT_WGS84_LAT_LONG,
            osr_axis_mapping_strategy=osr.OAMS_TRADITIONAL_GIS_ORDER)
    # downsample with average
    target_raster_path = '1_avg.tif'
    target_pixel_length = raster_path_list[0][2]
    geoprocessing.warp_raster(
        os.path.join(local_dir, '1.tif'), [target_pixel_length, -target_pixel_length],
        target_raster_path,
        'average', target_bb=wgs84_box.bounds,
        target_projection_wkt=osr.SRS_WKT_WGS84_LAT_LONG,
        osr_axis_mapping_strategy=osr.OAMS_TRADITIONAL_GIS_ORDER,
        output_type=gdal.GDT_Float32)

    raw_forest = geoprocessing.raster_to_numpy_array(
        os.path.join(local_dir, '1.tif'), band_id=1)
    fraction_of_forest = numpy.count_nonzero(raw_forest)/raw_forest.size
    carbon = geoprocessing.raster_to_numpy_array(
        os.path.join(local_dir, '0.tif'), band_id=raster_path_list[0][1])
    forest_cover = geoprocessing.raster_to_numpy_array(
        os.path.join(local_dir, '1_avg.tif'), band_id=1)

    valid_mask = (carbon < 600) & (carbon > 10) & (forest_cover > 0)
    if numpy.any(valid_mask) and fraction_of_forest > 0.1 and fraction_of_forest < 0.9:
        results = scipy.stats.pearsonr(carbon[valid_mask], forest_cover[valid_mask])
        # if results[0] < 0.00001:
        #     LOGGER.debug('real close to 0')
        #     sys.exit()
    else:
        results = (numpy.nan, numpy.nan)

    shutil.rmtree(local_dir, ignore_errors=True)

    return results


def main():
    """Entry point."""
    carbon_density_path = r"D:\repositories\carbon_edge_model\local_data_from_z\baccini_carbon_data_2003_2014_compressed_md5_11d1455ee8f091bf4be12c4f7ff9451b.tif"
    forest_cover_path = r"D:\repositories\carbon_edge_model\processed_rasters\fc_stack_hansen_forest_cover2014_compressed.tif"
    raster_path_list = [(carbon_density_path, 2014-2000+1, 500/110000), (forest_cover_path, 1, 90/110000)]
    raster_bounding_box_list = [(-179, -80, 179, 80)]
    basename_list = []
    nodata_list = []
    # find lat/lng bounding box
    for raster_path in [carbon_density_path, forest_cover_path]:
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

    countries_vector_path = r"D:\repositories\carbon_edge_model\countries_iso3_md5_9b11dd.gpkg"
    box_radius = 500/110000*10

    country_iso_list = [
        'BRA', 'ECU', 'URY', 'CRI', 'HTI', 'DOM', 'COD', 'COG', 'GAB',
        'GNQ', 'RWA', 'BDI', 'MYS', 'LKA', 'BRN',  'PNG', 'JPN', 'EST', 'MNE']
    country_stat_list = []
    for country_iso in country_iso_list:
        LOGGER.debug(country_iso)
        original_points = 100
        points_left = 100
        pearson_stat_list = []
        while points_left > 0:
            sample_regions = generate_sample_points(
                countries_vector_path, target_box_wgs84,
                original_points*2, box_radius, country_filter_list=[
                    country_iso])
            for index, box in enumerate(sample_regions):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    worker_list = []
                    # pv_stat, p_val = pearson_correlation(
                    #     raster_path_list, box)
                    worker_list.append(executor.submit(
                        pearson_correlation, raster_path_list, box))
                for worker in worker_list:
                    pv_stat, p_val = worker.result()
                    #LOGGER.debug(f'{pv_stat}, {p_val}')
                    if numpy.isnan(pv_stat) or p_val > 0.05:
                        continue
                    pearson_stat_list.append((pv_stat, p_val))
                    points_left -= 1
                    LOGGER.debug(f'{points_left} for {country_iso}')
                    if points_left == 0:
                        break

        pearson_stats = numpy.array(pearson_stat_list)
        country_stat_list.append(pearson_stats[:, 0])

    plt.title('Pearson Correlation Coefficient Study')
    plt.ylabel('Pearson Statistic')
    plt.xlabel('Country ISO3')
    plt.boxplot(
        country_stat_list, labels=country_iso_list)  # Plot a line at each location specified in a
    plt.show()
    # with open('data.csv', 'w') as data_file:
    #     for val in pearson_stats.flatten():
    #         data_file.write(f'{val}\n')


if __name__ == '__main__':
    main()
