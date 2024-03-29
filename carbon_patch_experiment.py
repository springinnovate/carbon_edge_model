"""Experiment:

Sample forest mask and carbon density and see if there's a relationship
between the resolution of the carbon density and the forest mask.
"""
import shutil
import tempfile
import concurrent
import logging
import os

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score
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
        target_raster_path = os.path.join(local_dir, f'{raster_index}.tif')
        geoprocessing.warp_raster(
            raster_path, [pixel_length, -pixel_length], target_raster_path,
            'near', target_bb=wgs84_box.bounds,
            target_projection_wkt=osr.SRS_WKT_WGS84_LAT_LONG,
            osr_axis_mapping_strategy=osr.OAMS_TRADITIONAL_GIS_ORDER)
    # downsample with average

    for downsample_index in range(1, 3):
        target_raster_path = os.path.join(local_dir, f'{downsample_index}_avg.tif')
        target_pixel_length = raster_path_list[0][2]
        geoprocessing.warp_raster(
            os.path.join(local_dir, f'{downsample_index}.tif'),
            [target_pixel_length, -target_pixel_length],
            target_raster_path,
            'average', target_bb=wgs84_box.bounds,
            target_projection_wkt=osr.SRS_WKT_WGS84_LAT_LONG,
            osr_axis_mapping_strategy=osr.OAMS_TRADITIONAL_GIS_ORDER,
            output_type=gdal.GDT_Float32)

    raw_forest = geoprocessing.raster_to_numpy_array(
        os.path.join(local_dir, '1_avg.tif'), band_id=1)
    fraction_of_forest = numpy.count_nonzero(raw_forest)/raw_forest.size
    carbon = geoprocessing.raster_to_numpy_array(
        os.path.join(local_dir, '0.tif'), band_id=raster_path_list[0][1]).flatten()
    forest_cover = geoprocessing.raster_to_numpy_array(
        os.path.join(local_dir, '1_avg.tif'), band_id=1).flatten()
    convolution_forest_cover = geoprocessing.raster_to_numpy_array(
        os.path.join(local_dir, '2_avg.tif'), band_id=1).flatten()

    valid_mask = convolution_forest_cover > 0
    convolution_forest_cover = convolution_forest_cover[valid_mask]
    forest_cover = forest_cover[valid_mask]
    carbon = carbon[valid_mask]

    # TODO: also do an edge convolution and use that as a 3rd variable

    valid_mask = (carbon < 600) & (carbon > 10) & (forest_cover > 0)
    if numpy.any(valid_mask) and fraction_of_forest > 0.1 and fraction_of_forest < 0.9:
        # results = scipy.stats.pearsonr(
        #     carbon[valid_mask], forest_cover[valid_mask])

        observations = numpy.vstack(
            [carbon, forest_cover, convolution_forest_cover])
        # observations = numpy.vstack(
        #     [carbon, forest_cover])
        # results = scipy.stats.spearmanr(
        #     observations, axis=1, alternative='two-sided')

        svm_1 = make_pipeline(StandardScaler(), LinearSVR(max_iter=5000, loss='squared_epsilon_insensitive', epsilon=0, dual=False))
        observations = forest_cover.reshape(-1, 1)
        svm_1.fit(observations, carbon)
        carbon_pred = svm_1.predict(observations)
        forest_cover_r2 = r2_score(carbon, carbon_pred)

        svm_2 = make_pipeline(StandardScaler(), LinearSVR(max_iter=5000, loss='squared_epsilon_insensitive', epsilon=0, dual=False))
        observations = convolution_forest_cover.reshape(-1, 1)
        svm_2.fit(observations, carbon)
        convolution_pred = svm_2.predict(observations)
        convolution_r2 = r2_score(carbon, convolution_pred)

        svm_3 = make_pipeline(StandardScaler(), LinearSVR(max_iter=5000, loss='squared_epsilon_insensitive', epsilon=0, dual=False))
        observations = numpy.vstack(
            [forest_cover, convolution_forest_cover]).transpose()
        svm_3.fit(observations, carbon)
        both_pred = svm_3.predict(observations)
        both_r2 = r2_score(carbon, both_pred)

        # This does forest convolution vs carbon minus forest vs carbon
        # results = (results[0][0][2]-results[0][0][1], results[1][0][1])
        # This does forest convolution vs forest
        results = (forest_cover_r2, convolution_r2, both_r2)
    else:
        results = (numpy.nan, numpy.nan, numpy.nan)

    shutil.rmtree(local_dir, ignore_errors=True)

    return results


def main():
    """Entry point."""
    carbon_density_path = r"D:\repositories\carbon_edge_model\local_data_from_z\baccini_carbon_data_2003_2014_compressed_md5_11d1455ee8f091bf4be12c4f7ff9451b.tif"
    # TODO: build convolution off this forest_cover_path raster
    forest_cover_path = r"D:\repositories\carbon_edge_model\processed_rasters\fc_stack_hansen_forest_cover2014_compressed.tif"
    forest_gaussian_path = r"D:\repositories\carbon_edge_model\processed_rasters\gf_5.0_fc_stack_hansen_forest_cover2014_compressed.tif"
    raster_path_list = [
        (carbon_density_path, 2014-2000+1, 500/110000),
        (forest_cover_path, 1, 90/110000),
        (forest_gaussian_path, 1, 90/110000)]
    raster_bounding_box_list = [(-179, -80, 179, 80)]
    basename_list = []
    nodata_list = []
    # find lat/lng bounding box
    for raster_path in [carbon_density_path, forest_cover_path, forest_gaussian_path]:
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
    box_radius = 500/110000*100

    country_iso_list = [
        'BRA', 'ECU', 'URY', 'CRI', 'HTI', 'DOM', 'COD', 'COG', 'GAB',
        'GNQ', 'RWA', 'BDI', 'MYS', 'LKA', 'BRN',  'PNG', 'JPN', 'EST', 'MNE']
    country_iso_list = ['BRA']
    country_stat_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        def process_country(country_iso):
            #with concurrent.futures.ProcessPoolExecutor() as thread_executor:
            LOGGER.debug(country_iso)
            original_points = 100
            points_left = original_points
            pearson_stat_list = []
            while points_left > 0:
                sample_regions = generate_sample_points(
                    countries_vector_path, target_box_wgs84,
                    original_points*5, box_radius, country_filter_list=[
                        country_iso])
                worker_list = []
                for index, box in enumerate(sample_regions):
                    # pv_stat, p_val = pearson_correlation(
                    #     raster_path_list, box)
                    worker_list.append(executor.submit(
                        pearson_correlation, raster_path_list, box))
                LOGGER.debug(f'waiting on {len(worker_list)} results for {country_iso}')
                for worker in worker_list:
                    (forest_cover_r2, convolution_r2, both_r2) = worker.result()
                    if any([numpy.isnan(x) for x in (
                            forest_cover_r2, convolution_r2, both_r2)]):
                        continue
                    # LOGGER.debug(f'********* her4e are results: {pv_stat}, {p_val}')
                    # if numpy.isnan(pv_stat) or p_val > 0.05:
                    #     continue
                    # pearson_stat_list.append((pv_stat, p_val))
                    pearson_stat_list.append((forest_cover_r2, convolution_r2, both_r2))
                    points_left -= 1
                    LOGGER.debug(f'{points_left} for {country_iso}')
                    if points_left == 0:
                        break
            #pearson_stats = numpy.array(pearson_stat_list)[:, 0]
            pearson_stats = [list(x) for x in zip(*pearson_stat_list)]
            LOGGER.debug(pearson_stats)
            return pearson_stats
        for country_iso in country_iso_list:
            country_stat_list.extend(process_country(country_iso))
        # work_list = [
        #     executor.submit(process_country, country_iso)
        #     for country_iso in country_iso_list]
        # for worker in work_list:
        #     country_stat_list.append(worker.result())

    plt.title(
        'Pearson Correlation Coefficient Study\nRed line indicates .75 '
        'significance')
    plt.ylabel('Pearson Statistic')
    plt.xlabel('Country ISO3')
    LOGGER.debug(len(country_stat_list))
    LOGGER.debug(country_stat_list)

    country_iso_labels = [(f'{iso}_forest_cover', f'{iso}_convolution', f'{iso}_both') for iso in country_iso_list]
    LOGGER.debug(country_iso_labels)
    # flatten
    country_iso_labels = [y for x in country_iso_labels for y in x]
    #country_stat_list = [y for x in country_stat_list for y in x]
    LOGGER.debug(len(country_stat_list))
    plt.boxplot(
        country_stat_list, labels=country_iso_labels)  # Plot a line at each location specified in a
    plt.yticks([0, .2, .4, .6, .8, .9, 1])
    plt.axhline(y=0.75, color='r', linestyle=':')
    plt.show()


if __name__ == '__main__':
    main()
