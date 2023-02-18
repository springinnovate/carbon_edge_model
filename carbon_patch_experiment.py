"""Experiment:

Sample forest mask and carbon density and see if there's a relationship
between the resolution of the carbon density and the forest mask.
"""
import logging
import os
import sys

import rasterio
from rasterio.plot import show
from rasterio.windows import Window
from ecoshard import geoprocessing
from osgeo import gdal
from osgeo import osr
from shapely.prepared import prep
import skimage.measure
import matplotlib.pyplot as plt
import geopandas
import shapely
import numpy
import pandas
import scipy


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


def sample_rasters(raster_path_list, wgs84_box):
    """Return a list of raster slices in the wsg84_box."""
    # warp out a clip of the raster into wgs84 on both sides, then clip to
    # same bounding box

    for raster_index, (raster_path, _, pixel_length) in enumerate(raster_path_list):
        target_raster_path = f'{raster_index}.tif'
        raster_info = geoprocessing.get_raster_info(raster_path)
        LOGGER.debug(wgs84_box.bounds)
        geoprocessing.warp_raster(
            raster_path, [pixel_length, -pixel_length], target_raster_path,
            'near', target_bb=wgs84_box.bounds,
            target_projection_wkt=osr.SRS_WKT_WGS84_LAT_LONG,
            osr_axis_mapping_strategy=osr.OAMS_TRADITIONAL_GIS_ORDER)
    # downsample with average
    target_raster_path = '1_avg.tif'
    target_pixel_length = raster_path_list[0][2]
    geoprocessing.warp_raster(
        '1.tif', [target_pixel_length, -target_pixel_length],
        target_raster_path,
        'average', target_bb=wgs84_box.bounds,
        target_projection_wkt=osr.SRS_WKT_WGS84_LAT_LONG,
        osr_axis_mapping_strategy=osr.OAMS_TRADITIONAL_GIS_ORDER,
        output_type=gdal.GDT_Float32)

    return
    slice_list = []
    for raster_path, band_id in raster_path_list:
        raster_info = geoprocessing.get_raster_info(raster_path)
        src = rasterio.open(raster_path)
        LOGGER.debug(raster_path)
        # convert bounds to raster coordinates
        local_bounds = geoprocessing.transform_bounding_box(
            wgs84_box.bounds,
            osr.SRS_WKT_WGS84_LAT_LONG, raster_info['projection_wkt'])
        LOGGER.debug(local_bounds)
        # this will transform coordinates in the raster projection to rows/cols to extract
        rows, cols = rasterio.transform.rowcol(src.transform, local_bounds[0:4:2], local_bounds[3:0:-2])

        src_slice = src.read(band_id, window=Window.from_slices(rows, cols))
        src_slice[src_slice > 1000] = 0
        LOGGER.debug(f'{rows}, {cols}, {src_slice.shape}')
        slice_list.append(src_slice)

    factor = slice_list[1].shape[0]//slice_list[0].shape[0]
    LOGGER.debug(factor)
    fig, (axr, axg, axb) = plt.subplots(1,3, figsize=(21 ,7))

    slice_list.append(
        skimage.measure.block_reduce(
            slice_list[1], block_size=factor,
            func=numpy.mean, cval=0))

    rasterio.plot.show(slice_list[0], ax=axr, cmap='Reds', title='carbon')
    rasterio.plot.show(slice_list[1], ax=axg, cmap='Greens', title='forest base')
    rasterio.plot.show(slice_list[2], ax=axb, cmap='Blues', title='forest avg')
    plt.title(f'{slice_list[0].shape}, {slice_list[2].shape}')
    plt.show()


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

    n_points = 100
    sample_regions = generate_sample_points(
        countries_vector_path, target_box_wgs84,
        n_points, box_radius, country_filter_list=None)
    LOGGER.debug(sample_regions)
    for box in sample_regions:
        sample_rasters(raster_path_list, box)
        break


if __name__ == '__main__':
    main()
