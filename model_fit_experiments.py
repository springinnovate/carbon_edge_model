"""Tracer code for regression models."""
import argparse
import glob
import logging
import multiprocessing
import os
import sys

from osgeo import gdal
import numpy
import retrying
import sklearn
import taskgraph

import model_files

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)

LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)

BASE_DATA_DIR = 'base_data_no_humans_please'

MODEL_FIT_WORKSPACE = 'carbon_model'


def generate_sample_points(n_points, valid_raster_path, max_min_lat=40):
    """Generate a set of lat/lng points that are evenly distributed."""
    raster = gdal.OpenEx(valid_raster_path, gdal.OF_RASTER)
    band = raster.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    gt = raster.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)

    points_remaining = n_points
    valid_points = []
    while points_remaining > 0:
        # from https://mathworld.wolfram.com/SpherePointPicking.html
        u = numpy.random.random((points_remaining,))
        v = numpy.random.random((points_remaining,))
        # pick between -180 and 180
        lng_arr = (2.0 * numpy.pi * u) * 180/numpy.pi - 180
        lat_arr = numpy.arccos(2*v-1) * 180/numpy.pi
        valid_mask = numpy.abs(lat_arr) <= max_min_lat

        for lng, lat in zip(lng_arr[valid_mask], lat_arr[valid_mask]):
            x, y = [int(v) for v in gdal.ApplyGeoTransform(inv_gt, lng, lat)]
            val = band.ReadAsArray(x, y, 1, 1)[0, 0]
            if val != nodata:
                valid_points.append((lng, lat))

        points_remaining -= numpy.count_nonzero(valid_mask)
    return valid_points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model maker')
    parser.add_argument(
        '--raster_directory', help=(
            'Path to directory containing rasters to build a model from'))
    args = parser.parse_args()

    task_graph = taskgraph.TaskGraph(
        BASE_DATA_DIR,
        -1,
        5.0)

    model_files.fetch_data(BASE_DATA_DIR, task_graph)
    LOGGER.debug('closing and joining taskgraph')
    task_graph.close()
    task_graph.join()

    baccini_10s_2014_biomass_path = os.path.join(
        BASE_DATA_DIR, os.path.basename(model_files.BACCINI_10s_2014_BIOMASS_URI))
    sample_points = generate_sample_points(30, baccini_10s_2014_biomass_path)

    LOGGER.debug(sample_points)

    LOGGER.debug('all done!')
