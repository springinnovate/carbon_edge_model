"""Tracer code for regression models."""
import argparse
import glob
import logging
import multiprocessing
import os
import sys
import time

from osgeo import gdal
import numpy
import pygeoprocessing
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


def generate_sample_points(
        n_points, dependent_raster_path_nodata,
        independent_raster_path_nodata_list, max_min_lat):
    """Generate a set of lat/lng points that are evenly distributed.

    Args:
        n_points (int): number of points to sample
        depedent_raster_path_nodata (tuple): tuple of path to dependent
            variable raster with expected nodata value.
        independent_raster_path_nodata_list (list): list of (path, nodata)
            tuples.
        max_min_lat (float): absolute maximum latitude allowed in a sampled
            point.

    Returns:
        List of (lng, lat, sample_list) where sample_list contains values
            ordered from the dependent variable through the independent
            variables.

    """
    band_inv_gt_list = []
    raster_list = []
    LOGGER.debug("build band list")
    for raster_path, nodata, nodata_replace in [
            dependent_raster_path_nodata + (None,)] + \
            independent_raster_path_nodata_list:
        raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
        raster_list.append(raster)
        band = raster.GetRasterBand(1)
        gt = raster.GetGeoTransform()
        inv_gt = gdal.InvGeoTransform(gt)
        band_inv_gt_list.append((raster_path, band, nodata, nodata_replace, inv_gt))
        raster = None
        band = None

    points_remaining = n_points
    valid_points = []
    LOGGER.debug(f'build {n_points}')
    last_time = time.time()
    while points_remaining > 0:
        # from https://mathworld.wolfram.com/SpherePointPicking.html
        u = numpy.random.random((points_remaining,))
        v = numpy.random.random((points_remaining,))
        # pick between -180 and 180
        lng_arr = (2.0 * numpy.pi * u) * 180/numpy.pi - 180
        lat_arr = numpy.arccos(2*v-1) * 180/numpy.pi
        valid_mask = numpy.abs(lat_arr) <= max_min_lat

        for lng, lat in zip(lng_arr[valid_mask], lat_arr[valid_mask]):
            if time.time() - last_time > 5.0:
                LOGGER.debug(f'working... {points_remaining} left')
                last_time = time.time()
            working_sample_list = []
            valid_working_list = True
            for raster_path, band, nodata, nodata_replace, inv_gt in band_inv_gt_list:
                x, y = [int(v) for v in gdal.ApplyGeoTransform(
                    inv_gt, lng, lat)]
                val = band.ReadAsArray(x, y, 1, 1)[0, 0]
                if nodata is None or not numpy.isclose(val, nodata):
                    working_sample_list.append(val)
                elif nodata_replace is not None:
                    working_sample_list.append(nodata_replace)
                else:
                    # nodata value, skip
                    LOGGER.debug(
                        f'got invalid value: {val} from {raster_path}')
                    valid_working_list = False
                    break
            if valid_working_list:
                points_remaining -= 1
                valid_points.append((lng, lat, working_sample_list))
    return valid_points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model maker')
    parser.add_argument(
        '--n_points', type=int, required=True,
        help='Number of points to sample')
    parser.add_argument(
        '--max_min_lat', type=float, default=40.0,
        help='Min/max lat to cutoff')
    args = parser.parse_args()

    task_graph = taskgraph.TaskGraph(
        BASE_DATA_DIR,
        -1,
        5.0)

    raster_path_nodata_replacement_list = (
        model_files.fetch_data(BASE_DATA_DIR, task_graph))
    LOGGER.debug(f'raster files: {raster_path_nodata_replacement_list}')
    LOGGER.debug('closing and joining taskgraph')

    task_graph.close()
    task_graph.join()

    baccini_10s_2014_biomass_path = os.path.join(
        BASE_DATA_DIR, os.path.basename(
            model_files.BACCINI_10s_2014_BIOMASS_URI))
    baccini_nodata = pygeoprocessing.get_raster_info(
        baccini_10s_2014_biomass_path)['nodata'][0]

    sample_points = generate_sample_points(
        args.n_points, (baccini_10s_2014_biomass_path, baccini_nodata),
        raster_path_nodata_replacement_list,
        args.max_min_lat)

    LOGGER.debug(f'generated {len(sample_points)}')

    LOGGER.debug('all done!')
