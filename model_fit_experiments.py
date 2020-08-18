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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor

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
EXPECTED_MAX_EDGE_EFFECT_KM = 3.0
MODEL_FIT_WORKSPACE = 'carbon_model'


def generate_sample_points_for_carbon_model(
        n_points,
        baccini_raster_path_nodata,
        forest_mask_raster_path,
        independent_raster_path_nodata_list, max_min_lat, seed=None):
    """Generate a set of lat/lng points that are evenly distributed.

    Args:
        n_points (int): number of points to sample
        baccini_raster_path_nodata (tuple): tuple of path to dependent
            variable raster with expected nodata value.
        independent_raster_path_nodata_list (list): list of
            (path, nodata, nodata_replace) tuples.
        max_min_lat (float): absolute maximum latitude allowed in a sampled
            point.
        seed (int): seed for randomization

    Returns:
        Tuple of (lng_lat_list, X_vector, y_vector).

    """
    if seed is not None:
        numpy.random.seed(seed)
    band_inv_gt_list = []
    raster_list = []
    LOGGER.debug("build band list")
    for raster_path, nodata, nodata_replace in [
            baccini_raster_path_nodata + (None,),
            (forest_mask_raster_path, 0, None)] + \
            independent_raster_path_nodata_list:
        raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
        raster_list.append(raster)
        band = raster.GetRasterBand(1)
        gt = raster.GetGeoTransform()
        inv_gt = gdal.InvGeoTransform(gt)
        band_inv_gt_list.append(
            (raster_path, band, nodata, nodata_replace, inv_gt))
        raster = None
        band = None

    points_remaining = n_points
    valid_points = []
    X_vector = []
    y_vector = []
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
                LOGGER.debug(f'working ... {points_remaining} left')
                last_time = time.time()
            working_sample_list = []
            valid_working_list = True
            for index, (raster_path, band, nodata, nodata_replace, inv_gt) in \
                    enumerate(band_inv_gt_list):
                x, y = [int(v) for v in gdal.ApplyGeoTransform(
                    inv_gt, lng, lat)]
                val = band.ReadAsArray(x, y, 1, 1)[0, 0]
                if nodata is None or not numpy.isclose(val, nodata):
                    working_sample_list.append(val)
                elif nodata_replace is not None:
                    working_sample_list.append(nodata_replace)
                else:
                    # nodata value, skip
                    valid_working_list = False
                    break
            if valid_working_list:
                points_remaining -= 1
                valid_points.append((lng, lat, working_sample_list))
                y_vector.append(working_sample_list[0])  # first element is dep
                # second element is forest mask -- don't include it
                X_vector.append(working_sample_list[2:])
    return valid_points, X_vector, y_vector


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model maker')
    parser.add_argument(
        '--n_points', type=int, required=True,
        help='Number of points to sample')
    parser.add_argument(
        '--max_min_lat', type=float, default=40.0,
        help='Min/max lat to cutoff')
    parser.add_argument(
        '--n_workers', type=int, default=1, help='number of taskgraph workers')
    args = parser.parse_args()

    task_graph = taskgraph.TaskGraph(BASE_DATA_DIR, args.n_workers, 5.0)

    raster_path_nodata_replacement_list = (
        model_files.fetch_data(BASE_DATA_DIR, task_graph))
    LOGGER.debug(f'raster files: {raster_path_nodata_replacement_list}')
    task_graph.join()

    LOGGER.debug('create convolutions')
    esa_lulc_raster_path = os.path.join(
        BASE_DATA_DIR, os.path.basename(model_files.ESA_LULC_URI))
    convolution_raster_list = model_files.create_convolutions(
        task_graph, esa_lulc_raster_path, EXPECTED_MAX_EDGE_EFFECT_KM,
        BASE_DATA_DIR)

    task_graph.join()

    baccini_10s_2014_biomass_path = os.path.join(
        BASE_DATA_DIR, os.path.basename(
            model_files.BACCINI_10s_2014_BIOMASS_URI))
    baccini_nodata = pygeoprocessing.get_raster_info(
        baccini_10s_2014_biomass_path)['nodata'][0]

    forest_mask_raster_path = os.path.join(BASE_DATA_DIR, 'forest_mask.tif')

    point_task_dict = {}
    for seed_val, data_type in [(1, 'training'), (2, 'validation')]:
        generate_point_task = task_graph.add_task(
            func=generate_sample_points_for_carbon_model,
            args=(
                args.n_points, (baccini_10s_2014_biomass_path, baccini_nodata),
                forest_mask_raster_path,
                raster_path_nodata_replacement_list + convolution_raster_list,
                args.max_min_lat),
            kwargs={'seed': seed_val},
            task_name=f'predict with seed {seed_val}')
        point_task_dict[data_type] = generate_point_task

    LOGGER.debug('fit model')
    models_to_test = [
        #('linear regression', LinearRegression),
        ('lasso lars CV', LassoLarsCV(n_jobs=-1, max_iter=5000, verbose=True)),
        #('lasso', Lasso),
        #('lasso CV', LassoCV),
        #('ridge', Ridge),
        ('ridge CV', RidgeCV(normalize=True)),
        ('SGDRegressor', SGDRegressor(max_iter=10000, verbose=True,)),
        ]

    for model_name, model_object in models_to_test:
        LOGGER.info(f'fitting {model_name} model')
        _, X_vector, y_vector = point_task_dict['training'].get()
        model = model_object.fit(X_vector, y_vector)
        _, valid_X_vector, valid_y_vector = point_task_dict['validation'].get()
        coeff_id_list = sorted([
            (coeff, os.path.basename(os.path.splitext(path_ndr[0])[0]))
            for coeff, path_ndr in zip(
                model.coef_,
                raster_path_nodata_replacement_list +
                convolution_raster_list)], reverse=True,
            key=lambda v: abs(v[0]))
        LOGGER.debug(f'validate {model_name}')
        LOGGER.info(
            f'R^2 fit: {model.score(X_vector, y_vector)}\n'
            f'''validation data R^2: {
                model.score(valid_X_vector, valid_y_vector)}'''
            f"coeff:\n" + '\n'.join([str(x) for x in coeff_id_list]) +
            f'y int: {model.intercept_}\n')

    task_graph.close()
    task_graph.join()
    LOGGER.debug('all done!')
