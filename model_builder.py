"""Tracer code for regression models."""
import argparse
import collections
import logging
import os
import sys
import time

from osgeo import gdal
import numpy
import pygeoprocessing
import rtree
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures

import taskgraph

import carbon_model_data
from carbon_model_data import BASE_DATA_DIR
from utils import esa_to_carbon_model_landcover_types

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)

LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)

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
        forest_mask_raster_path (str): path to the forest mask, this model
            should only generate a model for valid forest points.
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
            (raster_path, band, nodata, nodata_replace, gt, inv_gt))
        raster = None
        band = None

    # build a spatial index for efficient fetching of points later
    LOGGER.info('build baccini iterblocks spatial index')
    offset_list = list(pygeoprocessing.iterblocks(
        (baccini_raster_path_nodata[0], 1), offset_only=True))
    baccini_memory_block_index = rtree.index.Index()
    gt_baccini = band_inv_gt_list[0][-2]
    baccini_lng_lat_bb_list = []
    for index, offset_dict in enumerate(offset_list):
        bb_lng_lat = [
            coord for coord in (
                gdal.ApplyGeoTransform(
                    gt_baccini,
                    offset_dict['xoff'],
                    offset_dict['yoff']+offset_dict['win_ysize']) +
                gdal.ApplyGeoTransform(
                    gt_baccini,
                    offset_dict['xoff']+offset_dict['win_xsize'],
                    offset_dict['yoff']))]
        baccini_lng_lat_bb_list.append(bb_lng_lat)
        baccini_memory_block_index.insert(index, bb_lng_lat)

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

        window_index_to_point_list_map = collections.defaultdict(list)
        for lng, lat in zip(lng_arr[valid_mask], lat_arr[valid_mask]):
            if time.time() - last_time > 5.0:
                LOGGER.debug(f'working ... {points_remaining} left')
                last_time = time.time()

            window_index = list(baccini_memory_block_index.intersection(
                (lng, lat, lng, lat)))[0]
            window_index_to_point_list_map[window_index].append((lng, lat))

        for window_index, point_list in window_index_to_point_list_map.items():
            if not point_list:
                continue
            # load all raster blocks
            raster_index_to_array_list = []
            for index, (raster_path, band, nodata, nodata_replace,
                        gt, inv_gt) in enumerate(band_inv_gt_list):
                if time.time() - last_time > 5.0:
                    LOGGER.debug(f'working ... {points_remaining} left {index}')
                    last_time = time.time()
                lng_min, lat_min, lng_max, lat_max = baccini_lng_lat_bb_list[
                    window_index]
                x_min, y_min = [int(v) for v in (
                    gdal.ApplyGeoTransform(inv_gt, lng_min, lat_max))]
                x_max, y_max = [int(v) for v in (
                    gdal.ApplyGeoTransform(inv_gt, lng_max, lat_min))]

                try:
                    raster_index_to_array_list.append((
                        raster_path, x_min, y_min, nodata, nodata_replace, inv_gt,
                        band.ReadAsArray(
                            x_min, y_min, x_max-x_min, y_max-y_min)))
                except Exception:
                    LOGGER.exception(
                        f'error on {raster_path}'
                        f'\n{lng_min}, {lat_min}, {lng_max}, {lat_max}'
                        f'\n{x_min}, {y_min}, {x_max}, {y_max}, {x_max-x_min}, {y_max-y_min}')
                    raise

            # raster_index_to_array_list is an xoff, yoff, array list
            # TODO: loop through each point in point list
            for lng, lat in point_list:
                # check each array/raster and ensure it's not nodata or if it
                # is, set to the valid value
                working_sample_list = []
                valid_working_list = True
                for raster_path, xoff, yoff, nodata, nodata_replace, inv_gt, array in \
                        raster_index_to_array_list:
                    x, y = [int(v) for v in gdal.ApplyGeoTransform(
                        inv_gt, lng, lat)]

                    val = array[x-xoff, y-yoff]

                    if nodata is None or not numpy.isclose(val, nodata):
                        LOGGER.debug(
                            f'got nodata {nodata} {val} for {xoff} {yoff} on {raster_path}')
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
                    y_vector.append(working_sample_list[0])
                    # first element is dep
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

    LOGGER.info('initalizing')
    task_graph = taskgraph.TaskGraph(
        BASE_DATA_DIR, args.n_workers, 5.0)

    LOGGER.info('fetch model data')
    raster_path_nodata_replacement_list = carbon_model_data.fetch_data(
        BASE_DATA_DIR, task_graph)
    LOGGER.debug(f'raster files: {raster_path_nodata_replacement_list}')
    task_graph.join()

    LOGGER.info('create ESA to carbon model landcover type mask')
    esa_lulc_raster_path = os.path.join(
        BASE_DATA_DIR,
        os.path.basename(carbon_model_data.ESA_LULC_URI))
    esa_to_carbon_model_landcover_type_raster_path = os.path.join(
        BASE_DATA_DIR, 'esa_carbon_model_landcover_types.tif')
    create_carbon_lancover_mask_task = task_graph.add_task(
        func=pygeoprocessing.multiprocessing.raster_calculator,
        args=(
            [(esa_lulc_raster_path, 1)],
            esa_to_carbon_model_landcover_types._reclassify_esa_vals_op,
            esa_to_carbon_model_landcover_type_raster_path, gdal.GDT_Byte,
            None),
        target_path_list=[esa_to_carbon_model_landcover_type_raster_path],
        task_name='create carbon model land cover masks from ESA')
    create_carbon_lancover_mask_task.join()

    LOGGER.debug('create convolutions')
    convolution_raster_list = carbon_model_data.create_convolutions(
        esa_to_carbon_model_landcover_type_raster_path,
        EXPECTED_MAX_EDGE_EFFECT_KM, BASE_DATA_DIR, task_graph)

    task_graph.join()

    baccini_10s_2014_biomass_path = os.path.join(
        BASE_DATA_DIR, os.path.basename(
            carbon_model_data.BACCINI_10s_2014_BIOMASS_URI))
    baccini_nodata = pygeoprocessing.get_raster_info(
        baccini_10s_2014_biomass_path)['nodata'][0]

    forest_mask_raster_path = os.path.join(BASE_DATA_DIR, 'forest_mask.tif')

    point_task_dict = {}
    for n_points, seed_val, data_type in [
            (args.n_points, 1, 'training'),
            (int(args.n_points * 0.2), 2, 'validation')]:
        generate_point_task = task_graph.add_task(
            func=generate_sample_points_for_carbon_model,
            args=(
                args.n_points, (baccini_10s_2014_biomass_path, baccini_nodata),
                forest_mask_raster_path,
                raster_path_nodata_replacement_list + convolution_raster_list,
                args.max_min_lat),
            kwargs={'seed': seed_val},
            store_result=True,
            task_name=f'predict with seed {seed_val}')
        point_task_dict[data_type] = generate_point_task

    LOGGER.debug('fit model')
    poly = PolynomialFeatures(2)
    models_to_test = [
        #('linear regression', LinearRegression),
        #('LassoLarsIC', LassoLarsIC(max_iter=100000, verbose=True)),
        ('LassoLarsCV', LassoLarsCV(n_jobs=-1, max_iter=100000, verbose=True)),
        #('lasso', Lasso),
        #('lasso CV', LassoCV),
        #('ridge', Ridge),
        #('ridge CV', RidgeCV(normalize=True)),
        #('SGDRegressor', SGDRegressor(max_iter=10000, verbose=True,)),
        ]

    feature_name_list = [
            os.path.basename(os.path.splitext(path_ndr[0])[0])
            for path_ndr in (
                raster_path_nodata_replacement_list +
                convolution_raster_list)]

    for model_name, model_object in models_to_test:
        LOGGER.info(f'fitting {model_name} model')
        _, X_vector, y_vector = point_task_dict['training'].get()
        X_vector_transform = poly.fit_transform(X_vector)
        model = model_object.fit(X_vector_transform, y_vector)
        _, valid_X_vector, valid_y_vector = point_task_dict['validation'].get()
        coeff_id_list = sorted(zip(
            model.coef_, poly.get_feature_names(feature_name_list)),
            key=lambda v: abs(v[0]))
        LOGGER.info(
            f"coeff:\n" + '\n'.join([str(x) for x in coeff_id_list]) +
            f'R^2 fit: {model.score(X_vector_transform, y_vector)}\n'
            f'''validation data R^2: {
                model.score(poly.fit_transform(valid_X_vector), valid_y_vector)}'''
            f'y int: {model.intercept_}\n'
            )

    task_graph.close()
    task_graph.join()
    LOGGER.debug('all done!')
