"""Tracer code for regression models."""
import argparse
import collections
import logging
import pickle
import os
import sys
import time

from osgeo import gdal
import numpy
import pygeoprocessing
import rtree
from sklearn.linear_model import LassoLarsCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
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

EXPECTED_MAX_EDGE_EFFECT_KM = 2.0
HOLDBACK_PROPORTION = 0.2
MODEL_FIT_WORKSPACE = 'carbon_model'
POINTS_PER_STRIDE = 10000
N_POINT_SAMPLE_STRIDES = 2**6
N_POINTS = N_POINT_SAMPLE_STRIDES*POINTS_PER_STRIDE
POLY_ORDER = 3
MODEL_NAME = 'lsvr'


def generate_sample_points_for_carbon_model(
        n_points,
        baccini_raster_path_nodata,
        forest_mask_raster_path,
        independent_raster_path_nodata_list, max_min_lat,
        target_X_array_path,
        target_y_array_path,
        target_lng_lat_array_path,
        seed=None):
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
        target_X_array_path (str): path to X vector on disk,
        target_y_array_path (str): path to y vector on disk
        target_lng_lat_array_path (str): path to store lng/lat samples
        seed (int): seed for randomization

    Returns:
        None

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
        (baccini_raster_path_nodata[0], 1), offset_only=True, largest_block=0))
    gt_baccini = band_inv_gt_list[0][-2]
    baccini_lng_lat_bb_list = []
    LOGGER.debug(f'creating {len(offset_list)} index boxes')
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
        baccini_lng_lat_bb_list.append((index, bb_lng_lat, None))
    LOGGER.debug('creating the index all at once')
    baccini_memory_block_index = rtree.index.Index(baccini_lng_lat_bb_list)

    points_remaining = n_points
    lng_lat_vector = []
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
        lat_arr = numpy.arccos(2*v-1) * 180/numpy.pi - 90
        valid_mask = numpy.abs(lat_arr) <= max_min_lat

        window_index_to_point_list_map = collections.defaultdict(list)
        for lng, lat in zip(lng_arr[valid_mask], lat_arr[valid_mask]):
            if lat < -46:
                LOGGER.debug(f'{lat} is < -46')
            if time.time() - last_time > 5.0:
                LOGGER.debug(f'working ... {points_remaining} left')
                last_time = time.time()

            window_index = list(baccini_memory_block_index.intersection(
                (lng, lat, lng, lat)))[0]
            window_index_to_point_list_map[window_index].append((lng, lat))

        for window_index, point_list in window_index_to_point_list_map.items():
            if not point_list:
                continue

            # raster_index_to_array_list is an xoff, yoff, array list
            # TODO: loop through each point in point list
            for lng, lat in point_list:
                # check each array/raster and ensure it's not nodata or if it
                # is, set to the valid value
                working_sample_list = []
                valid_working_list = True
                for index, (raster_path, band, nodata, nodata_replace,
                            gt, inv_gt) in enumerate(band_inv_gt_list):
                    x, y = [int(v) for v in (
                        gdal.ApplyGeoTransform(inv_gt, lng, lat))]

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
                    lng_lat_vector.append((lng, lat))
                    y_vector.append(working_sample_list[0])
                    # first element is dep
                    # second element is forest mask -- don't include it
                    X_vector.append(working_sample_list[2:])

    numpy.savez_compressed(target_X_array_path, X_vector)
    numpy.savez_compressed(target_y_array_path, y_vector)
    numpy.savez_compressed(target_lng_lat_array_path, lng_lat_vector)


def build_model(
        X_vector_path_list, y_vector_path, n_arrays, target_model_path):
    """Create and test a model wn_lists
    Args:
        X_vector_path_list (list): list containing paths to npz files of
            10000 points each for the X vector
        y_vector_path_list (list): list containing paths to npz files of
            10000 points each for the y vector
        n_arrays (int): the nubmer of 10000 arrays to use in the model training
        target_model_path (str): path to file to save the regression model to

    Returns:
        (r^2 fit of training, r^2 fit of test)

    """
    poly_trans = PolynomialFeatures(POLY_ORDER, interaction_only=False)
    lasso_lars_cv = LassoLarsCV(
        n_jobs=-1, max_iter=100000, verbose=True, eps=1e-2)
    model_dict = {
        'lasso': Pipeline([
            ('poly_trans', poly_trans),
            ('Normalizer', Normalizer()),
            ('lasso_lars_cv', lasso_lars_cv),
         ]),

        'lsvr': Pipeline([
            ('poly_trans', poly_trans),
            ('StandardScaler', StandardScaler()),
            ('lsvr', LinearSVR(verbose=1, max_iter=1000000)),
        ]),

        'sgdr': Pipeline([
            ('poly_trans', poly_trans),
            ('Normalizer', Normalizer()),
            ('StandardScaler', StandardScaler()),
            ('Nystroem', Nystroem()),
            ('SGDRegressor', SGDRegressor(max_iter=100000)),
        ])
    }

    raw_X_vector = numpy.concatenate(
        [numpy.load(path)['arr_0'] for path in X_vector_path_list[0:n_arrays]])
    LOGGER.info('collect raw y vector')
    raw_y_vector = numpy.concatenate(
        [numpy.load(path)['arr_0'] for path in y_vector_path[0:n_arrays]])

    n_points = len(raw_X_vector)

    X_vector, test_X_vector, y_vector, test_y_vector = train_test_split(
        raw_X_vector, raw_y_vector,
        shuffle=False, test_size=0.2)

    LOGGER.info(f'doing fit on {n_points} points {MODEL_NAME}')
    model = model_dict[MODEL_NAME]
    model.fit(X_vector, y_vector)
    r_squared = model.score(X_vector, y_vector)
    r_squared_test = model.score(test_X_vector, test_y_vector)

    with open(target_model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    return r_squared, r_squared_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model maker')
    parser.add_argument(
        '--max_min_lat', type=float, default=40.0,
        help='Min/max lat to cutoff')
    parser.add_argument(
        '--n_workers', type=int, default=1, help='number of taskgraph workers')
    args = parser.parse_args()

    LOGGER.info('initalizing')
    task_graph = taskgraph.TaskGraph(
        BASE_DATA_DIR, args.n_workers, 15.0)

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
        func=pygeoprocessing.raster_calculator,
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

    try:
        array_cache_dir = os.path.join(BASE_DATA_DIR, 'array_cache')
        os.makedirs(array_cache_dir)
    except OSError:
        pass

    LOGGER.info(f'create {N_POINTS} points')
    task_xy_vector_list = []
    for point_stride in range(N_POINT_SAMPLE_STRIDES):
        lat_lng_array_path = os.path.join(
            array_cache_dir, f'lng_lat_array_{point_stride}.npz')
        target_X_array_path = os.path.join(
            array_cache_dir, f'X_array_{point_stride}.npz')
        target_y_array_path = os.path.join(
            array_cache_dir, f'y_array_{point_stride}.npz')
        generate_point_task = task_graph.add_task(
            func=generate_sample_points_for_carbon_model,
            args=(
                POINTS_PER_STRIDE,
                (baccini_10s_2014_biomass_path, baccini_nodata),
                forest_mask_raster_path,
                raster_path_nodata_replacement_list + convolution_raster_list,
                args.max_min_lat, target_X_array_path, target_y_array_path,
                lat_lng_array_path),
            kwargs={'seed': point_stride},
            ignore_path_list=[
                target_X_array_path, target_y_array_path, lat_lng_array_path],
            target_path_list=[target_X_array_path, target_y_array_path],
            store_result=True,
            task_name=(
                f'calculating points {point_stride*POINTS_PER_STRIDE} to '
                f'{(point_stride+1)*POINTS_PER_STRIDE}'))
        task_xy_vector_list.append(
            (generate_point_task, target_X_array_path, target_y_array_path))

    feature_name_list = [
        os.path.basename(os.path.splitext(path_ndr[0])[0])
        for path_ndr in (
            raster_path_nodata_replacement_list +
            convolution_raster_list)]

    model_dir = 'models'
    try:
        os.makedirs(model_dir)
    except OSError:
        pass

    build_model_task_list = []
    # TODO: note this is hard-coded to be 10,000 to 100,000 points
    for test_strides in range(1, N_POINT_SAMPLE_STRIDES):
        n_points = test_strides*POINTS_PER_STRIDE
        model_filename = os.path.join(
            model_dir,
            #f'carbon_model_svmlasso_lars_cv_poly_{POLY_ORDER}_'
            f'carbon_model_{MODEL_NAME}_'
            f'poly_{POLY_ORDER}_'
            f'{EXPECTED_MAX_EDGE_EFFECT_KM}_gf_{n_points}_pts.mod')
        LOGGER.info(f'build {model_filename} model')
        X_vector_path_list = []
        y_vector_path = []

        local_point_task_list = []
        for (generate_point_task, target_X_array_path,
                target_y_array_path) in task_xy_vector_list[0:test_strides]:
            local_point_task_list.append(generate_point_task)
            X_vector_path_list.append(target_X_array_path)
            y_vector_path.append(target_y_array_path)

        build_model_task = task_graph.add_task(
            func=build_model,
            args=(
                X_vector_path_list, y_vector_path, test_strides,
                model_filename),
            store_result=True,
            target_path_list=[model_filename],
            dependent_task_list=local_point_task_list,
            # dependent_task_list=local_point_task_list + [
            #     v[1] for v in build_model_task_list[-1::]],
            task_name=f'build model for {n_points} points')
        build_model_task_list.append((n_points, build_model_task))

    with open(f'fit_test_{N_POINTS}_{MODEL_NAME}_p{POLY_ORDER}_points.csv', 'w') as fit_file:
        fit_file.write(f'n_points,r_squared,r_squared_test\n')

    for n_points, build_model_task in build_model_task_list:
        r_2_fit, r_2_test_fit = build_model_task.get()
        with open(f'fit_test_{N_POINTS}_{MODEL_NAME}_p{POLY_ORDER}_points.csv', 'a') as fit_file:
            fit_file.write(f'{n_points},{r_2_fit},{r_2_test_fit}\n')
        LOGGER.info(f'{n_points},{r_2_fit},{r_2_test_fit}')

    LOGGER.debug('all done!')
    task_graph.close()
    task_graph.join()
