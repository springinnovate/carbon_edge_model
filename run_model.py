"""Execute carbon model on custom forest edge data."""
import argparse
import glob
import pickle
import logging
import os
import multiprocessing
import shutil

from osgeo import gdal
from ecoshard import geoprocessing
from ecoshard import taskgraph
import numpy
from ecoshard.geoprocessing import DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS

import gaussian_filter_rasters
import train_regression_model

# GLOBAL_BOUNDING_BOX_TUPLE = (
#     'global_eckert_iv_bb', [-16921202, -8460601, 16921797, 8461398])

GLOBAL_BOUNDING_BOX_TUPLE = (
    'amazon_region', [-6215550, -773278, -4310775, 115131])

ECKERT_PIXEL_SIZE = (90, -90)
WORLD_ECKERT_IV_WKT = """PROJCRS["unknown",
    BASEGEOGCRS["GCS_unknown",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1]],
            ID["EPSG",6326]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["unnamed",
        METHOD["Eckert IV"],
        PARAMETER["Longitude of natural origin",0,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8802]],
        PARAMETER["False easting",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8806]],
        PARAMETER["False northing",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8807]]],
    CS[Cartesian,2],
        AXIS["(E)",east,
            ORDER[1],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]],
        AXIS["(N)",north,
            ORDER[2],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]]]"""
ZSTD_CREATION_TUPLE = ('GTIFF', (
    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
    'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'NUM_THREADS=1'))


logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)
logging.getLogger('fiona').setLevel(logging.WARN)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARN)
logging.getLogger('matplotlib').setLevel(logging.WARN)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARN)

REGRESSION_TARGET_NODATA = -1


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='execute carbon model')
    parser.add_argument(
        'carbon_model_path', type=str, help='path to trained carbon model')
    parser.add_argument(
        'forest_cover_path', type=str, help='path to forest mask')
    parser.add_argument(
        '--predictor_raster_dir', type=str, default='',
        help='path to directory containing base data for model')
    parser.add_argument(
        '--prefix', type=str, help='string to add to output filename')
    parser.add_argument(
        '--pre_warp_dir', type=str,
        help='if defined uses matching files as prewarps')
    args = parser.parse_args()
    regression_carbon_model(
        args.carbon_model_path, args.forest_cover_path,
        args.predictor_raster_dir, args.prefix, args.pre_warp_dir)


def _pre_warp_rasters(
        task_graph, bounding_box_tuple, carbon_model_path,
        predictor_raster_dir, pre_warp_dir):
    """Create a pre-warping of base rasters against ECKERT bounding box.

    Args:
        task_graph (TaskGraph): object to avoid recomputation
        bounding_box_tuple (tuple): a str/bounding box tuple with the str
            being used for the prefix of target files
        carbon_model_path (str): path to pickled carbon model.
        task_graph (TaskGraph): used to parallelize pre-warped rasters.
        predictor_raster_dir (str): path to directory with predictor rasters
            in it.
        pre_warp_dir (str): path to directory to dump warped files.

    Return:
        None.
    """
    predictor_id_path_list = []
    missing_predictor_list = []
    with open(carbon_model_path, 'rb') as model_file:
        model = pickle.load(model_file).copy()

    LOGGER.debug(f'predictor id is {model["gf_forest_id"]} {model["gf_forest_id"] in model["predictor_list"]}')
    for predictor_id in model['predictor_list']:
        if model['gf_forest_id'] == predictor_id:
            continue
        predictor_path = os.path.join(
            predictor_raster_dir, f'{predictor_id}.tif')
        if not os.path.exists(predictor_path):
            possible_predictor_match = glob.glob(
                '%s*%s' % os.path.splitext(predictor_path))
            if len(possible_predictor_match) == 1:
                LOGGER.warn(
                    f'exact match not found for {predictor_path} using '
                    f'{possible_predictor_match[0]} instead')
                predictor_path = possible_predictor_match[0]
            else:
                missing_predictor_list.append(
                    f'{predictor_id}: {predictor_path}')
        predictor_id_path_list.append((predictor_id, predictor_path))
    if missing_predictor_list:
        predictor_str = "\n".join(missing_predictor_list)
        raise ValueError(
            f'missing the following predictor rasters:\n{predictor_str}')

    os.makedirs(pre_warp_dir, exist_ok=True)
    for predictor_id, predictor_path in predictor_id_path_list:
        warped_predictor_path = os.path.join(
            pre_warp_dir,
            f'warped_{predictor_id}_{bounding_box_tuple[0]}.tif')
        warp_task = task_graph.add_task(
            func=geoprocessing.warp_raster,
            args=(predictor_path, ECKERT_PIXEL_SIZE,
                  warped_predictor_path, 'near'),
            kwargs={
                'target_bb': bounding_box_tuple[1],
                'target_projection_wkt': WORLD_ECKERT_IV_WKT,
                'n_threads': multiprocessing.cpu_count(),
                'working_dir': pre_warp_dir,
                'raster_driver_creation_tuple': ZSTD_CREATION_TUPLE
            },
            target_path_list=[warped_predictor_path],
            task_name=f'warp {predictor_path}')
    task_graph.join()


def regression_carbon_model(
    carbon_model_path, bounding_box_tuple, forest_cover_path,
        predictor_raster_dir, pre_warp_dir=None,
        target_result_path=None):
    """
    Run carbon model.

    Args:
        carbon_model_path (str): path to picked sklearn trained model
        bounding_box_tuple (tuple): str/bounding box pair to ensure
            clip of rasters
        forest_cover_path (str): path to raster with 1 indicating where
            the forest cover is
        predictor_raster_dir (str): if not empty string, looks here for
            predictor rasters defined by carbon model
        pre_warp_dir (str): path to directory containing pre-warped
            predictor rasters
        target_result_path (str): path to target raster

    Returns:
        None
    """
    LOGGER.info(f'load model at {carbon_model_path}')
    with open(carbon_model_path, 'rb') as model_file:
        model = pickle.load(model_file).copy()
    LOGGER.info(f'ensure raster base data are present')
    missing_predictor_list = []
    predictor_id_path_list = []
    for predictor_id in model['predictor_list']:
        predictor_path = os.path.join(
            predictor_raster_dir, f'{predictor_id}.tif')
        predictor_id_path_list.append((predictor_id, predictor_path))
        if model['gf_forest_id'] == predictor_id:
            continue
        if not os.path.exists(predictor_path):
            missing_predictor_list.append(
                f'{predictor_id}: {predictor_path}')
    if missing_predictor_list:
        predictor_str = "\n".join(missing_predictor_list)
        raise ValueError(
            f'missing the following predictor rasters:\n{predictor_str}')
    LOGGER.info(f'all found: {predictor_id_path_list}')

    workspace_dir = f'''workspace_{os.path.splitext(os.path.basename(
        forest_cover_path))[0]}'''
    os.makedirs(workspace_dir, exist_ok=True)

    task_graph = taskgraph.TaskGraph(
        workspace_dir, min(
            multiprocessing.cpu_count(),
            len(predictor_id_path_list)))
    forest_mask_raster_info = geoprocessing.get_raster_info(forest_cover_path)
    if abs(forest_mask_raster_info['pixel_size'][0]) < 3:
        LOGGER.info(
            f'{forest_cover_path} appears to not be projected in meters with '
            f'a pixel size of {forest_mask_raster_info["pixel_size"]}, '
            f'projecting into eckert IV {ECKERT_PIXEL_SIZE}')
        projected_forest_cover_path = os.path.join(
            workspace_dir, '%s_projected%s' % os.path.splitext(
                os.path.basename(forest_cover_path)))
        task_graph.add_task(
            func=geoprocessing.warp_raster,
            args=(forest_cover_path, ECKERT_PIXEL_SIZE,
                  projected_forest_cover_path, 'near'),
            kwargs={
                'target_bb': bounding_box_tuple[1],
                'target_projection_wkt': WORLD_ECKERT_IV_WKT,
                'working_dir': workspace_dir,
                'n_threads': multiprocessing.cpu_count(),
                'raster_driver_creation_tuple': ZSTD_CREATION_TUPLE},
            target_path_list=[projected_forest_cover_path],
            task_name=f'project {projected_forest_cover_path}')
        task_graph.join()
        forest_cover_path = projected_forest_cover_path
    forest_mask_raster_info = geoprocessing.get_raster_info(forest_cover_path)

    LOGGER.info('gaussian filter forest cover')

    LOGGER.info('clip input rasters to forest cover')
    aligned_predictor_path_list = []
    gf_index = None
    for predictor_id, predictor_path in predictor_id_path_list:
        predictor_is_forest = (model['gf_forest_id'] == predictor_id)
        if predictor_is_forest:
            # override the model's pretrained gf with one passed in
            predictor_path = forest_cover_path

            gf_forest_cover_path = os.path.join(
                workspace_dir, f'''{model["gf_size"]}_{
                os.path.basename(forest_cover_path)}''')
            task_graph.add_task(
                func=gaussian_filter_rasters.filter_raster,
                args=((forest_cover_path, 1), model['gf_size'],
                      gf_forest_cover_path),
                target_path_list=[gf_forest_cover_path],
                task_name=f'gaussian filter {gf_forest_cover_path}')
            gf_index = len(aligned_predictor_path_list)
            aligned_predictor_path_list.append(gf_forest_cover_path)
            continue

        if pre_warp_dir:
            warped_predictor_path = os.path.join(
                pre_warp_dir,
                f'warped_{os.path.basename(predictor_path)}')
        else:
            warped_predictor_path = os.path.join(
                workspace_dir, f'warped_{os.path.basename(predictor_path)}')

        warp_task_list = []
        if not(pre_warp_dir and os.path.exists(warped_predictor_path)):
            warp_task = task_graph.add_task(
                func=geoprocessing.warp_raster,
                args=(predictor_path, forest_mask_raster_info['pixel_size'],
                      warped_predictor_path, 'nearest'),
                kwargs={
                    'target_bb': forest_mask_raster_info['bounding_box'],
                    'target_projection_wkt': forest_mask_raster_info['projection_wkt'],
                    'working_dir': workspace_dir,
                    'raster_driver_creation_tuple': ZSTD_CREATION_TUPLE
                },
                target_path_list=[warped_predictor_path],
                task_name=f'warp {predictor_path}')
            warp_task_list.append(warp_task)
            aligned_predictor_path_list.append(warped_predictor_path)

    task_graph.join()

    LOGGER.info('apply model')

    if target_result_path is None:
        target_result_path = f'''{os.path.basename(os.path.splitext(
            forest_cover_path)[0])}_std_forest_edge_result.tif'''
    forest_edge_task = task_graph.add_task(
        func=geoprocessing.raster_calculator,
        args=(
            [(path, 1) for path in aligned_predictor_path_list] +
            [(geoprocessing.get_raster_info(path)['nodata'][0], 'raw')
             for path in aligned_predictor_path_list] + [(model, 'raw')],
            _apply_model, target_result_path,
            gdal.GDT_Float32, REGRESSION_TARGET_NODATA),
        kwargs={
            'largest_block': 2**22,
            'raster_driver_creation_tuple': (
                'GTiff', DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[1])},
        target_path_list=[target_result_path],
        task_name=f'regression model for {target_result_path}')
    forest_edge_task.join()

    task_graph.join()
    task_graph.close()
    del task_graph
    shutil.rmtree(workspace_dir, ignore_errors=True)


def _apply_model(*raster_nodata_array):
    """Scikit-learn pre-trained model determines order."""
    n = len(raster_nodata_array) // 2
    raster_array = raster_nodata_array[0:n]
    nodata_array = raster_nodata_array[n:2*n]
    model = raster_nodata_array[2*n]
    valid_mask = numpy.all(
        [~numpy.isclose(array, nodata) for array, nodata in
         zip(raster_array, nodata_array)], axis=(0,))
    result = numpy.full(valid_mask.shape, REGRESSION_TARGET_NODATA)
    value_list = numpy.asarray([
        array[valid_mask] for array in raster_array])
    value_list = value_list.transpose()
    if value_list.shape[0] > 0:
        result[valid_mask] = train_regression_model.clip_to_range(
            model['model'].predict(value_list), 10, 400)
    return result

if __name__ == '__main__':
    main()
