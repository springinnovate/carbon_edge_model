"""Execute carbon model on custom forest edge data."""
import argparse
import pickle
import logging
import os

from osgeo import gdal
from ecoshard import geoprocessing
from ecoshard import taskgraph
import numpy

import gaussian_filter_rasters
import train_regression_model

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)
logging.getLogger('fiona').setLevel(logging.WARN)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARN)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARN)


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
    args = parser.parse_args()

    LOGGER.info(f'load model at {args.carbon_model_path}')
    with open(args.carbon_model_path, 'rb') as model_file:
        model = pickle.load(model_file).copy()
    LOGGER.info(f'ensure raster base data are present')
    missing_predictor_list = []
    predictor_id_path_list = []
    for predictor_id in model['predictor_list']:
        predictor_path = os.path.join(
            args.predictor_raster_dir, f'{predictor_id}.tif')
        predictor_id_path_list.append((predictor_id, predictor_path))
        if not os.path.exists(predictor_path):
            missing_predictor_list.append(
                f'{predictor_id}: {predictor_path}')
    if missing_predictor_list:
        predictor_str = "\n".join(missing_predictor_list)
        raise ValueError(
            f'missing the following predictor rasters:\n{predictor_str}')
    LOGGER.info(f'all found: {predictor_id_path_list}')

    workspace_dir = f'''workspace_{os.path.splitext(os.path.basename(
        args.forest_cover_path))[0]}'''
    os.makedirs(workspace_dir, exist_ok=True)

    task_graph = taskgraph.TaskGraph(workspace_dir, -1)

    raster_info = geoprocessing.get_raster_info(args.forest_cover_path)
    if abs(raster_info['pixel_size'][0]) < 3:
        raise ValueError(
            f'{args.forest_cover_path} must be projected in meters')

    LOGGER.info('gaussian filter forest cover')

    LOGGER.info('clip input rasters to forest cover')
    aligned_predictor_path_list = []
    for predictor_id, predictor_path in predictor_id_path_list:
        if model['gf_forest_id'] == predictor_id:
            # warp the predictor to be correct blocksize etc
            predictor_path = args.forest_cover_path
        warped_predictor_path = os.path.join(
            workspace_dir, f'warped_{os.path.basename(predictor_path)}')
        warp_task = task_graph.add_task(
            func=geoprocessing.warp_raster,
            args=(predictor_path, raster_info['pixel_size'],
                  warped_predictor_path, 'nearest'),
            kwargs={
                'target_bb': raster_info['bounding_box'],
                'target_projection_wkt': raster_info['projection_wkt'],
                'working_dir': workspace_dir,
            },
            target_path_list=[warped_predictor_path],
            task_name=f'warp {predictor_path}')
        if model['gf_forest_id'] == predictor_id:
            gf_forest_cover_path = os.path.join(
                workspace_dir, f'''{model["gf_size"]}_{
                os.path.basename(args.forest_cover_path)}''')
            task_graph.add_task(
                func=gaussian_filter_rasters.filter_raster,
                args=((warped_predictor_path, 1), model['gf_size'],
                      gf_forest_cover_path),
                dependent_task_list=[warp_task],
                target_path_list=[gf_forest_cover_path],
                task_name=f'gaussian filter {gf_forest_cover_path}')
            aligned_predictor_path_list.append(gf_forest_cover_path)
        else:
            aligned_predictor_path_list.append(warped_predictor_path)

    task_graph.join()

    LOGGER.info('apply model')
    nodata = -1

    def _apply_model(*raster_nodata_array):
        n = len(raster_nodata_array)//2
        raster_array = raster_nodata_array[0:n]
        nodata_array = raster_nodata_array[n:]
        valid_mask = numpy.all(
            [~numpy.isclose(array, nodata) for array, nodata in
             zip(raster_array, nodata_array)], axis=(0,))
        result = numpy.full(valid_mask.shape, nodata)
        value_list = numpy.asarray([
            array[valid_mask] for array in raster_array]).transpose()
        LOGGER.debug(f'len of valuelist: {value_list.shape}')
        if value_list.shape[0] > 0:
            result[valid_mask] = train_regression_model.clip_to_range(
                model['model'].predict(value_list), 10, 400)
        return result

    model_result_path = f'''forest_edge_result_{os.path.basename(os.path.splitext(
        args.forest_cover_path)[0])}.tif'''
    geoprocessing.raster_calculator(
        [(path, 1) for path in aligned_predictor_path_list] +
        [(geoprocessing.get_raster_info(path)['nodata'][0], 'raw')
         for path in aligned_predictor_path_list],
        _apply_model, model_result_path,
        gdal.GDT_Float32, nodata)

    model_result_path = f'''full_forest_edge_result_{os.path.basename(os.path.splitext(
        args.forest_cover_path)[0])}.tif'''
    geoprocessing.raster_calculator(
        [(path, 1) if path == gf_forest_cover_path else (1.0, 'raw') for path in aligned_predictor_path_list] +
        [(geoprocessing.get_raster_info(path)['nodata'][0], 'raw')
         for path in aligned_predictor_path_list],
        _apply_model, model_result_path,
        gdal.GDT_Float32, nodata)

    model_result_path = f'''no_forest_edge_result_{os.path.basename(os.path.splitext(
        args.forest_cover_path)[0])}.tif'''
    geoprocessing.raster_calculator(
        [(path, 1) if path == gf_forest_cover_path else (0.0, 'raw') for path in aligned_predictor_path_list] +
        [(geoprocessing.get_raster_info(path)['nodata'][0], 'raw')
         for path in aligned_predictor_path_list],
        _apply_model, model_result_path,
        gdal.GDT_Float32, nodata)


if __name__ == '__main__':
    main()
