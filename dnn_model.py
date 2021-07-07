"""Run Carbon Edge NN model."""
import os
import multiprocessing
import time
import logging

from osgeo import gdal
import ecoshard
import pygeoprocessing
import numpy
import taskgraph
import torch
import train_model
from train_model import NeuralNetwork
from train_model import PREDICTOR_LIST
from train_model import URL_PREFIX
from train_model import MASK_TYPES
from train_model import CELL_SIZE
from train_model import PROJECTION_WKT
from train_model import EXPECTED_MAX_EDGE_EFFECT_KM_LIST

gdal.SetCacheMax(2**27)
logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)

BOUNDING_BOX = [-179, -60, 179, 60]

WORKSPACE_DIR = 'workspace'
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
ALIGN_DIR = os.path.join(
    WORKSPACE_DIR, f'align{"_".join([str(v) for v in BOUNDING_BOX])}')
CHURN_DIR = os.path.join(
    WORKSPACE_DIR, f'churn{"_".join([str(v) for v in BOUNDING_BOX])}')
for dir_path in [WORKSPACE_DIR, ECOSHARD_DIR, ALIGN_DIR, CHURN_DIR]:
    os.makedirs(dir_path, exist_ok=True)
RASTER_LOOKUP_PATH = os.path.join(
    WORKSPACE_DIR,
    f'raster_lookup{"_".join([str(v) for v in BOUNDING_BOX])}.dat')


def download_data(task_graph, bounding_box):
    """Download the whole data stack."""
    # First download the response raster to align all the rest
    LOGGER.info(f'download data and clip to {bounding_box}')
    # download the rest and align to response
    aligned_predictor_list = []
    for filename, nodata in PREDICTOR_LIST:
        # it's a path/nodata tuple
        aligned_path = os.path.join(ALIGN_DIR, filename)
        aligned_predictor_list.append((aligned_path, nodata))
        url = URL_PREFIX + filename
        ecoshard_path = os.path.join(ECOSHARD_DIR, filename)
        download_task = task_graph.add_task(
            func=ecoshard.download_url,
            args=(url, ecoshard_path),
            target_path_list=[ecoshard_path],
            task_name=f'download {ecoshard_path}')
        aligned_path = os.path.join(ALIGN_DIR, filename)
        _ = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(ecoshard_path, CELL_SIZE, aligned_path, 'near'),
            kwargs={
                'target_bb': BOUNDING_BOX,
                'target_projection_wkt': PROJECTION_WKT,
                'working_dir': WORKSPACE_DIR},
            dependent_task_list=[download_task],
            target_path_list=[aligned_path],
            task_name=f'align {aligned_path}')
    return aligned_predictor_list


def _create_lulc_mask(lulc_raster_path, mask_codes, target_mask_raster_path):
    """Create a mask raster given an lulc and mask code."""
    numpy_mask_codes = numpy.array(mask_codes)
    pygeoprocessing.raster_calculator(
        [(lulc_raster_path, 1)],
        lambda array: numpy.isin(array, numpy_mask_codes),
        target_mask_raster_path, gdal.GDT_Byte, None)


def mask_lulc(task_graph, lulc_raster_path, workspace_dir):
    """Create all the masks and convolutions off of lulc_raster_path."""
    # this is calculated as 111km per degree
    convolution_raster_list = []
    edge_effect_index = None
    current_raster_index = -1

    for expected_max_edge_effect_km in EXPECTED_MAX_EDGE_EFFECT_KM_LIST:
        pixel_radius = (CELL_SIZE[0] * 111 / expected_max_edge_effect_km)**-1
        kernel_raster_path = os.path.join(
            workspace_dir, f'kernel_{pixel_radius}.tif')
        if not os.path.exists(kernel_raster_path):
            kernel_task = task_graph.add_task(
                func=train_model.make_kernel_raster,
                args=(pixel_radius, kernel_raster_path),
                target_path_list=[kernel_raster_path],
                task_name=f'make kernel of radius {pixel_radius}')
            kernel_task.join()

    for mask_id, mask_codes in MASK_TYPES:
        mask_raster_path = os.path.join(
            workspace_dir, f'''{os.path.basename(
                os.path.splitext(lulc_raster_path)[0])}_{mask_id}_mask.tif''')
        create_mask_task = task_graph.add_task(
            func=_create_lulc_mask,
            args=(lulc_raster_path, mask_codes, mask_raster_path),
            target_path_list=[mask_raster_path],
            task_name=f'create {mask_id} mask')
        if mask_id == 'forest':
            forest_mask_raster_path = mask_raster_path
            if edge_effect_index is None:
                LOGGER.debug(f'CURRENT EDGE INDEX {current_raster_index}')
                edge_effect_index = current_raster_index

        for expected_max_edge_effect_km in EXPECTED_MAX_EDGE_EFFECT_KM_LIST:
            current_raster_index += 1
            pixel_radius = (
                CELL_SIZE[0] * 111 / expected_max_edge_effect_km)**-1
            kernel_raster_path = os.path.join(
                workspace_dir, f'kernel_{pixel_radius}.tif')
            mask_gf_path = (
                f'{os.path.splitext(mask_raster_path)[0]}_gf_'
                f'{expected_max_edge_effect_km}.tif')
            LOGGER.debug(f'making convoluion for {mask_gf_path}')

            _ = task_graph.add_task(
                func=pygeoprocessing.convolve_2d,
                args=(
                    (mask_raster_path, 1), (kernel_raster_path, 1),
                    mask_gf_path),
                dependent_task_list=[create_mask_task],
                target_path_list=[mask_gf_path],
                task_name=(
                    f'create gaussian filter of {mask_id} at {mask_gf_path}'))
            convolution_raster_list.append(((mask_gf_path, None)))
    task_graph.join()
    LOGGER.debug(f'all done convolution list - {convolution_raster_list}')
    return forest_mask_raster_path, convolution_raster_list, edge_effect_index


def align_predictors(
        task_graph, lulc_raster_base_path, predictor_list, workspace_dir):
    """Align all the predictors to lulc."""
    lulc_raster_info = pygeoprocessing.get_raster_info(lulc_raster_base_path)
    aligned_dir = os.path.join(
        workspace_dir, 'aligned',
        os.path.basename(os.path.splitext(lulc_raster_base_path)[0]))
    os.makedirs(aligned_dir, exist_ok=True)
    aligned_predictor_list = []
    for predictor_raster_path, nodata in predictor_list:
        if nodata is None:
            nodata = pygeoprocessing.get_raster_info(
                predictor_raster_path)['nodata'][0]
        aligned_predictor_raster_path = os.path.join(
            aligned_dir, os.path.basename(predictor_raster_path))
        task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(
                predictor_raster_path, lulc_raster_info['pixel_size'],
                aligned_predictor_raster_path, 'near'),
            kwargs={
                'target_bb': lulc_raster_info['bounding_box'],
                'target_projection_wkt': lulc_raster_info['projection_wkt']},
            target_path_list=[aligned_predictor_raster_path],
            task_name=f'align {aligned_predictor_raster_path}')
        aligned_predictor_list.append((aligned_predictor_raster_path, nodata))
    return aligned_predictor_list


def model_predict(
            model, lulc_raster_path, forest_mask_raster_path,
            aligned_predictor_list, predicted_biomass_raster_path):
    """Predict biomass given predictors."""
    pygeoprocessing.new_raster_from_base(
        lulc_raster_path, predicted_biomass_raster_path, gdal.GDT_Float32,
        [-1])
    predicted_biomass_raster = gdal.OpenEx(
        predicted_biomass_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    predicted_biomass_band = predicted_biomass_raster.GetRasterBand(1)

    predictor_band_nodata_list = []
    raster_list = []
    # simple lookup to map predictor band/nodata to a list
    for predictor_path, nodata in aligned_predictor_list:
        predictor_raster = gdal.OpenEx(predictor_path, gdal.OF_RASTER)
        raster_list.append(predictor_raster)
        predictor_band = predictor_raster.GetRasterBand(1)

        if nodata is None:
            nodata = predictor_band.GetNoDataValue()
        predictor_band_nodata_list.append((predictor_band, nodata))
    forest_raster = gdal.OpenEx(forest_mask_raster_path, gdal.OF_RASTER)
    forest_band = forest_raster.GetRasterBand(1)

    last_time = time.time()
    n_pixels = forest_band.XSize * forest_band.YSize
    current_pixels = 0
    for offset_dict in pygeoprocessing.iterblocks(
            (lulc_raster_path, 1), offset_only=True):
        current_pixels += offset_dict['win_xsize']*offset_dict['win_ysize']
        if time.time() - last_time > 10:
            LOGGER.info(f'{100*current_pixels/n_pixels}% complete')
            last_time = time.time()
        forest_array = forest_band.ReadAsArray(**offset_dict)
        valid_mask = (forest_array == 1)
        x_vector = None
        array_list = []
        for band, nodata in predictor_band_nodata_list:
            array = band.ReadAsArray(**offset_dict)
            if nodata is None:
                nodata = band.GetNoDataValue()
            if nodata is not None:
                valid_mask &= array != nodata
            array_list.append(array)
        if not numpy.any(valid_mask):
            continue
        for array in array_list:
            if x_vector is None:
                x_vector = array[valid_mask].astype(numpy.float32)
                x_vector = numpy.reshape(x_vector, (-1, x_vector.size))
            else:
                valid_array = array[valid_mask].astype(numpy.float32)
                valid_array = numpy.reshape(
                    valid_array, (-1, valid_array.size))
                x_vector = numpy.append(x_vector, valid_array, axis=0)
        y_vector = model(torch.from_numpy(x_vector.T))
        result = numpy.full(forest_array.shape, -1)
        result[valid_mask] = (y_vector.detach().numpy()).flatten()
        predicted_biomass_band.WriteArray(
            result,
            xoff=offset_dict['xoff'],
            yoff=offset_dict['yoff'])
    predicted_biomass_band = None
    predicted_biomass_raster = None


def run_model(lulc_raster_path, model_path, target_biomass_path):
    """Run DNN carbon edge model."""
    task_graph = taskgraph.TaskGraph('.', multiprocessing.cpu_count())
    LOGGER.info('model data with input lulc')
    local_workspace = os.path.join(
        WORKSPACE_DIR,
        os.path.basename(os.path.splitext(lulc_raster_path)[0]))
    LOGGER.info('fetch predictors')
    predictor_list = download_data(task_graph, BOUNDING_BOX)
    LOGGER.info(f'align predictors to {lulc_raster_path}')
    aligned_predictor_list = align_predictors(
        task_graph, lulc_raster_path, predictor_list,
        local_workspace)
    LOGGER.info('mask forest and build convolutions')
    payload = mask_lulc(task_graph, lulc_raster_path, local_workspace)
    forest_mask_raster_path, convolution_raster_list, edge_effect_index = \
        payload
    task_graph.join()
    task_graph.close()
    task_graph = None

    LOGGER.info(
        f'load model {len(aligned_predictor_list)} predictors '
        f'{len(convolution_raster_list)} convolutions')
    model = NeuralNetwork(
        len(convolution_raster_list)+len(aligned_predictor_list))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    LOGGER.info('predict biomass to {predicted_biomass_raster_path}')
    model_predict(
        model, lulc_raster_path, forest_mask_raster_path,
        aligned_predictor_list+convolution_raster_list,
        target_biomass_path)
    LOGGER.debug('all done')
