"""Run Carbon Edge NN model."""
import argparse
import os
import collections
import multiprocessing
import pickle
import time
import logging

from osgeo import gdal
from osgeo import osr
from osgeo import ogr
from utils import esa_to_carbon_model_landcover_types
import ecoshard
import pygeoprocessing
import numpy
import scipy
import taskgraph
import torch
from train_model import NeuralNetwork
from train_model import PREDICTOR_LIST
from train_model import URL_PREFIX
from train_model import MASK_TYPES
from train_model import CELL_SIZE
from train_model import PROJECTION_WKT
from train_model import EXPECTED_MAX_EDGE_EFFECT_KM_LIST

torch.autograd.set_detect_anomaly(True)

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
    WORKSPACE_DIR, f'raster_lookup{"_".join([str(v) for v in BOUNDING_BOX])}.dat')


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


def sample_data(
        time_domain_mask_list, predictor_lookup, sample_rate, edge_index,
        sample_point_vector_path):
    """Sample data stack.

    All input rasters are aligned.

    Args:
        response_path (str): path to response raster
        predictor_lookup (dict): dictionary with keys 'predictor' and
            'time_predictor'. 'time_predictor' are either a tuple of rasters
            or a single multiband raster with indexes that conform to the
            bands in ``response_path``.
        edge_index (int): this is the edge raster in the predictor stack
            that should be used to randomly select samples from.


    """
    raster_info = pygeoprocessing.get_raster_info(
        predictor_lookup['predictor'][0][0])
    inv_gt = gdal.InvGeoTransform(raster_info['geotransform'])

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_info['projection_wkt'])
    raster_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    gpkg_driver = gdal.GetDriverByName('GPKG')
    sample_point_vector = gpkg_driver.Create(
        sample_point_vector_path, 0, 0, 0, gdal.GDT_Unknown)
    sample_point_layer = sample_point_vector.CreateLayer(
        'sample_points', raster_srs, ogr.wkbPoint)
    sample_point_layer.StartTransaction()

    LOGGER.info(f'building sample data')
    predictor_band_nodata_list = []
    raster_list = []
    # simple lookup to map predictor band/nodata to a list
    for predictor_path, nodata in predictor_lookup['predictor']:
        predictor_raster = gdal.OpenEx(predictor_path, gdal.OF_RASTER)
        raster_list.append(predictor_raster)
        predictor_band = predictor_raster.GetRasterBand(1)

        if nodata is None:
            nodata = predictor_band.GetNoDataValue()
        predictor_band_nodata_list.append((predictor_band, nodata))

    # create a dictionary that maps time index to list of predictor band/nodata
    # values, will be used to create a stack of data with the previous
    # collection and one per timestep on this one
    time_predictor_lookup = collections.defaultdict(list)
    for payload in predictor_lookup['time_predictor']:
        # time predictors could either be a tuple of rasters or a single
        # raster with multiple bands
        if isinstance(payload, tuple):
            time_predictor_path, nodata = payload
            time_predictor_raster = gdal.OpenEx(
                time_predictor_path, gdal.OF_RASTER)
            raster_list.append(time_predictor_raster)
            for index in range(time_predictor_raster.RasterCount):
                time_predictor_band = time_predictor_raster.GetRasterBand(
                    index+1)
                if nodata is None:
                    nodata = time_predictor_band.GetNoDataValue()
                time_predictor_lookup[index].append(
                    (time_predictor_band, nodata))
        elif isinstance(payload, list):
            for index, (time_predictor_path, nodata) in enumerate(
                    payload):
                time_predictor_raster = gdal.OpenEx(
                    time_predictor_path, gdal.OF_RASTER)
                raster_list.append(time_predictor_raster)
                time_predictor_band = time_predictor_raster.GetRasterBand(1)
                if nodata is None:
                    nodata = time_predictor_band.GetNoDataValue()
                time_predictor_lookup[index].append((
                    time_predictor_band, nodata))
        else:
            raise ValueError(
                f'expected str or tuple but got {payload}')
    mask_band_list = []
    for time_domain_mask_raster_path in time_domain_mask_list:
        mask_raster = gdal.OpenEx(time_domain_mask_raster_path, gdal.OF_RASTER)
        raster_list.append(mask_raster)
        mask_band = mask_raster.GetRasterBand(1)
        mask_band_list.append(mask_band)

    # build up an array of predictor stack
    response_raster = gdal.OpenEx(predictor_lookup['response'], gdal.OF_RASTER)
    raster_list.append(response_raster)
    # if response_raster.RasterCount != len(time_predictor_lookup):
    #     raise ValueError(
    #         f'expected {response_raster.RasterCount} time elements but only '
    #         f'got {len(time_predictor_lookup)}')

    y_list = []
    x_vector = None
    i = 0
    last_time = time.time()
    total_pixels = predictor_raster.RasterXSize * predictor_raster.RasterYSize
    for offset_dict in pygeoprocessing.iterblocks(
            (predictor_lookup['response'], 1),
            offset_only=True, largest_block=2**20):
        if time.time() - last_time > 5.0:
            n_pixels_processed = offset_dict['xoff']+offset_dict['yoff']*predictor_raster.RasterXSize
            LOGGER.info(f"processed {100*n_pixels_processed/total_pixels:.3f}% so far ({n_pixels_processed}) (x/y {offset_dict['xoff']}/{offset_dict['yoff']}) y_list size {len(y_list)}")
            last_time = time.time()
        predictor_stack = []  # N elements long
        valid_array = numpy.ones(
            (offset_dict['win_ysize'], offset_dict['win_xsize']),
            dtype=bool)
        # load all the regular predictors
        for predictor_band, predictor_nodata in predictor_band_nodata_list:
            predictor_array = predictor_band.ReadAsArray(**offset_dict)
            if predictor_nodata is not None:
                valid_array &= predictor_array != predictor_nodata
            predictor_stack.append(predictor_array)

        if not numpy.any(valid_array):
            continue

        # load the time based predictors
        for index, time_predictor_band_nodata_list in \
                time_predictor_lookup.items():
            if index > MAX_TIME_INDEX:
                break
            mask_array = mask_band_list[index].ReadAsArray(**offset_dict)
            valid_time_array = valid_array & (mask_array == 1)
            predictor_time_stack = []
            predictor_time_stack.extend(predictor_stack)
            for predictor_index, (predictor_band, predictor_nodata) in \
                    enumerate(time_predictor_band_nodata_list):
                predictor_array = predictor_band.ReadAsArray(**offset_dict)
                if predictor_nodata is not None:
                    valid_time_array &= predictor_array != predictor_nodata
                predictor_time_stack.append(predictor_array)

            # load the time based responses
            response_band = response_raster.GetRasterBand(index+1)
            response_nodata = response_band.GetNoDataValue()
            response_array = response_band.ReadAsArray(**offset_dict)
            if response_nodata is not None:
                valid_time_array &= response_array != response_nodata

            if not numpy.any(valid_time_array):
                break

            sample_mask = numpy.random.rand(
                numpy.count_nonzero(valid_time_array)) < sample_rate

            X2D, Y2D = numpy.meshgrid(
                range(valid_time_array.shape[1]),
                range(valid_time_array.shape[0]))

            for i, j in zip(
                    (X2D[valid_time_array])[sample_mask],
                    (Y2D[valid_time_array])[sample_mask]):

                sample_point = ogr.Feature(sample_point_layer.GetLayerDefn())
                sample_geom = ogr.Geometry(ogr.wkbPoint)
                x, y = gdal.ApplyGeoTransform(
                    inv_gt, i+0.5+offset_dict['xoff'],
                    j+0.5+offset_dict['yoff'])
                sample_geom.AddPoint(x, y)
                sample_point.SetGeometry(sample_geom)
                sample_point_layer.CreateFeature(sample_point)

            # all of response_time_stack and response_array are valid, clip and add to set
            local_x_list = []
            # each element in array should correspond with an element in y
            for array in predictor_time_stack:
                local_x_list.append((array[valid_time_array])[sample_mask])
            if x_vector is None:
                x_vector = numpy.array(local_x_list)
            else:
                local_x_vector = numpy.array(local_x_list)
                x_vector = numpy.append(x_vector, local_x_vector, axis=1)
            y_list.extend(
                list((response_array[valid_time_array])[sample_mask]))

        i += 1
    y_vector = numpy.array(y_list)
    LOGGER.debug(f'got all done {x_vector.shape} {y_vector.shape}')
    sample_point_layer.CommitTransaction()
    return (x_vector.T).astype(numpy.float32), (y_vector.astype(numpy.float32))


def make_kernel_raster(pixel_radius, target_path):
    """Create kernel with given radius to `target_path`."""
    truncate = 2
    size = int(pixel_radius * 2 * truncate + 1)
    step_fn = numpy.zeros((size, size))
    step_fn[size//2, size//2] = 1
    kernel_array = scipy.ndimage.filters.gaussian_filter(
        step_fn, pixel_radius, order=0, mode='constant', cval=0.0,
        truncate=truncate)
    pygeoprocessing.numpy_array_to_raster(
        kernel_array, -1, (1., -1.), (0.,  0.), None,
        target_path)


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
                func=make_kernel_raster,
                args=(pixel_radius, kernel_raster_path),
                target_path_list=[kernel_raster_path],
                task_name=f'make kernel of radius {pixel_radius}')
            kernel_task.join()

    for mask_id, mask_codes in MASK_TYPES:
        mask_raster_path = os.path.join(
            workspace_dir, f'{os.path.basename(os.path.splitext(lulc_raster_path)[0])}_{mask_id}_mask.tif')
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

            pixel_radius = (CELL_SIZE[0] * 111 / expected_max_edge_effect_km)**-1
            kernel_raster_path = os.path.join(
                workspace_dir, f'kernel_{pixel_radius}.tif')
            mask_gf_path = (
                f'{os.path.splitext(mask_raster_path)[0]}_gf_'
                f'{expected_max_edge_effect_km}.tif')
            LOGGER.debug(f'making convoluion for {mask_gf_path}')

            convolution_task = task_graph.add_task(
                func=pygeoprocessing.convolve_2d,
                args=(
                    (mask_raster_path, 1), (kernel_raster_path, 1),
                    mask_gf_path),
                dependent_task_list=[create_mask_task],
                target_path_list=[mask_gf_path],
                task_name=f'create gaussian filter of {mask_id} at {mask_gf_path}')
            convolution_raster_list.append(((mask_gf_path, None)))
    task_graph.join()
    LOGGER.debug(f'all done convolution list - {convolution_raster_list}')
    return forest_mask_raster_path, convolution_raster_list, edge_effect_index


def align_predictors(
        task_graph, lulc_raster_base_path, predictor_list, workspace_dir):
    """Align all the predictors to lulc."""
    lulc_raster_info = pygeoprocessing.get_raster_info(lulc_raster_base_path)
    aligned_dir = os.path.join(workspace_dir, 'aligned')
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
                valid_array = numpy.reshape(valid_array, (-1, valid_array.size))
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


def main():
    parser = argparse.ArgumentParser(description='Run Carbon Edge Model')
    parser.add_argument(
        'lulc_raster_input', help='Path to lulc raster to predict biomass')
    parser.add_argument(
        '--model_path',
        default='./models/model_400.dat', help='path to pretrained model')
    args = parser.parse_args()

    task_graph = taskgraph.TaskGraph('.', multiprocessing.cpu_count())

    LOGGER.info('model data with input lulc')
    local_workspace = os.path.join(
        WORKSPACE_DIR,
        os.path.basename(os.path.splitext(args.lulc_raster_input)[0]))
    LOGGER.info('fetch predictors')
    predictor_list = download_data(task_graph, BOUNDING_BOX)
    LOGGER.info(f'align predictors to {args.lulc_raster_input}')
    aligned_predictor_list = align_predictors(
        task_graph, args.lulc_raster_input, predictor_list,
        local_workspace)
    LOGGER.info('mask forest and build convolutions')
    forest_mask_raster_path, convolution_raster_list, edge_effect_index = mask_lulc(
        task_graph, args.lulc_raster_input, local_workspace)
    task_graph.join()
    task_graph.close()
    task_graph = None

    LOGGER.info(
        f'load model {len(aligned_predictor_list)} predictors '
        f'{len(convolution_raster_list)} convolutions')
    model = NeuralNetwork(
        len(convolution_raster_list)+len(aligned_predictor_list))
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    predicted_biomass_raster_path = (
        f'modeled_biomass_{os.path.basename(args.lulc_raster_input)}')
    LOGGER.info('predict biomass to {predicted_biomass_raster_path}')
    model_predict(
        model, args.lulc_raster_input, forest_mask_raster_path,
        aligned_predictor_list+convolution_raster_list,
        predicted_biomass_raster_path)
    LOGGER.debug('all done')


if __name__ == '__main__':
    main()
