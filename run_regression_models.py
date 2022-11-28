"""Run carbon model on all the IPCC masks."""
import glob
import os
import logging
import multiprocessing

import numpy
from run_model import regression_carbon_model
from run_model import GLOBAL_BOUNDING_BOX_TUPLE
from run_model import ECKERT_PIXEL_SIZE
from run_model import WORLD_ECKERT_IV_WKT
from run_model import ZSTD_CREATION_TUPLE
from ecoshard import taskgraph
from ecoshard import geoprocessing

BASE_FOREST_MASK_PATH = './output_global/forest_mask_esa.tif'
CARBON_MODEL_PATH = './models/hansen_model_2022_07_14.dat'
PREDICTOR_RASTER_DIR = './processed_rasters'
PRE_WARP_DIR = os.path.join(PREDICTOR_RASTER_DIR, f'pre_warped_{GLOBAL_BOUNDING_BOX_TUPLE[0]}')

LOGGER = logging.getLogger(__name__)

LOG_FORMAT = (
    '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
    ' [%(funcName)s:%(lineno)d] %(message)s')
logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FORMAT)


def sum_raster(raster_path):
    """Return the sum of non-nodata value pixels in ``raster_path``."""
    running_sum = 0.0
    nodata = geoprocessing.get_raster_info(raster_path)['nodata'][0]
    for _, block_array in geoprocessing.iterblocks((raster_path, 1)):
        if nodata is not None:
            valid_array = block_array != nodata
        else:
            valid_array = slice(-1)
        running_sum += numpy.sum(block_array[valid_array])
    return running_sum


def sum_by_mask(raster_path, mask_path):
    """Return tuple of sum of non-nodata values in raster_path in and out of mask.

    Returns:
        (in sum, out sum)
    """
    in_running_sum = 0.0
    out_running_sum = 0.0
    nodata = geoprocessing.get_raster_info(raster_path)['nodata'][0]
    for ((_, block_array), (_, mask_array)) in \
            zip(geoprocessing.iterblocks((raster_path, 1)),
                geoprocessing.iterblocks((mask_path, 1))):
        if nodata is not None:
            valid_array = block_array != nodata
        else:
            valid_array = numpy.ones(block_array.shape, dtype=bool)
        in_running_sum += numpy.sum(block_array[valid_array & (mask_array == 1)])
        out_running_sum += numpy.sum(block_array[valid_array & (mask_array != 1)])
    return (in_running_sum, out_running_sum)


def add_masks(mask_a_path, mask_b_path, target_path):
    """Combine two masks as a logical OR.

    Args:
        mask_a_path, mask_b_path (str): path to mask rasters with 1 indicating mask
        target_path (str): path to combined mask raster.

    Return:
        None
    """
    raster_info = geoprocessing.get_raster_info(mask_a_path)
    nodata = raster_info['nodata'][0]

    def _add_masks(mask_a_array, mask_b_array):
        """Combine a and b."""
        valid_mask = (mask_a_array == 1) | (mask_b_array == 1)
        return valid_mask

    geoprocessing.raster_calculator(
        [(mask_a_path, 1), (mask_b_path, 1)],
        _add_masks, target_path, raster_info['datatype'], nodata)


def main():
    """Entry point."""
    task_graph = taskgraph.TaskGraph('./output_global/regression_optimization', multiprocessing.cpu_count(), 15)
    search_path = './output_global/regression_optimization/regressioncoarsened_marginal_value_regression_mask_*full_forest_mask.tif'
    full_forest_mask_path_list = []
    raster_sum_list = []
    transient_run = True


    for full_forest_mask_path in glob.glob(search_path):
        area_substring = os.path.splitext((full_forest_mask_path.split('_')[-4]))[0]
        new_forest_mask_path = f'./output_global/regression_optimization/regressioncoarsened_marginal_value_regression_mask_{area_substring}_new_forest_mask.tif'
        modeled_carbon_path = f'./output_global/regression_optimization/regressioncoarsened_marginal_value_regression_mask_{area_substring}_regression.tif'

        count_full_forest_pixel_task = task_graph.add_task(
            func=sum_raster,
            args=(full_forest_mask_path,),
            transient_run=transient_run,
            task_name=f'sum raster of {full_forest_mask_path}',
            store_result=True)
        # TODO: get the right mask here ->
        sum_in_out_forest_carbon_density_by_mask_task = task_graph.add_task(
            func=sum_by_mask,
            args=(modeled_carbon_path, new_forest_mask_path),
            store_result=True,
            transient_run=transient_run,
            task_name=f'separate out old and new carbon for {modeled_carbon_path}')

        # count number of new forest pixels
        count_new_forest_pixel_task = task_graph.add_task(
            func=sum_raster,
            args=(new_forest_mask_path,),
            transient_run=transient_run,
            task_name=f'sum raster of {new_forest_mask_path}',
            store_result=True)

        raster_sum_list.append(
            (os.path.basename(modeled_carbon_path),
             count_full_forest_pixel_task,
             count_new_forest_pixel_task,
             sum_in_out_forest_carbon_density_by_mask_task))

    task_graph.join()

    task_graph.join()
    raster_info = geoprocessing.get_raster_info(new_forest_mask_path)
    with open('regression_optimization_regression_modeled_carbon.csv', 'w') as opt_table:
        opt_table.write(
            'file,'
            'number of forest pixels,'
            'number of old forest pixels,'
            'number of new forest pixels,'
            'sum of carbon density for all forest pixels,'
            'sum of carbon density for old forest pixels,'
            'sum of carbon density for new forest pixels,'
            'carbon density per pixel for all forest,'
            'carbon density per pixel in old forest,'
            'carbon density per pixel in new forest,'
            'carbon density per pixel in esa scenario,'
            'area of pixel in m^2\n')
        for path, count_full_forest_pixel_task, count_new_forest_pixel_task, sum_in_out_forest_carbon_density_by_mask_task in raster_sum_list:
            LOGGER.debug(f'processing {path}')
            new_carbon_density_sum = sum_in_out_forest_carbon_density_by_mask_task.get()[0]
            old_carbon_density_sum = sum_in_out_forest_carbon_density_by_mask_task.get()[1]
            all_forest_pixel_count = count_full_forest_pixel_task.get()
            new_forest_pixel_count = count_new_forest_pixel_task.get()
            old_forest_pixel_count = all_forest_pixel_count-new_forest_pixel_count
            # LOGGER.debug(
            #     f'(new_carbon_density_sum+old_carbon_density_sum==regression_forest_density_sum_task.get())\n'
            #     f'({new_carbon_density_sum}+{old_carbon_density_sum}=={regression_forest_density_sum_task.get()})')
            # note that new_carbon_density_sum+old_carbon_density_sum should ==regression_forest_density_sum_task but due to little roundoff error its off by a relative 1e-6 value
            opt_table.write(
                f'{path},'
                f'{all_forest_pixel_count},'
                f'{old_forest_pixel_count},'
                f'{new_forest_pixel_count},'
                f'{old_carbon_density_sum+new_carbon_density_sum},'
                f'{old_carbon_density_sum},'
                f'{new_carbon_density_sum},'
                #  divide the total carbon in the mask by number of pixels in mask
                f'{(new_carbon_density_sum+old_carbon_density_sum)/(all_forest_pixel_count)},'
                f'{old_carbon_density_sum/old_forest_pixel_count},'
                f'{new_carbon_density_sum/new_forest_pixel_count},'
                f'{abs(numpy.prod(raster_info["pixel_size"]))}\n')

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()
