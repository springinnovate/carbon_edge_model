"""Run carbon model on all the IPCC masks."""
import glob
import os
import logging
import multiprocessing

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
    task_graph = taskgraph.TaskGraph('./output_global/ipcc_optimization', multiprocessing.cpu_count(), 15)
    search_path = './output_global/ipcc_optimization/ipcccoarsened_marginal_value_ipcc_mask_*.tif'
    full_forest_mask_path_list = []
    for coarse_new_forest_mask_path in glob.glob(search_path):
        area_substring = os.path.splitext((coarse_new_forest_mask_path.split('_')[-1]))[0]

        carbon_opt_forest_step_path = (
            '%s_uncoarsened_forest_mask%s' % os.path.splitext(coarse_new_forest_mask_path))
        uncoarsen_forest_mask_task = task_graph.add_task(
            func=geoprocessing.warp_raster,
            args=(coarse_new_forest_mask_path, ECKERT_PIXEL_SIZE,
                  carbon_opt_forest_step_path, 'near'),
            kwargs={
                'target_bb': GLOBAL_BOUNDING_BOX_TUPLE[1],
                'target_projection_wkt': WORLD_ECKERT_IV_WKT,
                'n_threads': multiprocessing.cpu_count(),
                'working_dir': PRE_WARP_DIR,
                'raster_driver_creation_tuple': ZSTD_CREATION_TUPLE
            },
            target_path_list=[carbon_opt_forest_step_path],
            task_name=f'uncoarsen {carbon_opt_forest_step_path}')

        combined_forest_mask_path = f'./output_global/ipcc_optimization/ipcc_total_forest_mask_{area_substring}.tif'
        _ = task_graph.add_task(
            func=add_masks,
            args=(carbon_opt_forest_step_path, BASE_FOREST_MASK_PATH,
                  combined_forest_mask_path),
            target_path_list=[combined_forest_mask_path],
            dependent_task_list=[uncoarsen_forest_mask_task],
            task_name=f'add {carbon_opt_forest_step_path} to ESA base')
        full_forest_mask_path_list.append(carbon_opt_forest_step_path)

    task_graph.join()
    for full_forest_mask_path in full_forest_mask_path_list:
        modeled_carbon_path = f'./output_global/ipcc_optimization/ipcc_carbon_modeled_by_regression_{area_substring}.tif'
        LOGGER.debug(f'calculating carbon for {modeled_carbon_path}')

        regression_carbon_model(
            CARBON_MODEL_PATH, GLOBAL_BOUNDING_BOX_TUPLE,
            full_forest_mask_path, PREDICTOR_RASTER_DIR,
            pre_warp_dir=PRE_WARP_DIR,
            target_result_path=modeled_carbon_path,
            external_task_graph=task_graph,
            clean_workspace=False)

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()
