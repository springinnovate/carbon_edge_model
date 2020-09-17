"""Edge effect only analysis.

Input:
    base LULC
    optimization mask
    "new" forest mask
    base biomass

Process:
    Calculate biomass for new optimization mask
    Mask new biomass against the "new" forest mask so it only shows old biomass
    Subtract new biomass from old biomass (output this raster based on
        optimization mask name)
    Report the sum based on the optimization mask name

Output:
    Raster of diff of new biomass in "old" forest
    Number with sum of biomass of optimization mask name.
"""
import glob
import logging
import os

import numpy
import pygeoprocessing
import taskgraph

import esa_restoration_optimization

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)

BASE_BIOMASS_RASTER_PATH = (
    './esa_restoration_optimization/biomass_rasters/'
    'biomass_modeled_mode_carbon_model_lsvr_poly_2_90000_pts_base.tif')

NEW_FOREST_RASTER_PATH = (
    './esa_restoration_optimization/new_forest_masks/'
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed_restoration_'
    'limited_md5_372bdfd9ffaf810b5f68ddeb4704f48f.tif')

IPCC_MASK_DIR_PATTERN = (
    './esa_restoration_optimization/optimization_workspaces/'
    'optimization_ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_'
    'compressed_restoration_limited_md5_372bdfd9ffaf810b5f68ddeb4704f48f'
    '_ipcc_mode/optimal_mask_*.tif')

MODELED_MASK_DIR_PATTERN = (
    './esa_restoration_optimization/optimization_workspaces/'
    'optimization_ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_'
    'compressed_restoration_limited_md5_372bdfd9ffaf810b5f68ddeb4704f48f'
    '_ipcc_mode/optimal_mask_*.tif')

WORKSPACE_DIR = 'edge_effect_only_workspace'
CSV_REPORT = os.path.join(WORKSPACE_DIR, 'edge_effect_only.csv')


def mask_to_nodata(
        base_raster_path, mask_raster_path, target_masked_base_raster_path):
    """Mask base to nodata where mask raster path is 1.

    Args:
        base_raster_path (str): arbitrary raster.
        mask_raster_path (str): path to mask raster containing 1 where mask
            is set.
        target_masked_base_raster_path (str): copy of base except where mask
            is 1, base is nodata.

    Returns:
        None.
    """
    base_info = pygeoprocessing.get_raster_info(base_raster_path)

    def mask_op(base_array, mask_array):
        result = numpy.copy(base_array)
        result[mask_array == 1] = base_info['nodata'][0]
        return result

    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1), (mask_raster_path, 1)], mask_op,
        target_masked_base_raster_path, base_info['datatype'],
        base_info['nodata'][0])


def diff_valid(a_raster_path, b_raster_path, target_diff_raster_path):
    """Calculate a-b.

    Args:
        a_raster_path (str): path to arbitrary raster
        b_raster_path (str): path to raster that is the same size as a
        target_diff_raster_path (str): result of a-b where both a and b are
            not nodata.

    Returns:
        None.
    """
    a_info = pygeoprocessing.get_raster_info(a_raster_path)
    b_info = pygeoprocessing.get_raster_info(b_raster_path)

    def valid_diff_op(a_array, b_array):
        """Calc a-b."""
        result = numpy.empty_like(a_array)
        result[:] = a_info['nodata'][0]
        valid_mask = (
            ~numpy.is_close(a_array, a_info['nodata'][0]) &
            ~numpy.is_close(b_array, b_info['nodata'][0]))
        result[valid_mask] = a_array[valid_mask] - b_array[valid_mask]
        return result

    pygeoprocessing.raster_calculator(
        [(a_raster_path, 1), (b_raster_path, 1)], valid_diff_op,
        target_diff_raster_path, a_info['datatype'], a_info['nodata'][0])


def sum_valid(raster_path):
    """Sum non-nodata pixesl in raster_path.

    Args:
        raster_path (str): path to arbitrary raster.

    Returns:
        sum of nodata pixels in raster at `raster_path`.
    """
    accumulator_sum = 0.0
    raster_nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    for _, raster_block in pygeoprocessing.iterblocks((raster_path, 1)):
        accumulator_sum += numpy.sum(
            raster_block[~numpy.isclose(raster_block, raster_nodata)])
    return accumulator_sum


def calculate_old_forest_biomass_increase(mask_raster_path):
    """Calculate increase due to new forest in only the old forest.

    Calculate the total new biomass, mask it to old forest only, subtract
    from base, then sum the difference. This is the amount of new biomass
    in "old forest" due to the new forest.

    Args:
        mask_raster_path (str): 1 where there's new forest, same size as
            BASE_BIOMASS_RASTER_PATH.

    Returns:
        sum of total biomass increase due to the new mask.
    """
    LOGGER.info(f'calculate biomass for {mask_raster_path}')
    biomass_raster_path = os.path.join(WORKSPACE_DIR, f'''{
        os.path.basename(os.path.splitext(mask_raster_path))}''')
    esa_restoration_optimization._calculate_modeled_biomass_from_mask(
        BASE_BIOMASS_RASTER_PATH, mask_raster_path,
        biomass_raster_path)
    old_forest_biomass_masked_raster = os.path.join(
        WORKSPACE_DIR, f'''old_forest_only_{os.path.basename(
            os.path.splitext(biomass_raster_path)[0])}''')

    LOGGER.info(
        f'mask {biomass_raster_path} opposite of NEW_FOREST_RASTER_PATH')
    mask_to_nodata(
        biomass_raster_path, NEW_FOREST_RASTER_PATH,
        old_forest_biomass_masked_raster)

    LOGGER.info(
        f'diff {old_forest_biomass_masked_raster} against base biomass')
    old_forest_biomass_diff_raster = os.path.join(
        WORKSPACE_DIR, f'''old_forest_diff_{os.path.basename(
            os.path.splitext(biomass_raster_path)[0])}''')
    diff_valid(
        old_forest_biomass_masked_raster, BASE_BIOMASS_RASTER_PATH,
        old_forest_biomass_diff_raster)

    # TODO: sum it
    LOGGER.info(f'sum {old_forest_biomass_diff_raster}')
    biomass_diff_sum = sum_valid(old_forest_biomass_diff_raster)
    return biomass_diff_sum


if __name__ == '__main__':
    try:
        os.makedirs(WORKSPACE_DIR)
    except OSError:
        pass

    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1)

    with open(CSV_REPORT, 'a') as csv_report_file:
        csv_report_file.write(
            'mask file,ipcc biomass increase,regression biomass increase\n')
    for ipcc_mask_raster_path, modeled_mask_raster_path in zip(
            glob.glob(IPCC_MASK_DIR_PATTERN),
            glob.glob(MODELED_MASK_DIR_PATTERN)):
        with open(CSV_REPORT, 'a') as csv_report_file:
            csv_report_file.write(f'''{os.path.basename(
                os.path.splitext(ipcc_mask_raster_path)[0])},''')
        for mask_raster_path, model_type in [
                (ipcc_mask_raster_path, 'ipcc'),
                (modeled_mask_raster_path, 'regression')]:
            LOGGER.debug(f'{model_type}: {mask_raster_path}')

        biomass_diff_sum_task = task_graph.add_task(
            func=calculate_old_forest_biomass_increase,
            args=(mask_raster_path,),
            task_name=f'calculate old forest biomass for {mask_raster_path}')

        with open(CSV_REPORT, 'a') as csv_report_file:
            csv_report_file.write(f',{biomass_diff_sum_task.get()}')

    with open(CSV_REPORT, 'a') as csv_report_file:
        csv_report_file.write('\n')
