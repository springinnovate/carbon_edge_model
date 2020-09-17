"""Edge effect only analysis.

Input:
    base LULC
    optimization mask
    "new" forest mask
    base biomass

Process:
    Calculate biomass for new optimization mask
    Mask new biomass against the "new" forest mask so it only shows old biomass
    Subtract new biomass from old biomass (output this raster based on optimization mask name)
    Report the sum based on the optimization mask name

Output:
    Raster of diff of new biomass in "old" forest
    Number with sum of biomass of optimization mask name.
"""
import glob
import logging

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


if __name__ == '__main__':
    for ipcc_mask_raster_path, modeled_mask_raster_path in zip(
            glob.glob(IPCC_MASK_DIR_PATTERN),
            glob.glob(MODELED_MASK_DIR_PATTERN)):
        for mask_raster_path, model_type in [
                (ipcc_mask_raster_path, 'ipcc'),
                (modeled_mask_raster_path, 'regression')]:
            LOGGER.debug(f'{model_type}: {mask_raster_path}')
    # TODO: to calculate modeled biomass ->
    # esa_restoration_optimization._calculate_modeled_biomass_from_mask(
    #     base_lulc_raster_path, new_forest_mask_raster_path,
    #     target_biomass_raster_path)
