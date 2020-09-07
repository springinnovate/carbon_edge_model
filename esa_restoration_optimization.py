"""Script to calculate ESA/restoration optimization."""
import collections
import os
import logging
import pickle
import sys

from osgeo import gdal
import numpy
import pygeoprocessing
import taskgraph
import carbon_edge_model

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)

LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)

# Working directories for substeps
WORKSPACE_DIR = './esa_restoration_optimization'
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
BIOMASS_RASTER_DIR = os.path.join(WORKSPACE_DIR, 'biomass_rasters')
MARGINAL_VALUE_WORKSPACE = os.path.join(
    WORKSPACE_DIR, 'marginal_value_rasters')
OPTIMIZATION_WORKSPACE = os.path.join(
    WORKSPACE_DIR, 'optimization_workspaces')
OPTIMIAZATION_SCENARIOS_DIR = os.path.join(
    WORKSPACE_DIR, 'optimization_scenarios')

MODEL_PATH = './models/carbon_model_lsvr_poly_2_90000_pts.mod'
MODEL_BASE_DIR = './model_base_data'
LOGGER.info(f'load the biomass model at {MODEL_PATH}')
with open(MODEL_PATH, 'rb') as MODEL_FILE:
    BIOMASS_MODEL = pickle.load(MODEL_FILE)


# *** DATA SECTION ***
# There are two landcover configurations, ESA and restoration of ESA
BASE_LULC_RASTER_PATH = os.path.join(
    MODEL_BASE_DIR,
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif')
ESA_RESTORATION_SCENARIO_RASTER_PATH = os.path.join(
    MODEL_BASE_DIR,
    'scenario_data/ESA_restoration_scenario.tif')

# These are used in combination with an ESA landcover map to calculate carbon
CARBON_ZONES_VECTOR_URI = os.path.join(
    MODEL_BASE_DIR,
    'carbon_zones_md5_aa16830f64d1ef66ebdf2552fb8a9c0d.gpkg')
IPCC_CARBON_TABLE_URI = os.path.join(
    MODEL_BASE_DIR,
    'IPCC_carbon_table_md5_a91f7ade46871575861005764d85cfa7.csv')

# Constants useful for code readability
CARBON_MODEL_ID = os.path.basename(os.path.splitext(MODEL_PATH)[0])
IPCC_MODE = 'ipcc_mode'
# model mode is based off of carbon model ID
MODELED_MODE = f'modeled_mode_{CARBON_MODEL_ID}'
BASE_SCENARIO = 'base'
RESTORATION_SCENARIO = 'scenario'
FOREST_CODE = 50
TARGET_AREA_HA = 350000000
AREA_REPORT_STEP_AMOUNT_HA = TARGET_AREA_HA/20


def _mkdir(dir_path):
    """Safely make directory."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def _sum_raster(raster_path):
    """Return sum of non-nodata values in ``raster_path``."""
    nodata = pygeoprocessing.get_raster_info(raster_path)['nodata'][0]
    running_sum = 0.0
    for _, raster_block in pygeoprocessing.iterblocks((raster_path, 1)):
        running_sum += numpy.sum(
            raster_block[~numpy.isclose(raster_block, nodata)])
    return running_sum


def _replace_value_by_mask(
        base_raster_path, replacement_value,
        replacement_mask_raster_path, target_replacement_raster_path):
    """Overwrite values in raster based on mask.

    Args:
        base_raster_path (str): base raster to modify
        replacement_value (numeric): value to write into base raster
            where the mask indicates.
        replacement_mask_raster_path (str): path to raster indicating (1) where
            a pixel should be replaced in base.
        target_replacement_raster_path (str): path to a target replacement
            raster.

    Returns:
        None
    """
    base_info = pygeoprocessing.get_raster_info(base_raster_path)
    pygeoprocessing.new_raster_from_base(
        base_raster_path, target_replacement_raster_path,
        base_info['datatype'], base_info['nodata'])
    target_raster = gdal.OpenEx(
        target_replacement_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    target_band = target_raster.GetRasterBand(1)
    mask_raster = gdal.OpenEx(
        replacement_mask_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    mask_band = mask_raster.GetRasterBand(1)

    for offset_dict, base_block in pygeoprocessing.iterblocks(
            (base_raster_path, 1)):
        mask_block = mask_band.ReadAsArray(**offset_dict)
        target_band.WriteArray(
            numpy.where(mask_block == 1), replacement_value, base_block)

    target_band = None
    target_raster = None


def _greedy_select_pixels_to_area(
        base_value_raster_path, workspace_dir, area_ha_to_step_report,
        target_area_ha):
    """Greedy select pixels in base with a report every area steps.

    workspace_dir will contain a set of mask rasters with filenames of the form
    {area_selected}_mask_{base_id}.tif and a csv table with the filename
    {base_id}_{target_area_ha}_report.csv containing columns (area slected),
    (sum of value selected), (path to raster mask).

    Args:
        base_value_raster_path (str): path to raster with value pixels,
            preferably positive.
        workspace_dir (str): path to directory to write output files into.
        area_ha_to_step_report (float): amount of area selected in hectares
            to write to the .csv table and save a raster mask for.
        target_area_ha (float): maximum amount of area to select before
            terminating. If target_area_ha > maximum area possible to select
            it will terminate at maxium possible area to select.

    Returns:
        A tuple containing (path_to_taret_area_mask_raster,
            maximum area selected), where the raster is the largest amount
            selected and the value is the area that is selected, will either
            be very close to target_area_ha or the maximum available area.
    """
    pass


def _diff_rasters(
        a_raster_path, b_raster_path, target_diff_raster_path):
    """Calculate a-b.

    Args:
        a_raster_path (str): raster A, same size as B
        b_raster_path (str): raster B
        target_diff_raster_path (str): result of A-B accounting for nodata.

    Returns:
        None
    """
    pass


def _calculate_modeled_biomass(
        landcover_raster_path, churn_dir,
        target_biomass_raster_path):
    """Calculate modeled biomass for given landcover.

    Args:
        landcover_raster_path (str): path to ESA landcover raster.
        churn_dir (str): path to use for temporary files.
        target_biomass_raster_path (str): path to raster to create target
            biomass (not biomass per ha).

    Return:
        None
    """
    landtype_mask_raster_path = os.path.join(
        churn_dir, f'''carbon_model_landtype_mask_{
            os.path.basename(os.path.splitext(
                landcover_raster_path)[0])}.tif''')

    BIOMASS_MODEL
    LOGGER.info(f'evaluate carbon model for {scenario_id}')
    carbon_edge_model.evaluate_model_with_landcover(
        carbon_model, scenario_mask_path,
        convolution_file_paths, workspace_dir, churn_dir, args.n_workers,
        args.file_suffix, task_graph)
    modeled_biomass_raster_dict[REGRESSION_MODE][scenario_id] = \
        target_regression_biomass_paths
    # TODO: convert biomass density into biomass stocks


def _calculate_ipcc_biomass(
        landcover_raster_path, churn_dir, target_biomass_raster_path):
    """Calculate IPCC method for biomass for given landcover.

    Args:
        landcover_raster_path (str): path to ESA landcover raster.
        churn_dir (str): path to use for temporary files.
        target_biomass_raster_path (str): path to raster to create target
            biomass (not in density)

    Return:
        None
    """
    basename = os.path.basename(os.path.splitext(landcover_raster_path)[0])
    LOGGER.info(f'calculate IPCC biomass for {basename}')
    # TODO: convert to biomass not just /ha


def main():
    """Entry point."""
    for dir_path in [
            WORKSPACE_DIR, CHURN_DIR, BIOMASS_RASTER_DIR,
            MARGINAL_VALUE_WORKSPACE, OPTIMIZATION_WORKSPACE,
            OPTIMIAZATION_SCENARIOS_DIR]:
        _mkdir(dir_path)

    # TODO: task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, 2, 15.0)

    # modeled_biomass_raster_dict indexed by
    #   [MODELED_MODE/IPCC_MODE] -> [BASE_SCENARIO/RESTORATION_SCENARIO]
    modeled_biomass_raster_dict = collections.defaultdict(dict)
    for scenario_id, landcover_raster_path in [
            (BASE_SCENARIO, BASE_LULC_RASTER_PATH),
            (RESTORATION_SCENARIO, ESA_RESTORATION_SCENARIO_RASTER_PATH)]:
        # create churn directory and id for modeled biomass.
        base_landcover_id = os.path.basename(
            os.path.splitext(landcover_raster_path)[0])
        biomass_churn_dir = _mkdir(os.path.join(
            CHURN_DIR, f'churn_{base_landcover_id}_{MODELED_MODE}'))

        # calculated modeled biomass
        LOGGER.info(
            f'model biomass {MODELED_MODE} for {base_landcover_id}/'
            f'{scenario_id}')
        modeled_biomass_raster_path = os.path.join(
            BIOMASS_RASTER_DIR,
            f'biomass_{MODELED_MODE}_{scenario_id}.tif')
        _calculate_modeled_biomass(
            landcover_raster_path, biomass_churn_dir,
            modeled_biomass_raster_path)
        modeled_biomass_raster_dict[MODELED_MODE][scenario_id] = \
            modeled_biomass_raster_path

        # calculate IPCC biomass
        LOGGER.info(
            f'calculate IPCC method for {base_landcover_id}/'
            f'{scenario_id}')
        target_ipcc_biomass_path = os.path.join(
            BIOMASS_RASTER_DIR,
            f'biomass_per_ha_{IPCC_MODE}_{scenario_id}.tif')
        ipcc_churn_dir = os.path.join(
            CHURN_DIR, f'churn_{base_landcover_id}_{IPCC_MODE}')
        _calculate_ipcc_biomass(
            landcover_raster_path, ipcc_churn_dir, target_ipcc_biomass_path)
        modeled_biomass_raster_dict[IPCC_MODE][scenario_id] = \
            target_ipcc_biomass_path

    LOGGER.info('create marginal value maps')
    # optimal_lulc_scenario_raster_dict indexed by
    #   [MODELED_MODE/IPCC_MODE]
    optimal_lulc_scenario_raster_dict = {}
    for model_mode in [MODELED_MODE, IPCC_MODE]:
        marginal_value_biomass_raster = os.path.join(
            MARGINAL_VALUE_WORKSPACE,
            f'marginal_value_biomass_{model_mode}.tif')
        _diff_rasters(
            modeled_biomass_raster_dict[model_mode][RESTORATION_SCENARIO],
            modeled_biomass_raster_dict[model_mode][BASE_SCENARIO],
            marginal_value_biomass_raster)

        LOGGER.info(
            f'create optimal land selection mask to target '
            f'{TARGET_AREA_HA} ha')
        optimization_dir = _mkdir(os.path.join(
            OPTIMIZATION_WORKSPACE, f'optimization_{model_mode}'))
        # returns a (optimal mask, area selected) tuple
        optimal_mask_raster_path, area_selected = \
            _greedy_select_pixels_to_area(
                marginal_value_biomass_raster, optimization_dir,
                AREA_REPORT_STEP_AMOUNT_HA, TARGET_AREA_HA)

        # Evaluate the optimal result for biomass, first convert optimal
        # scenario to LULC map
        optimal_scenario_lulc_raster_path = os.path.join(
            OPTIMIAZATION_SCENARIOS_DIR,
            f'optimal_scenario_lulc_'
            f'{model_mode}_{TARGET_AREA_HA}_ha.tif')
        _replace_value_by_mask(
            BASE_LULC_RASTER_PATH, FOREST_CODE, optimal_mask_raster_path,
            optimal_scenario_lulc_raster_path)
        optimal_lulc_scenario_raster_dict[model_mode] = \
            optimal_scenario_lulc_raster_path

    # evaluate the MODELED driven optimal scenario with the biomass model
    optimal_modeled_churn_dir = os.path.join(
        CHURN_DIR, f'churn_optimal_{MODELED_MODE}_{TARGET_AREA_HA}_ha')
    optimal_biomass_modeled_raster_path = os.path.join(
        WORKSPACE_DIR,
        f'optimal_scenario_biomass_{MODELED_MODE}_{TARGET_AREA_HA}_ha.tif')
    _calculate_modeled_biomass(
        optimal_lulc_scenario_raster_dict[MODELED_MODE],
        optimal_modeled_churn_dir, optimal_biomass_modeled_raster_path)

    # evaluate the IPCC driven optimal scenario with the biomass model
    optimal_ipcc_churn_dir = os.path.join(
        CHURN_DIR,
        f'churn_optimal_{IPCC_MODE}_{TARGET_AREA_HA}_ha')
    optimal_biomass_ipcc_raster_path = os.path.join(
        WORKSPACE_DIR,
        f'optimal_scenario_biomass_{IPCC_MODE}_{TARGET_AREA_HA}.tif')
    _calculate_modeled_biomass(
        optimal_lulc_scenario_raster_dict[IPCC_MODE],
        optimal_ipcc_churn_dir, optimal_biomass_ipcc_raster_path)

    LOGGER.info(
        'calculate difference between modeled biomass optimization and IPCC '
        'optimization')
    modeled_vs_optimal_biomass_diff_raster_dict = {}
    for model_mode, optimal_biomass_raster_path in [
            (IPCC_MODE, optimal_biomass_ipcc_raster_path),
            (MODELED_MODE, optimal_biomass_modeled_raster_path),
            ]:
        modeled_vs_optimal_biomass_diff_raster_path = os.path.join(
            WORKSPACE_DIR,
            f'optimal_{model_mode}_biomass_gain_{TARGET_AREA_HA}_ha.tif')
        _diff_rasters(
            optimal_biomass_raster_path,
            modeled_biomass_raster_dict[MODELED_MODE][BASE_SCENARIO],
            modeled_vs_optimal_biomass_diff_raster_path)
        modeled_vs_optimal_biomass_diff_raster_dict[model_mode] = \
            modeled_vs_optimal_biomass_diff_raster_path

    LOGGER.info('report')
    report_csv_path = os.path.join(
        WORKSPACE_DIR, f'report_{MODELED_MODE}_vs_{IPCC_MODE}.csv')
    with open(report_csv_path, 'w') as report_csv_file:
        report_csv_file.write(
            'description,biomass sum,raster path\n')
        for description, raster_path in [
            ('base scenario modeled with IPCC',
             modeled_biomass_raster_dict[IPCC_MODE][BASE_SCENARIO]),
            ('restoration scenario modeled with IPCC',
             modeled_biomass_raster_dict[IPCC_MODE][RESTORATION_SCENARIO]),
            (f'base scenario modeled with {MODELED_MODE}',
             modeled_biomass_raster_dict[MODELED_MODE][BASE_SCENARIO]),
            (f'restoration scenario modeled with {MODELED_MODE}',
             modeled_biomass_raster_dict[MODELED_MODE][RESTORATION_SCENARIO]),
            (f'optimal {TARGET_AREA_HA} ha target driven by {MODELED_MODE}',
             optimal_biomass_modeled_raster_path),
            (f'optimal {TARGET_AREA_HA} ha target driven by {IPCC_MODE}',
             optimal_biomass_ipcc_raster_path),
            (f'optimal {TARGET_AREA_HA} ha gain driven by {MODELED_MODE}',
             modeled_vs_optimal_biomass_diff_raster_dict[MODELED_MODE]),
            (f'optimal {TARGET_AREA_HA} ha gain driven by {IPCC_MODE}',
             modeled_vs_optimal_biomass_diff_raster_dict[IPCC_MODE])]:

            biomass_sum = _sum_raster(raster_path)
            report_csv_file.write(
                f'{description},{biomass_sum},{raster_path}\n')


if __name__ == '__main__':
    main()
