"""Compare ESA vs DNN modeling."""
import collections
import glob
import os
import logging
import re
import shutil

from osgeo import gdal
import numpy
import ecoshard.geoprocessing
import taskgraph
import tempfile

import dnn_model
from utils.density_per_ha_to_total_per_pixel import \
    density_per_ha_to_total_per_pixel
from train_model import make_kernel_raster

gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))

LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.DEBUG)


# Working directories for substeps
def _mkdir(dir_path):
    """Safely make directory."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


WORKSPACE_DIR = _mkdir('./ipcc_vs_dnn_optimization')
CHURN_DIR = _mkdir(os.path.join(WORKSPACE_DIR, 'churn'))
BIOMASS_RASTER_DIR = _mkdir(
    os.path.join(WORKSPACE_DIR, 'biomass_rasters'))
MARGINAL_VALUE_WORKSPACE = _mkdir(
    os.path.join(WORKSPACE_DIR, 'marginal_value_rasters'))
OPTIMIZATION_WORKSPACE = _mkdir(
    os.path.join(WORKSPACE_DIR, 'optimization_workspaces'))
NEW_FOREST_MASK_DIR = _mkdir(
    os.path.join(WORKSPACE_DIR, 'new_forest_masks'))
IPCC_VS_DNN_DIR = _mkdir(
    os.path.join(WORKSPACE_DIR, 'ipcc_vs_dnn'))

MODEL_PATH = './models/model_400.dat'
BASE_DATA_DIR = 'workspace/ecoshard'

# *** DATA SECTION ***
# There are two landcover configurations, ESA and restoration of ESA
BASE_LULC_RASTER_PATH = os.path.join(
    BASE_DATA_DIR,
    'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif')
ESA_RESTORATION_SCENARIO_RASTER_PATH = os.path.join(
    BASE_DATA_DIR,
    'restoration_limited_md5_372bdfd9ffaf810b5f68ddeb4704f48f.tif')

# These are used in combination with an ESA landcover map to calculate carbon
CARBON_ZONES_VECTOR_PATH = os.path.join(
    BASE_DATA_DIR,
    'carbon_zones_md5_aa16830f64d1ef66ebdf2552fb8a9c0d.gpkg')
IPCC_CARBON_TABLE_PATH = os.path.join(
    BASE_DATA_DIR,
    'IPCC_carbon_table_md5_a91f7ade46871575861005764d85cfa7.csv')

# Constants useful for code readability
CARBON_MODEL_ID = os.path.basename(os.path.splitext(MODEL_PATH)[0])
IPCC_MODE = 'ipcc_mode'
# model mode is based off of carbon model ID
MODELED_MODE = f'modeled_mode_{CARBON_MODEL_ID}'
BASE_SCENARIO = 'base'
RESTORATION_SCENARIO = 'scenario'
FOREST_CODE = 50
TARGET_AREA_HA = 350000000*2
AREA_N_STEPS = 20
AREA_REPORT_STEP_LIST = numpy.linspace(
    TARGET_AREA_HA / AREA_N_STEPS, TARGET_AREA_HA, AREA_N_STEPS)
# number of pixels to blur to capture edge effect of marginal value
MARGINAL_VALUE_PIXEL_BLUR = 16


def _raw_basename(file_path):
    """Return just the filename without extension."""
    return os.path.basename(os.path.splitext(file_path)[0])[-20:]


def _sum_raster(raster_path):
    """Return sum of non-nodata values in ``raster_path``."""
    nodata = ecoshard.geoprocessing.get_raster_info(raster_path)['nodata'][0]
    running_sum = 0.0
    for _, raster_block in ecoshard.geoprocessing.iterblocks((raster_path, 1)):
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
    base_info = ecoshard.geoprocessing.get_raster_info(base_raster_path)
    ecoshard.geoprocessing.new_raster_from_base(
        base_raster_path, target_replacement_raster_path,
        base_info['datatype'], base_info['nodata'])
    target_raster = gdal.OpenEx(
        target_replacement_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    target_band = target_raster.GetRasterBand(1)
    mask_raster = gdal.OpenEx(
        replacement_mask_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    mask_band = mask_raster.GetRasterBand(1)

    for offset_dict, base_block in ecoshard.geoprocessing.iterblocks(
            (base_raster_path, 1)):
        mask_block = mask_band.ReadAsArray(**offset_dict)
        base_block[mask_block == 1] = replacement_value
        target_band.WriteArray(
            base_block, xoff=offset_dict['xoff'], yoff=offset_dict['yoff'])

    target_band = None
    target_raster = None


def _greedy_select_pixels_to_area(
        base_value_raster_path, workspace_dir, area_ha_to_step_report_list):
    """Greedy select pixels in base with a report every area steps.

    workspace_dir will contain a set of mask rasters with filenames of the form
    {area_selected}_mask_{base_id}.tif and a csv table with the filename
    {base_id}_{target_area_ha}_report.csv containing columns (area slected),
    (sum of value selected), (path to raster mask).

    Args:
        base_value_raster_path (str): path to raster with value pixels,
            preferably positive.
        workspace_dir (str): path to directory to write output files into.
        area_ha_to_step_report (list): list of areas in Ha to record.

    Returns:
        A tuple containing (path_to_taret_area_mask_raster,
            maximum area selected), where the raster is the largest amount
            selected and the value is the area that is selected, will either
            be very close to target_area_ha or the maximum available area.
    """
    raster_id = _raw_basename(base_value_raster_path)
    all_ones_raster_path = os.path.join(
        workspace_dir, f'all_ones_{raster_id}.tif')
    pixel_area_in_ha_raster_path = os.path.join(
        workspace_dir, f'pixel_area_in_ha_{raster_id}.tif')

    ecoshard.geoprocessing.new_raster_from_base(
        base_value_raster_path, all_ones_raster_path, gdal.GDT_Byte,
        [None], fill_value_list=[1])

    density_per_ha_to_total_per_pixel(
        all_ones_raster_path, 1.0,
        pixel_area_in_ha_raster_path)

    LOGGER.info(
        f'calculating greedy pixels for value raster {base_value_raster_path} '
        f'and area {pixel_area_in_ha_raster_path}')
    ecoshard.geoprocessing.greedy_pixel_pick_by_area_v2(
        (base_value_raster_path, 1), (pixel_area_in_ha_raster_path, 1),
        area_ha_to_step_report_list, workspace_dir)
    LOGGER.debug(
        f'done with greedy pixels for value raster {base_value_raster_path}')


def _create_marginal_value_layer(
        future_raster_path, base_raster_path,
        gaussian_blur_pixel_radius, mask_raster_path, target_raster_path):
    """Calculate marginal value layer.

    Calculated by taking the difference of future from base, Gaussian blurring
    that result by the given radius, and masking by the given raster mask.

    Args:
        future_raster_path (str): raster A, same nodata and size as B
        base_raster_path (str): raster B
        gaussian_blur_pixel_radius (int): number of pixels to blur out when
            determining marginal value of that pixel.
        mask_raster_path (str): path to raster where anything not 1 is masked
            to 0/nodata.
        target_diff_raster_path (str): result of A-B accounting for nodata.

    Returns:
        None
    """
    raster_info = ecoshard.geoprocessing.get_raster_info(future_raster_path)
    nodata = raster_info['nodata'][0]

    def _diff_op(a_array, b_array):
        """Return a-b and consider nodata."""
        result = numpy.copy(a_array)
        valid_mask = ~numpy.isclose(a_array, nodata)
        result[valid_mask] -= b_array[valid_mask]
        return result

    churn_dir = tempfile.mkdtemp(dir=os.path.dirname(target_raster_path))
    diff_raster_path = os.path.join(churn_dir, 'diff.tif')
    ecoshard.geoprocessing.raster_calculator(
        [(future_raster_path, 1), (base_raster_path, 1)], _diff_op,
        diff_raster_path, raster_info['datatype'], nodata)

    # Gaussian filter
    if gaussian_blur_pixel_radius is not None:
        kernel_raster_path = os.path.join(churn_dir, 'kernel.tif')
        mask_gf_path = os.path.join(churn_dir, 'gf.tif')
        if os.path.exists(mask_gf_path):
            os.remove(mask_gf_path)
        make_kernel_raster(
            gaussian_blur_pixel_radius, kernel_raster_path)
        ecoshard.geoprocessing.convolve_2d(
            (diff_raster_path, 1), (kernel_raster_path, 1), mask_gf_path,
            ignore_nodata_and_edges=False, mask_nodata=True,
            target_nodata=0.0)
    else:
        mask_gf_path = diff_raster_path

    def _mask_op(base_array, mask_array):
        """Return base where mask is 1, otherwise 0 or nodata."""
        result = numpy.copy(base_array)
        zero_mask = (~numpy.isclose(base_array, nodata)) & (mask_array != 1)
        result[zero_mask] = 0
        return result

    ecoshard.geoprocessing.raster_calculator(
        [(mask_gf_path, 1), (mask_raster_path, 1)], _mask_op,
        target_raster_path, raster_info['datatype'], nodata)

    shutil.rmtree(churn_dir)


def _diff_rasters(
        a_raster_path, b_raster_path, target_diff_raster_path):
    """Calculate a-b.

    Args:
        a_raster_path (str): raster A, same nodata and size as B
        b_raster_path (str): raster B
        target_diff_raster_path (str): result of A-B accounting for nodata.

    Returns:
        None
    """
    raster_info = ecoshard.geoprocessing.get_raster_info(a_raster_path)
    nodata = raster_info['nodata'][0]

    def _diff_op(a_array, b_array):
        """Return a-b and consider nodata."""
        result = numpy.copy(a_array)
        valid_mask = ~numpy.isclose(a_array, nodata)
        result[valid_mask] -= b_array[valid_mask]
        return result

    ecoshard.geoprocessing.raster_calculator(
        [(a_raster_path, 1), (b_raster_path, 1)], _diff_op,
        target_diff_raster_path, raster_info['datatype'], nodata)


def _calculate_new_forest(
        base_lulc_raster_path, future_lulc_raster_path,
        new_forest_mask_raster_path):
    """Calculate where there is new forest from base to future.

    Args:
        base_lulc_raster_path (str):
        future_lulc_raster_path (str):
        new_forest_mask_raster_path (str):

    Returns:
        None
    """
    FOREST_CODES = (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170)

    def _mask_new_forest(base, future):
        """Remap values from ESA codes to basic MASK_TYPES."""
        result = numpy.empty(base.shape, dtype=numpy.uint8)
        base_forest = numpy.in1d(base, FOREST_CODES).reshape(result.shape)
        future_forest = numpy.in1d(future, FOREST_CODES).reshape(result.shape)
        result[:] = future_forest & ~base_forest
        return result

    ecoshard.geoprocessing.raster_calculator(
        [(base_lulc_raster_path, 1), (future_lulc_raster_path, 1)],
        _mask_new_forest, new_forest_mask_raster_path, gdal.GDT_Byte, None)


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
    def _ipcc_carbon_op(
            lulc_array, zones_array, zone_lulc_to_carbon_map):
        """Map carbon to LULC/zone values and multiply by conversion map."""
        result = numpy.zeros(lulc_array.shape)
        for zone_id in numpy.unique(zones_array):
            if zone_id in zone_lulc_to_carbon_map:
                zone_mask = zones_array == zone_id
                result[zone_mask] = (
                    zone_lulc_to_carbon_map[zone_id][lulc_array[zone_mask]])
        return result

    def _parse_carbon_lulc_table(ipcc_carbon_table_path):
        """Parse out the IPCC carbon table by zone and lulc."""
        with open(IPCC_CARBON_TABLE_PATH, 'r') as carbon_table_file:
            header_line = carbon_table_file.readline()
            lulc_code_list = [
                int(lucode) for lucode in header_line.split(',')[1:]]
            max_code = max(lulc_code_list)

            zone_lucode_to_carbon_map = {}
            for line in carbon_table_file:
                split_line = line.split(',')
                if split_line[0] == '':
                    continue
                zone_id = int(split_line[0])
                zone_lucode_to_carbon_map[zone_id] = numpy.zeros(max_code+1)
                for lucode, carbon_value in zip(
                        lulc_code_list, split_line[1:]):
                    zone_lucode_to_carbon_map[zone_id][lucode] = float(
                        carbon_value)
        return zone_lucode_to_carbon_map

    rasterized_zones_raster_path = os.path.join(churn_dir, 'carbon_zones.tif')
    LOGGER.info(
        f'rasterize carbon zones of {landcover_raster_path} to '
        f'{rasterized_zones_raster_path}')
    ecoshard.geoprocessing.new_raster_from_base(
        landcover_raster_path, rasterized_zones_raster_path, gdal.GDT_Int32,
        [-1])
    ecoshard.geoprocessing.rasterize(
        CARBON_ZONES_VECTOR_PATH, rasterized_zones_raster_path,
        option_list=['ATTRIBUTE=CODE'])

    zone_lucode_to_carbon_map = _parse_carbon_lulc_table(
        IPCC_CARBON_TABLE_PATH)

    # uncommenting this so we get the density
    #biomass_per_ha_raster_path = os.path.join(churn_dir, 'biomass_per_ha.tif')
    ecoshard.geoprocessing.raster_calculator(
        [(landcover_raster_path, 1), (rasterized_zones_raster_path, 1),
         (zone_lucode_to_carbon_map, 'raw')],
        _ipcc_carbon_op, target_biomass_raster_path,
        gdal.GDT_Float32, -1)

    #density_per_ha_to_total_per_pixel(
    #    biomass_per_ha_raster_path, 1.0,
    #    target_biomass_raster_path)


def _calculate_modeled_biomass_from_mask(
        base_lulc_raster_path, new_forest_mask_raster_path,
        target_biomass_raster_path, task_graph):
    """Calculate new biomass raster from base layer and new forest mask.

    Args:
        base_lulc_raster_path (str): path to base ESA LULC raster.
        new_forest_mask_raster_path (str): path to raster that indicates
            where new forest is applied with a 1.
        target_biomass_raster_path (str): created by this function, a
            raster that has biomass per pixel for the scenario given by
            new_forest_mask_raster_path from base_lulc_raster_path.
        n_workers (int): number of workers to allow for reprojection.

    Returns:
        None
    """
    churn_dir = os.path.join(
        os.path.dirname(target_biomass_raster_path),
        os.path.basename(os.path.splitext(target_biomass_raster_path)[0]))

    # this raster is base with new forest in it
    converted_lulc_raster_path = os.path.join(churn_dir, 'converted_lulc.tif')
    LOGGER.info(
        f'creating converted LULC off of {base_lulc_raster_path} to '
        f'{converted_lulc_raster_path}')
    replace_value_by_mask_task = task_graph.add_task(
        func=_replace_value_by_mask,
        args=(
            base_lulc_raster_path, FOREST_CODE, new_forest_mask_raster_path,
            converted_lulc_raster_path),
        target_path_list=[converted_lulc_raster_path],
        task_name=f'replace by mask to {converted_lulc_raster_path}')
    replace_value_by_mask_task.join()

    # calculate biomass for that raster
    model_task = dnn_model.run_model(
        converted_lulc_raster_path,
        MODEL_PATH, target_biomass_raster_path, base_task_graph=task_graph)
    return model_task


def main():
    """Entry point."""
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1)

    unique_scenario_id = f'''{
        _raw_basename(BASE_LULC_RASTER_PATH)}_{
        _raw_basename(ESA_RESTORATION_SCENARIO_RASTER_PATH)}'''

    LOGGER.info(f'calculate new forest mask on {BASE_LULC_RASTER_PATH}')
    new_forest_raster_path = os.path.join(
        NEW_FOREST_MASK_DIR, f'{unique_scenario_id}.tif')
    new_forest_mask_task = task_graph.add_task(
        func=_calculate_new_forest,
        args=(
            BASE_LULC_RASTER_PATH, ESA_RESTORATION_SCENARIO_RASTER_PATH,
            new_forest_raster_path),
        target_path_list=[new_forest_raster_path],
        task_name=f'create forest mask for {new_forest_raster_path}')

    # modeled_biomass_raster_task_dict indexed by
    #   [MODELED_MODE/IPCC_MODE] -> [BASE_SCENARIO/RESTORATION_SCENARIO]
    modeled_biomass_raster_task_dict = collections.defaultdict(dict)
    for scenario_id, landcover_raster_path in [
            (BASE_SCENARIO, BASE_LULC_RASTER_PATH),
            (RESTORATION_SCENARIO, ESA_RESTORATION_SCENARIO_RASTER_PATH)]:
        # create churn directory and id for modeled biomass.
        base_landcover_id = os.path.basename(
            os.path.splitext(landcover_raster_path)[0])

        # calculated modeled biomass
        LOGGER.info(
            f'model biomass {MODELED_MODE} for {base_landcover_id}/'
            f'{scenario_id}')
        modeled_biomass_raster_path = os.path.join(
            BIOMASS_RASTER_DIR,
            f'biomass_{MODELED_MODE}_{scenario_id}.tif')
        biomass_model_task = dnn_model.run_model(
            landcover_raster_path, MODEL_PATH,
            modeled_biomass_raster_path, base_task_graph=task_graph)
        biomass_model_task.join()

        modeled_biomass_raster_task_dict[MODELED_MODE][scenario_id] = \
            (modeled_biomass_raster_path, biomass_model_task)

        # calculate IPCC biomass
        LOGGER.info(
            f'calculate IPCC method for {base_landcover_id}/'
            f'{scenario_id}')
        target_ipcc_biomass_path = os.path.join(
            BIOMASS_RASTER_DIR,
            f'biomass_per_ha_{IPCC_MODE}_{scenario_id}.tif')
        ipcc_churn_dir = os.path.join(
            CHURN_DIR, f'churn_{base_landcover_id}_{IPCC_MODE}')
        biomass_ipcc_task = task_graph.add_task(
            func=_calculate_ipcc_biomass,
            args=(
                landcover_raster_path, ipcc_churn_dir,
                target_ipcc_biomass_path),
            target_path_list=[target_ipcc_biomass_path],
            task_name=f'calculate biomass{IPCC_MODE} for {scenario_id}')
        modeled_biomass_raster_task_dict[IPCC_MODE][scenario_id] = \
            (target_ipcc_biomass_path, biomass_ipcc_task)

    LOGGER.info('create marginal value maps')
    # this will have (mode, task, dir) tuples for this section
    optimization_mode_task_dir_list = []
    for model_mode in [MODELED_MODE, IPCC_MODE]:
        marginal_value_biomass_raster = os.path.join(
            MARGINAL_VALUE_WORKSPACE,
            f'marginal_value_biomass_{model_mode}.tif')
        restoration_biomass_raster, restoration_task = \
            modeled_biomass_raster_task_dict[model_mode][RESTORATION_SCENARIO]
        base_biomass_raster, base_task = \
            modeled_biomass_raster_task_dict[model_mode][BASE_SCENARIO]

        marginal_value_task = task_graph.add_task(
            func=_create_marginal_value_layer,
            args=(
                restoration_biomass_raster,
                base_biomass_raster,
                (MARGINAL_VALUE_PIXEL_BLUR
                 if model_mode == MODELED_MODE else None),
                new_forest_raster_path,
                marginal_value_biomass_raster),
            target_path_list=[marginal_value_biomass_raster],
            dependent_task_list=[
                restoration_task, base_task, new_forest_mask_task],
            task_name=(
                f'''calc marginal value for {restoration_biomass_raster} '''
                f'''and {base_biomass_raster}'''))

        LOGGER.info(
            f'create optimal land selection mask to target '
            f'{TARGET_AREA_HA} ha')
        optimization_dir = _mkdir(os.path.join(
            OPTIMIZATION_WORKSPACE,
            f'optimization_{unique_scenario_id}_{model_mode}'))
        # returns a (optimal mask, area selected) tuple
        optimization_task = task_graph.add_task(
            func=_greedy_select_pixels_to_area,
            args=(
                marginal_value_biomass_raster, optimization_dir,
                AREA_REPORT_STEP_LIST),
            target_path_list=[os.path.join(optimization_dir, f'marginal_value_biomass_{model_mode}_greedy_pick.csv')],
            dependent_task_list=[marginal_value_task],
            task_name=f'optimize on {marginal_value_biomass_raster}')
        optimization_mode_task_dir_list.append(
            (model_mode, optimization_task, optimization_dir))

    # indexed by MODELED_MODE vs. IPCC_MODE then by area of the new forest
    optimization_biomass_area_path_task_dict = \
        collections.defaultdict(dict)
    for (model_mode, optimization_task, optimization_dir) in \
            optimization_mode_task_dir_list:
        # okay to join here because it's going to trigger a whole set of
        # other tasks and nothing can be done until this one is ready anyway
        optimization_task.join()

        optimization_biomass_dir = _mkdir(os.path.join(
            OPTIMIZATION_WORKSPACE,
            f'biomass_{unique_scenario_id}_{model_mode}'))

        for optimal_mask_raster_path in glob.glob(
                os.path.join(optimization_dir, 'mask_*.tif')):
            optimization_biomass_raster_path = os.path.join(
                optimization_biomass_dir,
                f'''biomass_per_pixel_{
                    _raw_basename(optimal_mask_raster_path)}.tif''')
            model_task = _calculate_modeled_biomass_from_mask(
                BASE_LULC_RASTER_PATH, optimal_mask_raster_path,
                optimization_biomass_raster_path, task_graph)
            mask_area = float(re.match(
                r'mask_(.*)\.tif', os.path.basename(
                    optimal_mask_raster_path)).group(1))
            optimization_biomass_area_path_task_dict[
                model_mode][mask_area] = (
                    optimization_biomass_raster_path, model_task)

    # TODO: calculate difference between modeled vs IPCC
    task_graph.join()
    LOGGER.info(
        'calculate difference between modeled biomass optimization and IPCC '
        'optimization')
    mask_areas = sorted([
        float(x) for x in
        optimization_biomass_area_path_task_dict[MODELED_MODE].keys()])
    modeled_diff_ipcc_biomass_sum_task_list = []
    modeled_diff_mode_base_biomass_sum_task_list = \
        collections.defaultdict(list)
    LOGGER.debug(f'********* {mask_areas} {optimization_dir}')
    for mask_area in mask_areas:
        model_biomass_raster_path, dnn_model_task = \
            optimization_biomass_area_path_task_dict[MODELED_MODE][mask_area]
        ipcc_biomass_raster_path, ipcc_model_task = \
            optimization_biomass_area_path_task_dict[IPCC_MODE][mask_area]
        modeled_vs_ipcc_optimal_biomass_diff_raster_path = os.path.join(
            IPCC_VS_DNN_DIR,
            f'modeled_vs_ipcc_diff_{mask_area}_ha.tif')
        diff_task = task_graph.add_task(
            func=_diff_rasters,
            args=(
                model_biomass_raster_path,
                ipcc_biomass_raster_path,
                modeled_vs_ipcc_optimal_biomass_diff_raster_path),
            target_path_list=[
                modeled_vs_ipcc_optimal_biomass_diff_raster_path],
            dependent_task_list=[dnn_model_task, ipcc_model_task],
            task_name=f'''modeled diff ipcc {
                modeled_vs_ipcc_optimal_biomass_diff_raster_path}''')

        sum_task = task_graph.add_task(
            func=_sum_raster,
            args=(modeled_vs_ipcc_optimal_biomass_diff_raster_path,),
            store_result=True,
            dependent_task_list=[diff_task],
            task_name=f'''sum the modeled vs. ippc diff for {
                modeled_vs_ipcc_optimal_biomass_diff_raster_path}''')
        modeled_diff_ipcc_biomass_sum_task_list.append(sum_task)

        LOGGER.info(
            'subtract modeled and ipcc from base modeled to get the gain')
        modeled_vs_base_biomass_diff_raster_path, dnn_model_task = os.path.join(
            IPCC_VS_DNN_DIR, f'modeled_vs_base_{mask_area}_ha.tif')
        ipcc_vs_base_biomass_diff_raster_path, ipcc_model_task = os.path.join(
            IPCC_VS_DNN_DIR, f'ipcc_vs_base_{mask_area}_ha.tif')
        modeled_base_biomass_raster_path = (
            modeled_biomass_raster_task_dict[MODELED_MODE][BASE_SCENARIO][0])
        for modeled_path, target_diff_path, mode, model_task in [
                (model_biomass_raster_path,
                 modeled_vs_base_biomass_diff_raster_path, MODELED_MODE, dnn_model_task),
                (ipcc_biomass_raster_path,
                 ipcc_vs_base_biomass_diff_raster_path, IPCC_MODE, ipcc_model_task)]:
            diff_task = task_graph.add_task(
                func=_diff_rasters,
                args=(
                    modeled_path, modeled_base_biomass_raster_path,
                    target_diff_path),
                dependent_task_list=[model_task],
                target_path_list=[target_diff_path],
                task_name=f'modeled diff ipcc {target_diff_path}')

            sum_task = task_graph.add_task(
                func=_sum_raster,
                args=(target_diff_path,),
                store_result=True,
                dependent_task_list=[diff_task],
                task_name=f'''sum the modeled/ipcc vs. base for {
                    target_diff_path}''')
            modeled_diff_mode_base_biomass_sum_task_list[mode].append(sum_task)

    task_graph.join()

    LOGGER.info('report')
    report_csv_path = os.path.join(
        WORKSPACE_DIR, f'''report_{_raw_basename(BASE_LULC_RASTER_PATH)}_{
        _raw_basename(ESA_RESTORATION_SCENARIO_RASTER_PATH)}.csv''')
    with open(report_csv_path, 'w') as report_csv_file:
        report_csv_file.write(
            'area,biomass gain modeled driven, biomass gain IPCC driven, '
            'modeled vs ipcc diff\n')

        for (area, biomass_modeled_gain, ipcc_modeled_gain,
                modeled_vs_ipcc) in zip(
                mask_areas,
                modeled_diff_mode_base_biomass_sum_task_list[MODELED_MODE],
                modeled_diff_mode_base_biomass_sum_task_list[IPCC_MODE],
                modeled_diff_ipcc_biomass_sum_task_list):
            report_csv_file.write(
                f'{area},{biomass_modeled_gain.get()},'
                f'{ipcc_modeled_gain.get()},{modeled_vs_ipcc.get()}\n')
    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()
