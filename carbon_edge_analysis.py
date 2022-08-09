"""
********
Start with two LULC maps:

1) restoration_limited
2) ESACCI-LC-L4-LCCS

build two forest masks off of these:

3) restoration_limited_forest_mask
4) ESACCI-LC-L4-LCCS_forest_mask
5) ESA_to_restoration_new_forest_mask

Build ESA carbon map since the change is just static and covert to co2
* note, requires access to
    * IPCC_carbon_table_md5_a91f7ade46871575861005764d85cfa7
    * carbon_zones_md5_aa16830f64d1ef66ebdf2552fb8a9c0d

6) restoration_limited_new_forest_co2
7) ESACCI-LC-L4-LCCS_new_forest_co2

Build regression carbon maps and convert to co2

8) restoration_limited_regression_co2
9) ESACCI-LC-L4-LCCS_regression_co2

Build ESA marginal value map:

10) (calculate 6-7) ESA_marginal_value_co2_new_forest

Build regression marginal value:

* calculate convolution on new forest mask to show how many pixels from the
  new forest benefit the pixel under question
  11) new_forest_coverage_5km

* calculate 8-9 to find marginal value as a whole
  12) regression_marginal_value_raw

* calculate 12/11 to determine average coverage
  13) regression_marginal_value_average

* convolve 13 to 5km to estimate local marginal value benefit
  14) regression_marginal_value_co2_raw

* mask 14 to new forest
  15) regression_marginal_value_co2_new_forest
"""
import logging
import os
import tempfile

from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal
import pandas

gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = './output'

# Base data
LULC_RESTORATION_PATH = "./ipcc_carbon_data/.restoration_limited_md5_372bdfd9ffaf810b5f68ddeb4704f48f.tif"
LULC_ESA_PATH = "./ipcc_carbon_data/.ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif"
FOREST_LULC_CODES = (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170)

CARBON_TABLE_PATH = "./ipcc_carbon_data/IPCC_carbon_table_md5_a91f7ade46871575861005764d85cfa7.csv"
CARBON_ZONES_PATH = "./ipcc_carbon_data/carbon_zones_md5_aa16830f64d1ef66ebdf2552fb8a9c0d.gpkg"

# Forest masks created by script
FOREST_MASK_RESTORATION_PATH = './output/forest_mask_restoration_limited.tif'
FOREST_MASK_ESA_PATH = './output/forest_mask_esa.tif'
NEW_FOREST_MASK_ESA_TO_RESTORATION_PATH = './output/new_forest_mask_esa_to_restoration.tif'

IPCC_CARBON_RESTORATION_PATH = './output/ipcc_carbon_restoration_limited.tif'
IPCC_CARBON_ESA_PATH = './output/ipcc_carbon_esa.tif'


def build_ipcc_carbon(lulc_path, lulc_table_path, zone_path, lulc_codes, target_carbon_path):
    """Calculate IPCC carbon.

    Args:
        lulc_path (str): path to raster with LULC codes found in lulc_table_path and lulc_codes.
        lulc_table_path (str): maps carbon zone (rows) to lulc codes (columns) to map carbon total.
        zone_path (str): path to vector with carbon zones in 'CODE' field.
        lulc_codes (tuple): only evaluate codes in this tuple
        target_carbon_path (str): created raster that contains carbon values

    Returns:
        None
    """
    raster_info = geoprocessing.get_raster_info(lulc_path)

    # rasterize zones
    working_dir = tempfile.mkdtemp(dir=os.path.dirname(target_carbon_path))
    projected_zone_path = os.path.join(working_dir, 'proj_zone.gpkg')
    geoprocessing.reproject_vector(
        zone_path, raster_info['projection_wkt'], projected_zone_path,
        layer_id=0, driver_name='GPKG')

    zone_raster_path = os.path.join(working_dir, 'zone.tif')
    geoprocessing.new_raster_from_base(
        lulc_path, zone_raster_path, gdal.GDT_Int32, [None])

    geoprocessing.rasterize(
        projected_zone_path, zone_raster_path, option_list=['ATTRIBUTE=CODE'])
    # load table
    table = pandas.read_csv(lulc_table_path, index_col=0, header=0).to_dict()
    table = {int(k): v for k, v in table.items()}

    # raster calculator of lulc, zones, table, and codes


def main():
    """Entry point."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # create forest masks
    create_mask(LULC_RESTORATION_PATH, FOREST_LULC_CODES, FOREST_MASK_RESTORATION_PATH)
    create_mask(LULC_ESA_PATH, FOREST_LULC_CODES, FOREST_MASK_ESA_PATH)
    and_rasters(FOREST_MASK_RESTORATION_PATH, FOREST_MASK_ESA_PATH, NEW_FOREST_MASK_ESA_TO_RESTORATION_PATH)

    # Build ESA carbon map since the change is just static and covert to co2
    build_ipcc_carbon(LULC_RESTORATION_PATH, CARBON_TABLE_PATH, CARBON_ZONES_PATH, FOREST_LULC_CODES, IPCC_CARBON_RESTORATION_PATH)
    build_ipcc_carbon(LULC_ESA_PATH, CARBON_TABLE_PATH, CARBON_ZONES_PATH, FOREST_LULC_CODES, IPCC_CARBON_ESA_PATH)


if __name__ == '__main__':
    main()
