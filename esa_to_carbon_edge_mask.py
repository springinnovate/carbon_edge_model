"""Convert ESA landcover type to carbon mask type."""
import argparse
import tempfile

CROPLAND_LULC_CODES = range(10, 41)
URBAN_LULC_CODES = (190,)
FOREST_CODES = (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170)

MASK_TYPES = [
    ('is_cropland_10sec', CROPLAND_LULC_CODES, ''),
    ('is_urban_10sec', URBAN_LULC_CODES, ''),
    ('not_forest_10sec', FOREST_CODES, 'inv'),
    ('forest_10sec', FOREST_CODES, '')]

0: cropland
1: urban
2: forest
3: other

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clip ESA landcover map to carbon edge land type raster.')
    parser.add_argument(
        'esa_lulc_raster_path', help='Path to ESA lulc raster.')
    parser.add_argument(
        'target_mask_raster_path', help='Path to target mask raster.')
    parser.add_argument(
        '--clipping_shapefile_path', help=(
            'Path to a shapefile that will be used for clipping. The target '
            'mask will be in the same projection as this file. If not '
            'present, the result will be the same size and projection as the '
            'base input.'))
    parser.add_argument(
        '--workspace_dir', help=(
            'Path to workspace dir, the carbon stock file will be '))
    args = parser.parse_args()

    # TODO: make workspace dir
    # TODO: reproject using warp raster:
    # pygeoprocessing.warp_raster(
    #     base_raster_path, target_pixel_size, target_raster_path,
    #     resample_method, target_bb=None, base_projection_wkt=None,
    #     target_projection_wkt=None, n_threads=None, vector_mask_options=None,
    #     gdal_warp_options=None, working_dir=None,
    #     raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
    #     osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY):
    # TODO: reclassify clipped file as the output file
    #   TODO: create mapping