"""Convert ESA landcover type to carbon mask type."""
import argparse
import os
import logging
import shutil
import tempfile

from osgeo import gdal
from osgeo import osr
import pygeoprocessing
import numpy

gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)


CROPLAND_LULC_CODES = range(10, 41)
URBAN_LULC_CODES = (190,)
FOREST_CODES = (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170)

MASK_TYPES = [
    (1, CROPLAND_LULC_CODES),
    (2, URBAN_LULC_CODES),
    (3, FOREST_CODES)]
OTHER_TYPE = 4
# 1: cropland
# 2: urban
# 3: forest
# 4: other


def _reclassify_vals_op(array):
    """Set values 1d array/array to nodata unless `inverse` then opposite."""
    result = numpy.empty(array.shape, dtype=numpy.uint8)
    result[:] = OTHER_TYPE  # default is '4 -- other'
    for mask_id, code_list in MASK_TYPES:
        mask_array = numpy.in1d(array, code_list).reshape(result.shape)
        result[mask_array] = mask_id
    return result


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

    # make workspace dir
    if args.workspace_dir:
        workspace_dir = args.workspace_dir
    else:
        workspace_dir = os.path.dirname(args.target_mask_raster_path)

    churn_dir = tempfile.mkdtemp(dir=workspace_dir)

    # TODO: reproject using warp raster:
    target_bb = None
    target_projection_wkt = None
    if args.clipping_shapefile_path:
        clip_vector_info = pygeoprocessing.get_vector_info(
            args.clipping_shapefile_path)
        target_bb = clip_vector_info['bounding_box']
        target_projection_wkt = clip_vector_info['projection_wkt']

        # 1) get centroid of bb
        bb_centroid = (
            (target_bb[2]-target_bb[0])/2,
            (target_bb[3]-target_bb[1])/2)

        # 2) convert to lat/lng
        base_projection_wkt = pygeoprocessing.get_raster_info(
            args.esa_lulc_raster_path)['projection_wkt']
        base_srs = osr.SpatialReference()
        base_srs.ImportFromWkt(base_projection_wkt)
        target_srs = osr.SpatialReference()
        target_srs.ImportFromWkt(target_projection_wkt)

        base_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        target_to_wgs84 = osr.CreateCoordinateTransformation(
            target_srs, base_srs)
        wgs84_to_target = osr.CreateCoordinateTransformation(
            base_srs, target_srs)

        bb_centroid_lng_lat = target_to_wgs84.TransformPoint(
            bb_centroid[0], bb_centroid[1])

        # 3) add lulc pixel size to it to make a pixel bb
        esa_raster_info = pygeoprocessing.get_raster_info(
            args.esa_lulc_raster_path)
        lng_lat_pixel_bb = [
            bb_centroid_lng_lat[0],
            bb_centroid_lng_lat[1],
            bb_centroid_lng_lat[0] + abs(esa_raster_info['pixel_size'][0]),
            bb_centroid_lng_lat[1] + abs(esa_raster_info['pixel_size'][1])
            ]

        # 4) reproject that pixel bb to target
        target_pixel_bb = pygeoprocessing.transform_bounding_box(
            lng_lat_pixel_bb, base_srs.ExportToWkt(),
            target_srs.ExportToWkt())

        # 5) subtract width/height and choose the smalleset for target size
        min_size = min(
            abs(target_pixel_bb[2]-target_pixel_bb[0]),
            abs(target_pixel_bb[3]-target_pixel_bb[1]),
            )
        target_pixel_size = (min_size, -min_size)

        # 6) clip raster
        reclassify_raster_path = os.path.join(churn_dir, 'clipped.tif')
        pygeoprocessing.warp_raster(
            args.esa_lulc_raster_path, target_pixel_size,
            reclassify_raster_path, 'near', target_bb=target_bb,
            base_projection_wkt=base_srs.ExportToWkt(),
            target_projection_wkt=target_srs.ExportToWkt(),
            vector_mask_options={
                'mask_vector_path': args.clipping_shapefile_path})
    else:
        reclassify_raster_path = args.esa_lulc_raster_path

    # reclassify clipped file as the output file
    pygeoprocessing.raster_calculator(
        [(reclassify_raster_path, 1)], _reclassify_vals_op,
        args.target_mask_raster_path, gdal.GDT_Byte, None)

    try:
        shutil.rmtree(churn_dir)
    except Exception:
        LOGGER.exception(f'error when removing {churn_dir}')
