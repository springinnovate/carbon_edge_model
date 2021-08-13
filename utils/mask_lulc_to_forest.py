"""Given a mask and an LULC, create a new LULC with forest on that mask."""
import argparse
import logging

import ecoshard.geoprocessing
import numpy

from osgeo import gdal


gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))

LOGGER = logging.getLogger(__name__)


def _mask_to_forest(lulc_array, mask_array, forest_code, nodata):
    result = numpy.full(lulc_array.shape, nodata, dtype=numpy.int32)
    valid_mask = ~numpy.isclose(lulc_array, nodata) & (mask_array == 1)
    result[valid_mask] = forest_code
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create lulc with forest where mask is.')
    parser.add_argument('lulc_path', help='Path to base LULC raster')
    parser.add_argument(
        'mask_path', help='path to mask where 1 means turn to forest')
    parser.add_argument(
        'target_forest_lulc_mask_path', help='created by this call')
    parser.add_argument(
        '--forest_code', type=int, default=50,
        help='code to switch LULC to default is 50')

    args = parser.parse_args()
    nodata = ecoshard.geoprocessing.get_raster_info(
        args.base_value_raster_path)['nodata'][0]
    ecoshard.geoprocessing.raster_calculator(
        [(args.lulc_path, 1),
         (args.mask_path, 1), (args.forest_code, 'raw'),
         (nodata, 'raw')],
        _mask_to_forest, args.target_marginal_value_path, gdal.GDT_Int32,
        nodata)
