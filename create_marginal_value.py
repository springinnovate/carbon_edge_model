"""Create Marginal value raster."""
import argparse
import logging
import sys

import pygeoprocessing.multiprocessing
import numpy

from osgeo import gdal


gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)

LOGGER = logging.getLogger(__name__)

scenario_raster_path = r'biomass_per_ha_stocks_restoration_mask.tif'
base_raster_path = r'biomass_per_ha_stocks_ESA2014_mask.tif'
target_raster_path = r'marginal_value_restoration_biomass.tif'


def _diff_op(a, b, nodata):
    result = numpy.empty(a.shape, dtype=numpy.float32)
    result[:] = nodata
    valid_mask = ~(numpy.isclose(a, nodata) | numpy.isclose(b, nodata))
    result[valid_mask] = a[valid_mask] - b[valid_mask]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build marginal value map')
    parser.add_argument(
        '--base_value_raster_path', help='Path to base value raster')
    parser.add_argument(
        '--scenario_value_raster_path', help=(
            'Path scenario value raster, must be same size/projection as base '
            'value raster'))
    parser.add_argument(
        '--target_marginal_value_path',
        help='path to output marginal value raster')

    args = parser.parse_args()
    nodata = pygeoprocessing.get_raster_info(
        args.base_value_raster_path)['nodata'][0]
    pygeoprocessing.multiprocessing.raster_calculator(
        [(args.scenario_value_raster_path, 1),
         (args.base_value_raster_path, 1), (nodata, 'raw')],
        _diff_op, args.target_marginal_value_path, gdal.GDT_Float32, nodata)
