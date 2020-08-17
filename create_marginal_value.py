"""Create Marginal value raster."""
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


def diff(a, b, nodata):
    result = numpy.empty(a.shape, dtype=numpy.float32)
    result[:] = nodata
    valid_mask = ~(numpy.isclose(a, nodata) | numpy.isclose(b, nodata))
    result[valid_mask] = a[valid_mask] - b[valid_mask]
    return result


if __name__ == '__main__':
    nodata = pygeoprocessing.get_raster_info(
        scenario_raster_path)['nodata'][0]
    pygeoprocessing.multiprocessing.raster_calculator(
        [(scenario_raster_path, 1), (base_raster_path, 1), (nodata, 'raw')],
        diff, target_raster_path, gdal.GDT_Float32, nodata)
