"""Convert a raster in X/Ha to X*const/pixel."""
import argparse
import math

import logging
import sys

from osgeo import gdal
from osgeo import osr
import pygeoprocessing
import numpy


gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)

LOGGER = logging.getLogger(__name__)


def area_of_pixel(pixel_size, center_lat):
    """Calculate m^2 area of a wgs84 square pixel.

    Adapted from: https://gis.stackexchange.com/a/127327/2397

    Args:
        pixel_size (float): length of side of pixel in degrees.
        center_lat (float): latitude of the center of the pixel. Note this
            value +/- half the `pixel-size` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.

    Returns:
        Area of square pixel of side length `pixel_size` centered at
        `center_lat` in m^2.

    """
    a = 6378137  # meters
    b = 6356752.3142  # meters
    e = math.sqrt(1 - (b/a)**2)
    area_list = []
    for f in [center_lat+pixel_size/2, center_lat-pixel_size/2]:
        zm = 1 - e*math.sin(math.radians(f))
        zp = 1 + e*math.sin(math.radians(f))
        area_list.append(
            math.pi * b**2 * (
                math.log(zp/zm) / (2*e) +
                math.sin(math.radians(f)) / (zp*zm)))
    return abs(pixel_size / 360. * (area_list[0] - area_list[1]))


def mult_op(array, conversion, nodata):
    """Mult array by conversion, skipping nodata."""
    result = numpy.copy(array)
    if nodata is not None:
        valid_mask = ~numpy.isclose(array, nodata)
        result[valid_mask] *= conversion[valid_mask]
    else:
        result *= conversion
    return result


def density_per_ha_to_total_per_pixel(
        base_value_per_ha_raster_path, mult_factor,
        target_total_per_pixel_raster_path):
    """Convert X/Ha in base to (total X)/pixel."""
    base_raster_info = pygeoprocessing.get_raster_info(
        base_value_per_ha_raster_path)

    base_srs = osr.SpatialReference()
    base_srs.ImportFromWkt(base_raster_info['projection_wkt'])
    if base_srs.IsProjected():
        # convert m^2 of pixel size to Ha
        pixel_conversion = numpy.array([[
            abs(base_raster_info['pixel_size'][0] *
                base_raster_info['pixel_size'][1])]]) / 10000.0
    else:
        # create 1D array of pixel size vs. lat
        n_rows = base_raster_info['raster_size'][1]
        pixel_height = abs(base_raster_info['geotransform'][5])
        # the / 2 is to get in the center of the pixel
        miny = base_raster_info['bounding_box'][1] + pixel_height/2
        maxy = base_raster_info['bounding_box'][3] - pixel_height/2
        lat_vals = numpy.linspace(maxy, miny, n_rows)

        pixel_conversion = 1.0 / 10000.0 * numpy.array([
            [area_of_pixel(pixel_height, lat_val)] for lat_val in lat_vals])

    pixel_conversion *= mult_factor

    nodata = base_raster_info['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(args.base_raster_path, 1), pixel_conversion, (nodata, 'raw')],
        mult_op, target_total_per_pixel_raster_path,
        gdal.GDT_Float32, nodata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert a raster in units X/Ha to (total X)/pixel.')
    parser.add_argument(
        'base_value_per_ha_raster_path',
        help='Path to base raster whose units are X/ha')
    parser.add_argument(
        'target_total_per_pixel_raster_path',
        help='Path to desired target raster with units (total X)/pixel.')
    parser.add_argument(
        '--mult_factor', default=1.0, type=float,
        help='Additional factor to multiply total by, default is 1.0.')
    args = parser.parse_args()
    density_per_ha_to_total_per_pixel(
        args.base_value_per_ha_raster_path, args.mult_factor,
        args.target_total_per_pixel_raster_path)

