"""Convert a raster in X/Ha to X*const/pixel."""
import argparse
import math
import logging

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

    Parameters:
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
    return pixel_size / 360. * (area_list[0] - area_list[1])


def conversion_op(array, conversion, nodata):
    result = numpy.copy(array)
    nodata_mask = numpy.isclose(array, nodata)
    result[nodata_mask] *= conversion[nodata_mask]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert to carbon stocks')
    parser.add_argument(
        'base_raster_path',
        help='Path to base raster whose units are biomass/ha')
    parser.add_argument(
        'target_area_raster_path', help='Path to desired target raster.')
    parser.add_argument(
        '--factor', default=1.0, type=float,
        help='Additional factor to multiply values by.')
    args = parser.parse_args()

    base_raster_info = pygeoprocessing.get_raster_info(args.base_raster_path)

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
        maxy = base_raster_info['bounding_box'][1] - pixel_height/2
        lat_vals = numpy.linspace(maxy, miny, n_rows)

        pixel_conversion = numpy.array([
            area_of_pixel(pixel_height, lat_val) for lat_val in lat_vals])

    pixel_conversion *= args.factor

    nodata = base_raster_info['nodata'][0]
    pygeoprocessing.raster_calculator(
        [(args.base_raster_path, 1), pixel_conversion, (nodata, 'raw')],
        conversion_op, args.target_area_raster_path,
        base_raster_info['datatype'], nodata)
