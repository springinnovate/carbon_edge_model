# calc r^2
import argparse

from sklearn.metrics import r2_score
from osgeo import gdal
import pygeoprocessing
import numpy
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Run CE model')
    parser.add_argument('raster_a_path')
    parser.add_argument('raster_b_path')
    args = parser.parse_args()

    raster_a = gdal.OpenEx(args.raster_a_path, gdal.OF_RASTER)
    raster_b = gdal.OpenEx(args.raster_b_path, gdal.OF_RASTER)
    band_a = raster_a.GetRasterBand(1)
    band_b = raster_b.GetRasterBand(1)
    a_nodata = pygeoprocessing.get_raster_info(args.raster_a_path)['nodata'][0]
    b_nodata = pygeoprocessing.get_raster_info(args.raster_b_path)['nodata'][0]
    valid_a = numpy.array([])
    valid_b = numpy.array([])
    for offset_dict, array in pygeoprocessing.iterblocks(
            (args.raster_a_path, 1)):
        array_a = band_a.ReadAsArray(**offset_dict)
        array_b = band_b.ReadAsArray(**offset_dict)
        valid_mask = (array_a != a_nodata) & (array_b != b_nodata)
        valid_a = numpy.append(valid_a, array_a[valid_mask])
        valid_b = numpy.append(valid_b, array_b[valid_mask])

    r2 = r2_score(valid_a, valid_b)
    print(f'r2: {r2}')

    print(numpy.sum(valid_b/valid_a)/valid_a.size)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    n=100000
    ax.scatter(valid_b[:n], valid_a[:n], s=0.01)
    ax.set_xlim([200, 300])
    ax.set_ylim([200, 300])
    plt.show()


if __name__ == '__main__':
    main()
