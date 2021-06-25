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
    print(a_nodata, b_nodata)
    valid_a = numpy.array([])
    valid_b = numpy.array([])
    for offset_dict, array in pygeoprocessing.iterblocks(
            (args.raster_a_path, 1)):
        array_a = band_a.ReadAsArray(**offset_dict)
        array_b = band_b.ReadAsArray(**offset_dict)
        valid_mask = (array_a > 0) & (array_a < 500) & (array_b > 0) & (array_b < 500)
        valid_a = numpy.append(valid_a, array_a[valid_mask])
        valid_b = numpy.append(valid_b, array_b[valid_mask])

    n = 10000
    arr = numpy.arange(valid_a.size)
    numpy.random.shuffle(arr)
    index = arr[:n]
    r2 = r2_score(valid_a[index], valid_b[index], multioutput='variance_weighted')
    print(f'r2: {r2}')

    print(numpy.sum(valid_b/valid_a)/valid_a.size)
    max_val = numpy.max(valid_a)
    print(f'max val: {numpy.max(valid_a)} {numpy.max(valid_b)}')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(valid_b[index], valid_a[index], s=1, alpha=1)
    ax.plot(numpy.arange(max_val), numpy.arange(max_val), linewidth=0.5, c='b')
    ax.set_xlim([50, max_val])
    ax.set_ylim([50, max_val])
    plt.show()

    #heatmap, xedges, yedges = numpy.histogram2d(valid_b, valid_a, bins=50)
    #extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    #plt.clf()
    #plt.imshow(heatmap.T, extent=extent, origin='lower')
    #plt.show()


if __name__ == '__main__':
    main()
