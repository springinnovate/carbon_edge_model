"""Sample the data layers at the given points."""
import glob
import os

from osgeo import gdal
import pygeoprocessing
import numpy

MODEL_DATA_DIR = './model_base_data'
SAMPLE_VECTOR_POINTS = './sample_points.gpkg'
SAMPLE_CSV_FILE = './sample_table.csv'


def sample_rasters(lng, lat, sample_raster_path_list):
    """Sample all rasters in sample_raster_path_list and return list."""
    sample_list = []
    for path in sample_raster_path_list:
        raster_info = pygeoprocessing.get_raster_info(path)
        inv_gt = gdal.InvGeoTransform(raster_info['geotransform'])
        x, y = gdal.ApplyGeoTransform(inv_gt, lng, lat)
        raster = gdal.OpenEx(path, gdal.OF_RASTER)
        val = raster.ReadAsArray(int(x), int(y), 1, 1)[0, 0]
        if numpy.isclose(val, raster_info['nodata'][0]):
            val = 'nodata'
        sample_list.append(val)
    return sample_list


def main():
    """Entry point."""
    raster_path_list = glob.glob(os.path.join(MODEL_DATA_DIR, '*.tif'))
    raster_base_name_list = [
        os.path.basename(os.path.splitext(path)[0])
        for path in raster_path_list]

    with open(SAMPLE_CSV_FILE, 'w') as csv_file:
        csv_file.write('lng,lat,')
        csv_file.write(','.join(raster_base_name_list))
        csv_file.write('\n')

        point_vector = gdal.OpenEx(SAMPLE_VECTOR_POINTS, gdal.OF_VECTOR)
        point_layer = point_vector.GetLayer()
        for point_feature in point_layer:
            point_geom = point_feature.GetGeometryRef()
            sample_point_list = sample_rasters(
                point_geom.GetX(), point_geom.GetY(), raster_path_list)
            csv_file.write(f'{point_geom.GetX()}, {point_geom.GetY()},')
            csv_file.write(
                f'''{','.join([str(v) for v in sample_point_list])}''')
            csv_file.write('\n')


if __name__ == "__main__":
    main()
