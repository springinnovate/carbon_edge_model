"""Build a visualization of sample points for debugging."""
import glob
import os

from osgeo import ogr
from osgeo import osr
import numpy

import carbon_model_data

EXPECTED_MAX_EDGE_EFFECT_KM = 3.0


def generate_sample_point_csv(
        x_vector_list, y_vector_list, field_names, target_csv_path):
    """Dump samples to a CSV file."""
    with open(target_csv_path, 'w') as csv_file:
        csv_file.write(','.join(field_names) + '\n')
        for x_vector in x_vector_list:
            csv_file.write(','.join([str(v) for v in x_vector[0]]) + '\n')


def generate_sample_csv(
        x_vector_list, y_vector_list, lng_lat_vector_list, field_names,
        target_vector_path):
    """Create sample point vector."""


def generate_sample_point_vector(
        x_vector_list, y_vector_list, lng_lat_vector_list, field_names,
        target_vector_path):
    """Create sample point vector."""
    gpkg_driver = ogr.GetDriverByName("GPKG")
    vector = gpkg_driver.CreateDataSource(
        "carbon_model_sample_points.gpkg")
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = vector.CreateLayer("carbon_model_sample_points", srs, ogr.wkbPoint)
    for field_name in field_names:
        vector_field = ogr.FieldDefn(field_name, ogr.OFTReal)
        layer.CreateField(vector_field)

    y_fieldname = 'baccini_carbon'
    layer.CreateField(ogr.FieldDefn(y_fieldname, ogr.OFTReal))

    layer.StartTransaction()
    for index, (x_vector, y_vector, lng_lat_vector) in enumerate(zip(
                x_vector_list, y_vector_list, lng_lat_vector_list)):
        print(f'converting vector {index+1} of {len(lng_lat_vector_list)}')
        for x_values, y_val, (lng, lat) in zip(
                x_vector, y_vector, lng_lat_vector):
            feature = ogr.Feature(layer.GetLayerDefn())
            for x_val, field_name in zip(x_values, field_names):
                feature.SetField(field_name, float(x_val))
            feature.SetField(y_fieldname, float(y_val))

            point = ogr.CreateGeometryFromWkt(f"POINT({lng} {lat})")
            feature.SetGeometry(point)
            layer.CreateFeature(feature)
            feature = None
    layer.CommitTransaction()
    layer = None
    vector = None


if __name__ == '__main__':

    lng_lat_vector_path_list = glob.glob(
        os.path.join('model_base_data', 'array_cache', 'lng_lat_array*'))
    X_vector_path_list = glob.glob(
        os.path.join('model_base_data', 'array_cache', 'X_array*'))
    y_vector_path_list = glob.glob(
        os.path.join('model_base_data', 'array_cache', 'y_array*'))

    x_vector_list = []
    y_vector_list = []
    lng_lat_vector_list = []

    for index, (ll_path, xv_path, yv_path) in enumerate(zip(
            lng_lat_vector_path_list,
            X_vector_path_list,
            y_vector_path_list)):
        if index > 3:
            break
        lng_lat_vector_list.append(numpy.load(ll_path)['arr_0'])
        x_vector_list.append(numpy.load(xv_path)['arr_0'])
        y_vector_list.append(numpy.load(yv_path)['arr_0'])

    convolution_field_names = [
        f'{mask_id}_gf_{EXPECTED_MAX_EDGE_EFFECT_KM}'
        for mask_id, _ in carbon_model_data.MASK_TYPES]
    feature_name_list = [
        val[0] for val in carbon_model_data.CARBON_EDGE_MODEL_DATA_NODATA]

    target_vector_path = 'test_points.gpkg'
    generate_sample_point_vector(
        x_vector_list, y_vector_list, lng_lat_vector_list,
        feature_name_list + convolution_field_names,
        target_vector_path)

    target_csv_path = 'test_points.csv'
    generate_sample_point_csv(
        x_vector_list, y_vector_list,
        feature_name_list + convolution_field_names, target_csv_path)
