"""Feature selction preprocessor."""
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
import numpy

import carbon_model_data
if __name__ == '__main__':
    X_vector_path = 'model_base_data/array_cache/X_array_1.npz'
    with open(X_vector_path, 'rb') as X_vector_file:
        X_vector = numpy.load(X_vector_file)['arr_0']
    y_vector_path = 'model_base_data/array_cache/y_array_1.npz'
    with open(y_vector_path, 'rb') as y_vector_file:
        y_vector = numpy.load(y_vector_file)['arr_0']

    parameter_name_list = [
        val[0] for val in carbon_model_data.CARBON_EDGE_MODEL_DATA_NODATA] + [
        f'{raster_type[0]}_gf_{dist}'
        for raster_type in carbon_model_data.MASK_TYPES
        for dist in carbon_model_data.MAX_EFFECT_EDGEDIST]

    f_reg = f_regression(X_vector, y_vector)
    mut_info = mutual_info_regression(X_vector, y_vector)
    print(
        '\n'.join([str((name, p_val, m_val)) for name, p_val, m_val in zip(
            parameter_name_list, f_reg.pval, mut_info)]))
    print(mut_info)
