"""Tracer code for regression models."""
import argparse
import logging
import pickle
import sys


import carbon_model_data

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dump coeffients')
    parser.add_argument('model_path', help='path to pickled model')
    args = parser.parse_args()

    with open(args.model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    base_feature_names = carbon_model_data

    parameter_name_list = [
        val[0] for val in carbon_model_data.CARBON_EDGE_MODEL_DATA_NODATA] + [
        f'{raster_type[0]}_gf_{dist}'
        for raster_type in carbon_model_data.MASK_TYPES
        for dist in carbon_model_data.MAX_EFFECT_EDGEDIST]

    coeff_parameter_list = zip(
        model[-1].coef_,
        model[0].get_feature_names(parameter_name_list))

    print('\n'.join([
        f"{value:+.3e},{parameter_name.replace(' ', '*').replace('.tif', '')}"
        for value, parameter_name in sorted(
            coeff_parameter_list, reverse=True, key=lambda x: abs(x[0]))
        if abs(value) > 1e-4]))
