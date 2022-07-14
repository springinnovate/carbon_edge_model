"""Execute carbon model on custom forest edge data."""
import argparse
import pickle
import logging
import os

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)
logging.getLogger('fiona').setLevel(logging.WARN)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARN)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARN)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='execute carbon model')
    parser.add_argument(
        'carbon_model_path', type=str, help='path to trained carbon model')
    parser.add_argument(
        'forest_cover_path', type=str, help='path to forest mask')
    parser.add_argument(
        '--predictor_raster_directory', type=str, default='',
        help='path to directory containing base data for model')
    parser.add_argument(
        '--prefix', type=str, help='string to add to output filename')
    args = parser.parse_args()

    LOGGER.info(f'load model at {args.carbon_model_path}')
    with open(args.carbon_model_path, 'rb') as model_file:
        model = pickle.load(model_file).copy()
    LOGGER.info(f'ensure raster base data are present')
    predictor_id_to_path = {}
    missing_predictor_list = []
    for predictor_id in model['predictor_list']:
        predictor_path = os.path.join(
            args.predictor_raster_directory, f'{predictor_id}.tif')
        if os.path.exists(predictor_path):
            predictor_id_to_path[predictor_id] = predictor_path
        else:
            missing_predictor_list.append(
                f'{predictor_id}: {predictor_path}')
    if missing_predictor_list:
        predictor_str = "\n".join(missing_predictor_list)
        raise ValueError(
            f'missing the following predictor rasters:\n{predictor_str}')


if __name__ == '__main__':
    main()
