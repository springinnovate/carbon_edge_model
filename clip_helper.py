"""Clip raster to AOI."""
import argparse
import logging
import sys

import pygeoprocessing


logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)

LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clip raster')
    parser.add_argument(
        'base_raster_path', help='Path to base raster to clip.')
    parser.add_argument('aoi_vector_path', help='Path to vector to clip.')
    parser.add_argument('target_clipped_path', help='Path to cliped raster.')
    args = parser.parse_args()

    raster_info = pygeoprocessing.get_raster_info(args.base_raster_path)
    vector_info = pygeoprocessing.get_vector_info(args.aoi_vector_path)

    pygeoprocessing.warp_raster(
        args.base_raster_path,
        raster_info['pixel_size'],
        args.target_clipped_path, 'nearest',
        target_bb=vector_info['bounding_box'])
