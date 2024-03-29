"""Extract masks of landcover types from primary landcover."""
import argparse
import os
import logging
import multiprocessing

from ecoshard import geoprocessing
from ecoshard import taskgraph
import numpy

from osgeo import gdal
gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

logging.getLogger('matplotlib').setLevel(logging.WARN)
logging.getLogger('fiona').setLevel(logging.WARN)

CROPLAND_LULC_CODES = tuple(range(10, 41))
URBAN_LULC_CODES = (190,)
FOREST_CODES = (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170)

MASK_TYPES = [
    ('forest', FOREST_CODES),
    ]

MASK_NODATA = 127

LOGGER = logging.getLogger(__name__)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='extract lulc masks')
    parser.add_argument(
        'landcover_raster_path', type=str,
        help='path to lulc raster to mask')
    args = parser.parse_args()
    task_graph = taskgraph.TaskGraph('.', multiprocessing.cpu_count(), 5.0)
    LOGGER.info('process by year')
    basename = os.path.splitext(
        os.path.basename(args.landcover_raster_path))[0]
    mask_dir = os.path.dirname(args.landcover_raster_path)
    for mask_id, lucodes in MASK_TYPES:
        LOGGER.info(f'mask {mask_id} on {args.landcover_raster_path}')
        mask_raster_path = os.path.join(
            mask_dir, f'{basename}_{mask_id}.tif')
        _ = task_graph.add_task(
            func=_mask_raster,
            args=(args.landcover_raster_path, lucodes, mask_raster_path),
            target_path_list=[mask_raster_path],
            task_name=f'mask out {mask_id}')
    task_graph.join()
    task_graph.close()


def _mask_raster(base_raster_path, integer_codes, target_raster_path):
    """Mask any integer codes in base to 1."""
    def _reclassify_op(array):
        """Set values 1d array/array to nodata unless `inverse` then opposite."""
        result = numpy.in1d(array, integer_codes).reshape(array.shape)
        return result

    geoprocessing.raster_calculator(
        [(base_raster_path, 1)], _reclassify_op, target_raster_path,
        gdal.GDT_Byte, None)


if __name__ == '__main__':
    main()
