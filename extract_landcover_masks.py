"""Used to extract forest, crop, and urban masks from landcover."""
import argparse
import glob
import os
import logging
import multiprocessing

import numpy
from ecoshard import taskgraph
from ecoshard import geoprocessing
from osgeo import gdal

CROPLAND_LULC_CODES = tuple(range(10, 41))
URBAN_LULC_CODES = (190,)
FOREST_CODES = (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170)

MASK_TYPES = [
    ('cropland', CROPLAND_LULC_CODES),
    ('urban', URBAN_LULC_CODES),
    ('forest', FOREST_CODES)]

gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('ecoshard.taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)


def _mask_raster(raster_path, mask_int_list, target_path):
    """Create 0/1/nodata mask for raster given codes."""
    nodata = geoprocessing.get_raster_info(raster_path)['nodata'][0]
    mask_nodata = 2

    def _mask_op(base_array):
        """Mask base to mask_int_list."""
        if nodata is not None:
            valid_mask = base_array == nodata
        else:
            valid_mask = numpy.ones(base_array.shape, dtype=bool)
        mask_array = numpy.full(base_array.shape, mask_nodata)
        mask_array[valid_mask] = numpy.isin(
            base_array[valid_mask], mask_int_list)
        return mask_array

    geoprocessing.raster_calculator(
        [(raster_path, 1)], _mask_op, target_path, gdal.GDT_Byte, mask_nodata)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Extract ESA style crop, urban, and forest maps')
    parser.add_argument(
        'raster_pattern_list', type=str, nargs='+',
        help='path/pattern to list of rasters to sample')
    args = parser.parse_args()

    raster_path_list = [
        raster_path
        for raster_pattern in args.raster_pattern_list
        for raster_path in glob.glob(raster_pattern)]
    LOGGER.debug(raster_path_list)

    n_workers = min(
        len(raster_path_list)*len(MASK_TYPES),
        multiprocessing.cpu_count())
    LOGGER.info(f'{n_workers} workers are starting')
    task_graph = taskgraph.TaskGraph('.', n_workers, 10.0)

    for raster_path in raster_path_list:
        for mask_id, lulc_codes in MASK_TYPES:
            LOGGER.info(f'process {raster_path}:{mask_id}')
            base_dir = os.path.dirname(raster_path)
            target_path = os.path.join(
                base_dir, f'masked_{mask_id}_{os.path.basename(raster_path)}')
            task_graph.add_task(
                func=_mask_raster,
                args=(raster_path, lulc_codes, target_path),
                target_path_list=[target_path],
                task_name=f'mask {target_path}')

    LOGGER.info(f'waiting for jobs to complete')
    task_graph.close()
    task_graph.join()
    del task_graph
    LOGGER.info(f'all done!')


if __name__ == '__main__':
    main()
