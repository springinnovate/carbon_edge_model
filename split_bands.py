"""Split multiband raster into individual raster bands."""
import argparse
import os
import logging
import multiprocessing

from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal


gdal.SetCacheMax(2**30)
logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Split multiband raster into individual raster')
    parser.add_argument('base_path', type=str, help='path to multiband raster')
    parser.add_argument(
        '--offset_count', type=int,
        help='number to add to target raster band id suffix')
    args = parser.parse_args()

    raster_info = geoprocessing.get_raster_info(args.base_path)
    n_bands = raster_info['n_bands']

    target_path_list = [
        f'%s{args.offset_count+band_index}%s' % os.path.splitext(
            os.path.basename(args.base_path))
        for band_index in range(n_bands)]

    if any([os.path.exists(path) for path in target_path_list]):
        raise ValueError(
            f"expected paths arlready exist, don't want to overwrite: "
            f"{target_path_list}")

    task_graph = taskgraph.TaskGraph(
        '.', min(multiprocessing.cpu_count(), n_bands))
    task_graph.join()

    for band_index, target_path in enumerate(target_path_list):
        task_graph.add_task(
            func=geoprocessing.raster_calculator,
            args=[
                (args.base_path, band_index+1), passthrough_op, target_path,
                raster_info['datatype'], raster_info['nodata'][band_index]],
            target_path_list=[target_path],
            task_name=f'extract band {band_index}')

    task_graph.close()
    task_graph.join()


def passthrough_op(x): return x


if __name__ == '__main__':
    main()
