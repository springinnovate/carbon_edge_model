"""Convert to wgs84."""
import glob
import logging
import os

from osgeo import gdal
from osgeo import osr
from ecoshard import taskgraph
from ecoshard import geoprocessing

gdal.SetCacheMax(2**26)

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('ecoshard').setLevel(logging.DEBUG)
logging.getLogger('ecoshard.taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)


TARGET_WGS84_PIXEL_SIZE = (10/3600, -10/3600)

def main():
    """Entry point."""
    task_graph = taskgraph.TaskGraph(
        './processed_rasters', 32, parallel_mode='thread')
    for file_path in glob.glob('processed_rasters/*.tif'):
        LOGGER.debug(file_path)
        target_path = (
            f'{os.path.dirname(file_path)}/'
            f'wgs84_{os.path.basename(file_path)}')
        task_graph.add_task(
            func=geoprocessing.warp_raster,
            args=(file_path, TARGET_WGS84_PIXEL_SIZE, target_path),
            kwargs={'target_projection_wkt': osr.SRS_WKT_WGS84_LAT_LONG})
    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()