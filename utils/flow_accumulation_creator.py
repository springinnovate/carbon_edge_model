"""Create a global flow accumulation layer."""
import logging
import os
import sys

from osgeo import gdal
import ecoshard
import taskgraph

gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)

LOGGER = logging.getLogger(__name__)

DEM_URI = (
    'gs://ipbes-ndr-ecoshard-data/'
    'global_dem_3s_blake2b_0532bf0a1bedbe5a98d1dc449a33ef0c.zip')

if __name__ == '__main__':
    WORKSPACE_DIR = 'flow_accumulation_workspace'
    try:
        os.makedirs(WORKSPACE_DIR)
    except OSError:
        pass
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR)

    dem_dir = os.path.join(WORKSPACE_DIR, 'dem')

    _ = task_graph.add_task(
        func=ecoshard.download_and_unzip,
        args=(DEM_URI, dem_dir),
        task_name=f'download model {DEM_URI} to {dem_dir}')

    task_graph.close()
    task_graph.join()
