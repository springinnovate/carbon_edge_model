"""Create a global flow accumulation layer."""
import glob
import logging
import os
import sys

from osgeo import gdal
import pygeoprocessing.routing
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

DEM_URL = (
    'https://storage.googleapis.com/ipbes-ndr-ecoshard-data/'
    'global_dem_3s_blake2b_0532bf0a1bedbe5a98d1dc449a33ef0c.zip')

if __name__ == '__main__':
    WORKSPACE_DIR = 'flow_accumulation_workspace'
    try:
        os.makedirs(WORKSPACE_DIR)
    except OSError:
        pass
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1)

    dem_dir = os.path.join(WORKSPACE_DIR, 'dem')
    try:
        os.makedirs(dem_dir)
    except OSError:
        pass
    download_task = task_graph.add_task(
        func=ecoshard.download_and_unzip,
        args=(DEM_URL, dem_dir),
        task_name=f'download model {DEM_URL} to {dem_dir}')

    vrt_raster_path = os.path.join(dem_dir, 'dem.vrt')
    vrt_build_task = task_graph.add_task(
        func=gdal.BuildVRT,
        args=(vrt_raster_path, dem_dir),
        dependent_task_list=[download_task],
        target_path_list=[vrt_raster_path],
        task_name='build vrt')

    pitfill_dem_raster_path = './pitfilled_dem.tif'
    pitfill_task = task_graph.add_task(
        func=pygeoprocessing.routing.fill_pits,
        args=(dem_dir, list(glob.glob(os.path.join(dem_dir, 'global_dem_3s', '*.tif')))),
        dependent_task_list=[vrt_build_task],
        target_path_list=[pitfill_dem_raster_path],
        task_name='fill dem pits')

    flow_dir_mfd_raster_path = './mfd.tif'
    flow_dir_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_dir_mfd,
        args=(pitfill_dem_raster_path, flow_dir_mfd_raster_path),
        dependent_task_list=[pitfill_task],
        target_path_list=[flow_dir_mfd_raster_path],
        task_name='flow dir mfd')

    flow_accum_raster_path = './flow_accum.tif'
    flow_accum_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_accumulation_mfd,
        args=((flow_dir_mfd_raster_path, 1), flow_accum_raster_path),
        dependent_task_list=[flow_dir_task],
        target_path_list=[flow_accum_raster_path],
        task_name='flow accumulation')

    task_graph.close()
    task_graph.join()
