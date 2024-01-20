"""One-time script used to align and project rasters for carbon model."""
import argparse
import glob
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

WORLD_ECKERT_IV_WKT = """PROJCRS["unknown",
    BASEGEOGCRS["GCS_unknown",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1]],
            ID["EPSG",6326]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["unnamed",
        METHOD["Eckert IV"],
        PARAMETER["Longitude of natural origin",0,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8802]],
        PARAMETER["False easting",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8806]],
        PARAMETER["False northing",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8807]]],
    CS[Cartesian,2],
        AXIS["(E)",east,
            ORDER[1],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]],
        AXIS["(N)",north,
            ORDER[2],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]]]"""

WORKSPACE_DIR = 'processed_rasters'
os.makedirs(WORKSPACE_DIR, exist_ok=True)

GLOBAL_ECKERT_IV_BB = [-16921202.923, -8460601.461, 16921797.077, 8461398.539]

LOGGER = logging.getLogger(__name__)


def _limited_bounding_box_union(raster_path_list):
    """Return the Eckert IV projected & bounded bounding box union."""
    projected_bb_list = []
    for raster_path in raster_path_list:
        raster_info = geoprocessing.get_raster_info(raster_path)
        transformed_bb = geoprocessing.transform_bounding_box(
            raster_info['bounding_box'], raster_info['projection_wkt'],
            WORLD_ECKERT_IV_WKT, check_finite=False)
        local_bb = []
        for trans_coord, global_coord in zip(
                transformed_bb, GLOBAL_ECKERT_IV_BB):
            if numpy.isfinite(trans_coord):
                local_bb.append(trans_coord)
            else:
                local_bb.append(global_coord)
        projected_bb_list.append(local_bb)
    union_bb = geoprocessing.merge_bounding_box_list(
        projected_bb_list, 'union')
    return union_bb


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Align rasters.')
    parser.add_argument(
        'raster_path_pattern', nargs='+', help='path or pattern to rasters '
        'to align')
    parser.add_argument(
        '--pixel_size', default=300, type=float,
        help='pixel size, default is 300m')
    args = parser.parse_args()

    task_graph = taskgraph.TaskGraph('.', multiprocessing.cpu_count(), 5.0)

    raster_path_list = [
        raster_path
        for raster_pattern in args.raster_path_pattern
        for raster_path in glob.glob(raster_pattern)]

    for raster_path in raster_path_list:
        LOGGER.info(f'process {raster_path}')
        target_raster_path = os.path.join(
            WORKSPACE_DIR, os.path.basename(raster_path))
        task_graph.add_task(
            func=geoprocessing.warp_raster,
            args=(
                raster_path, (args.pixel_size, -args.pixel_size),
                target_raster_path, 'near'),
            kwargs={
                'target_bb': GLOBAL_ECKERT_IV_BB,
                'target_projection_wkt': WORLD_ECKERT_IV_WKT,
                },
            target_path_list=[target_raster_path],
            task_name=f'warp {os.path.basename(raster_path)}')

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()
