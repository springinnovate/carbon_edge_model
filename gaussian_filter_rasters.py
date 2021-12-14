"""Pass rasters and create gaussian filters for them."""
import argparse
import glob
import os
import logging
import multiprocessing

from ecoshard import geoprocessing
import numpy
import scipy
import taskgraph

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

PIXEL_SIZE = (300, -300)
ALIGNED_WORKSPACE = 'new_aligned_rasters'
os.makedirs(ALIGNED_WORKSPACE, exist_ok=True)
GF_DIR = os.path.join(ALIGNED_WORKSPACE, 'gf_dir')
os.makedirs(GF_DIR, exist_ok=True)
CHURN_DIR = os.path.join(ALIGNED_WORKSPACE, 'churn_dir')
os.makedirs(CHURN_DIR, exist_ok=True)

GLOBAL_ECKERT_IV_BB = [-16921202.923, -8460601.461, 16921797.077, 8461398.539]


def _make_kernel_raster(pixel_radius, target_path):
    """Create kernel with given radius to `target_path`."""
    truncate = 2
    size = int(pixel_radius * 2 * truncate + 1)
    step_fn = numpy.zeros((size, size))
    step_fn[size//2, size//2] = 1
    kernel_array = scipy.ndimage.filters.gaussian_filter(
        step_fn, pixel_radius, order=0, mode='constant', cval=0.0,
        truncate=truncate)
    geoprocessing.numpy_array_to_raster(
        kernel_array, -1, (1., -1.), (0.,  0.), None,
        target_path)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='create spatial samples of data on a global scale')
    parser.add_argument(
        'raster_path_list', type=str, nargs='+',
        help='path/pattern to list of rasters to sample')
    parser.add_argument(
        '--kernel_distance_list', type=float, nargs='+',
        help='distance in km for sample kernel', required=True)
    args = parser.parse_args()

    task_graph = taskgraph.TaskGraph('.', multiprocessing.cpu_count(), 5.0)
    LOGGER.info('build kernels')
    kernel_raster_path_list = []
    for expected_max_edge_effect_km in args.kernel_distance_list:
        pixel_radius = 1000*expected_max_edge_effect_km/PIXEL_SIZE[0]
        kernel_raster_path = os.path.join(
            CHURN_DIR, f'kernel_{pixel_radius}.tif')
        kernel_task = task_graph.add_task(
            func=_make_kernel_raster,
            args=(pixel_radius, kernel_raster_path),
            target_path_list=[kernel_raster_path],
            task_name=f'make kernel of radius {pixel_radius}')
        kernel_raster_path_list.append(
            (kernel_raster_path, kernel_task, expected_max_edge_effect_km))

    task_graph.join()
    LOGGER.info('align everything')
    for path_pattern in args.raster_path_list:
        for raster_path in glob.glob(path_pattern):
            LOGGER.debug(f'process {raster_path}')
            basename = os.path.basename(os.path.splitext(raster_path)[0])
            warped_raster_path = os.path.join(
                ALIGNED_WORKSPACE, f'{basename}.tif')
            warp_task = task_graph.add_task(
                func=geoprocessing.warp_raster,
                args=(
                    raster_path, PIXEL_SIZE, warped_raster_path,
                    'near'),
                kwargs={
                    'target_bb': GLOBAL_ECKERT_IV_BB,
                    'target_projection_wkt': WORLD_ECKERT_IV_WKT,
                    },
                target_path_list=[warped_raster_path],
                task_name=f'warp {os.path.basename(raster_path)}')

            for (kernel_raster_path, kernel_task,
                 expected_max_edge_effect_km) in kernel_raster_path_list:
                gf_path = os.path.join(
                    GF_DIR, f'{basename}_gf_{expected_max_edge_effect_km}.tif')
                LOGGER.debug(f'making convoluion for {gf_path}')
                _ = task_graph.add_task(
                    func=geoprocessing.convolve_2d,
                    args=(
                        (warped_raster_path, 1), (kernel_raster_path, 1),
                        gf_path),
                    dependent_task_list=[warp_task, kernel_task],
                    target_path_list=[gf_path],
                    task_name=f'create guassian filter at {gf_path}')

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()
