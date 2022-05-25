"""Pass rasters and create gaussian filters for them."""
import argparse
import glob
import logging
import multiprocessing
import os

from ecoshard import geoprocessing
from ecoshard import taskgraph
import numpy
import scipy
import scipy.integrate as integrate
import scipy.ndimage

from osgeo import gdal
gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('ecoshard.taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

CHURN_DIR = 'churn_dir'
os.makedirs(CHURN_DIR, exist_ok=True)

GLOBAL_ECKERT_IV_BB = [-16921202.923, -8460601.461, 16921797.077, 8461398.539]


def _make_kernel_raster(pixel_radius, target_path):
    """Create kernel with given radius to `target_path`."""
    truncate = 4
    size = int(pixel_radius * 2 * truncate + 1)
    step_fn = numpy.zeros((size, size))
    step_fn[size//2, size//2] = 1
    kernel_array = scipy.ndimage.filters.gaussian_filter(
        step_fn, pixel_radius, order=0, mode='constant', cval=0.0,
        truncate=truncate)

    sigma2 = pixel_radius * pixel_radius
    scale = integrate.quad(
        lambda x: numpy.exp(-0.5/sigma2*x**2), -2, 0)[0]

    geoprocessing.numpy_array_to_raster(
        kernel_array/scale, -1, (1., -1.), (0.,  0.), None, target_path)


def _process_gf(raw_raster_path, mask_raster_path, scale, target_raster_path):
    """Scale raster by a value."""
    pixel_size = geoprocessing.get_raster_info(raw_raster_path)['pixel_size']

    def _mask_op(base, mask):
        result = base.copy()
        result /= (scale * pixel_size[0])
        result[(mask == 0) | (result < 0)] = 0
        return result

    geoprocessing.raster_calculator(
        ((raw_raster_path, 1), (mask_raster_path, 1)), _mask_op,
        target_raster_path, gdal.GDT_Float32, -1)


def filter_raster(
        base_raster_path_band, expected_max_edge_effect_km, target_path):
    """Gaussian filter base by expected max edge to target_path."""
    base_raster_path, base_raster_band = base_raster_path_band
    pixel_size = geoprocessing.get_raster_info(base_raster_path)['pixel_size']
    pixel_radius = 1000*expected_max_edge_effect_km/pixel_size[0]
    basename = os.path.basename(target_path)
    kernel_raster_path = os.path.join(
        CHURN_DIR, f'kernel_{pixel_radius}_{basename}')
    _make_kernel_raster(pixel_radius, kernel_raster_path)
    raw_gf_path = os.path.join(
        CHURN_DIR, f'raw_gf_{expected_max_edge_effect_km}_{basename}')
    LOGGER.debug(f'making convolution for {raw_gf_path}')

    geoprocessing.convolve_2d(
        (base_raster_path, base_raster_band), (kernel_raster_path, 1),
        raw_gf_path, normalize_kernel=True, largest_block=2**24)

    def _mask_op(raw_gf, base_array):
        result = numpy.where(base_array == 1, raw_gf, -1.0)
        return result

    geoprocessing.raster_calculator(
        [(raw_gf_path, 1), (base_raster_path, base_raster_band)], _mask_op,
        target_path, gdal.GDT_Float32, -1.0)

    os.remove(kernel_raster_path)
    os.remove(raw_gf_path)


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

    raster_path_list = [
        raster_path for raster_pattern in args.raster_path_list
        for raster_path in glob.glob(raster_pattern)]
    n_bands = sum([
        geoprocessing.get_raster_info(raster_path)['n_bands']
        for raster_path in raster_path_list])
    n_workers = min(
        n_bands*len(args.kernel_distance_list),
        multiprocessing.cpu_count())

    task_graph = taskgraph.TaskGraph('.', n_workers, 15)

    for raster_path in raster_path_list:
        LOGGER.debug(f'process {raster_path}')
        basename = os.path.basename(raster_path)
        base_dir = os.path.dirname(raster_path)
        raster_info = geoprocessing.get_raster_info(raster_path)
        n_bands = raster_info['n_bands']
        for expected_max_edge_effect_km in args.kernel_distance_list:
            for band_id in range(1, n_bands+1):
                if n_bands > 1:
                    band_str = f'_{band_id}'
                else:
                    band_str = ''
                gf_path = os.path.join(
                    base_dir,
                    f'gf_{expected_max_edge_effect_km}{band_str}_{basename}')
                LOGGER.debug(gf_path)
                task_graph.add_task(
                    func=filter_raster,
                    args=(
                        (raster_path, band_id), expected_max_edge_effect_km,
                        gf_path),
                    target_path_list=[gf_path],
                    task_name=f'filter raster {gf_path}')

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()
