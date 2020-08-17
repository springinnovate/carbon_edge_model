"""One-time script used to generate base global data for the model."""
import os
import logging
import subprocess

import retrying

BASE_URL = 'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs'
BASE_URI = 'gs://ecoshard-root/global_carbon_regression/inputs'

BACCINI_10s_2014_BIOMASS_FILENAME = \
    'baccini_10s_2014_md5_5956a9d06d4dffc89517cefb0f6bb008.tif'

CARBON_EDGE_MODEL_DATA_NODATA = [
    ('accessibility_to_cities_2015_30sec.tif', -9999),
    ('ACDWRB_10sec.tif', 255),
    ('altitude_10sec.tif', None),
    ('AWCh1_10sec.tif', 255),
    ('AWCh2_10sec.tif', 255),
    ('AWCh3_10sec.tif', 255),
    ('AWCtS_10sec.tif', 255),
    ('bdod_10sec.tif', 0),
    ('BDRICM_10sec.tif', 255),
    ('BDRLOG_10sec.tif', 255),
    ('BDTICM_10sec.tif', 9999),
    ('bio_01_30sec.tif', -9999),
    ('bio_02_30sec.tif', -9999),
    ('bio_03_30sec.tif', -9999),
    ('bio_04_30sec.tif', -9999),
    ('bio_05_30sec.tif', -9999),
    ('bio_06_30sec.tif', -9999),
    ('bio_07_30sec.tif', -9999),
    ('bio_08_30sec.tif', -9999),
    ('bio_09_30sec.tif', -9999),
    ('bio_10_30sec.tif', -9999),
    ('bio_11_30sec.tif', -9999),
    ('bio_12_30sec.tif', -9999),
    ('bio_13_30sec.tif', -9999),
    ('bio_14_30sec.tif', -9999),
    ('bio_15_30sec.tif', -9999),
    ('bio_16_30sec.tif', -9999),
    ('bio_17_30sec.tif', -9999),
    ('bio_18_30sec.tif', -9999),
    ('bio_19_30sec.tif', -9999),
    ('BLDFIE_10sec.tif', 9999),
    ('cfvo_10sec.tif', -9999),
    ('clay_10sec.tif', -9999),
    ('CLYPPT_10sec.tif', 255),
    ('CRFVOL_10sec.tif', 255),
    ('hillshade_10sec.tif', 181),
    ('HISTPR_10sec.tif', 255),
    ('livestock_Bf_2010_5min.tif', -9999),
    ('livestock_Ch_2010_5min.tif', -1.7e308),
    ('livestock_Ct_2010_5min.tif', -1.7e308),
    ('livestock_Dk_2010_5min.tif', -1.7e308),
    ('livestock_Gt_2010_5min.tif', -1.7e308),
    ('livestock_Ho_2010_5min.tif', -1.7e308),
    ('livestock_Pg_2010_5min.tif', -1.7e308),
    ('livestock_Sh_2010_5min.tif', -1.7e308),
    ('ndvcec015_10sec.tif', 0),
    ('night_lights_10sec.tif', None),
    ('night_lights_5min.tif', None),
    ('nitrogen_10sec.tif', -9999),
    ('ocd_10sec.tif', -9999),
    ('OCDENS_10sec.tif', 9999),
    ('ocs_10sec.tif', -9999),
    ('OCSTHA_10sec.tif', 9999),
    ('phh2o_10sec.tif', -9999),
    ('PHIHOX_10sec.tif', 255),
    ('PHIKCL_10sec.tif', 255),
    ('population_2015_30sec.tif', 3.4028235e+38),
    ('population_2015_5min.tif', 3.4028235e+38),
    ('sand_10sec.tif', -9999),
    ('silt_10sec.tif', -9999),
    ('slope_10sec.tif', None),
    ('soc_10sec.tif', -9999),
    ('tri_10sec.tif', 0),
    ('wind_speed_10sec.tif', -999),
]

LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)


def download_gs(base_uri, target_path):
    """Download base to target."""
    subprocess.run(
        f'gsutil cp {base_uri} {target_path}', check=True, shell=True)


def fetch_data(data_dir, task_graph):
    """Download all the global data needed to run this analysis.

    Args:
        bounding_box (list): minx, miny, maxx, maxy list to clip to
        data_dir (str): path to directory to copy clipped rasters
            to
        task_graph (TaskGraph): taskgraph object to schedule work.

    Returns:
        None.

    """
    files_to_download = [
        os.path.join(BASE_URI, path)
        for path, _ in CARBON_EDGE_MODEL_DATA_NODATA] + [
            f'{os.path.join(BASE_URI, BACCINI_10s_2014_BIOMASS_FILENAME)}']

    LOGGER.debug(f'here are the files to download: {files_to_download}')

    try:
        os.makedirs(data_dir)
    except OSError:
        pass

    for file_uri in files_to_download:
        target_file_path = os.path.join(
            data_dir, os.path.basename(file_uri))
        _ = task_graph.add_task(
            func=download_gs,
            args=(file_uri, target_file_path),
            kwargs={'skip_if_target_exists': True},
            target_path_list=[target_file_path],
            task_name=f'download {file_uri} to {data_dir}')

    task_graph.join()
