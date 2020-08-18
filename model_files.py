"""One-time script used to generate base global data for the model."""
import os
import logging
import subprocess

import retrying

BASE_URL = 'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs'
BASE_URI = 'gs://ecoshard-root/global_carbon_regression/inputs'

BACCINI_10s_2014_BIOMASS_URI = (
    'gs://ecoshard-root/global_carbon_regression/'
    'baccini_10s_2014_md5_5956a9d06d4dffc89517cefb0f6bb008.tif')

CARBON_EDGE_MODEL_DATA_NODATA = [
    ('accessibility_to_cities_2015_30sec.tif', -9999, None),
    ('ACDWRB_10sec.tif', 255, None),
    ('altitude_10sec.tif', None, None),
    ('AWCh1_10sec.tif', 255, None),
    ('AWCh2_10sec.tif', 255, None),
    ('AWCh3_10sec.tif', 255, None),
    ('AWCtS_10sec.tif', 255, None),
    ('bdod_10sec.tif', 0, None),
    ('BDRICM_10sec.tif', 255, None),
    ('BDRLOG_10sec.tif', 255, None),
    ('BDTICM_10sec.tif', 9999, None),
    ('bio_01_30sec.tif', -9999, None),
    ('bio_02_30sec.tif', -9999, None),
    ('bio_03_30sec.tif', -9999, None),
    ('bio_04_30sec.tif', -9999, None),
    ('bio_05_30sec.tif', -9999, None),
    ('bio_06_30sec.tif', -9999, None),
    ('bio_07_30sec.tif', -9999, None),
    ('bio_08_30sec.tif', -9999, None),
    ('bio_09_30sec.tif', -9999, None),
    ('bio_10_30sec.tif', -9999, None),
    ('bio_11_30sec.tif', -9999, None),
    ('bio_12_30sec.tif', -9999, None),
    ('bio_13_30sec.tif', -9999, None),
    ('bio_14_30sec.tif', -9999, None),
    ('bio_15_30sec.tif', -9999, None),
    ('bio_16_30sec.tif', -9999, None),
    ('bio_17_30sec.tif', -9999, None),
    ('bio_18_30sec.tif', -9999, None),
    ('bio_19_30sec.tif', -9999, None),
    ('BLDFIE_10sec.tif', 9999, None),
    ('cfvo_10sec.tif', -9999, None),
    ('clay_10sec.tif', -9999, None),
    ('CLYPPT_10sec.tif', 255, None),
    ('CRFVOL_10sec.tif', 255, None),
    ('hillshade_10sec.tif', 181, None),
    ('HISTPR_10sec.tif', 255, None),
    ('livestock_Bf_2010_5min.tif', -9999, 0.0),
    ('livestock_Ch_2010_5min.tif', -1.7e308, 0.0),
    ('livestock_Ct_2010_5min.tif', -1.7e308, 0.0),
    ('livestock_Dk_2010_5min.tif', -1.7e308, 0.0),
    ('livestock_Gt_2010_5min.tif', -1.7e308, 0.0),
    ('livestock_Ho_2010_5min.tif', -1.7e308, 0.0),
    ('livestock_Pg_2010_5min.tif', -1.7e308, 0.0),
    ('livestock_Sh_2010_5min.tif', -1.7e308, 0.0),
    ('ndvcec015_10sec.tif', 0, None),
    ('night_lights_10sec.tif', None, None),
    ('night_lights_5min.tif', None, None),
    ('nitrogen_10sec.tif', -9999, None),
    ('ocd_10sec.tif', -9999, None),
    ('OCDENS_10sec.tif', 9999, None),
    ('ocs_10sec.tif', -9999, None),
    ('OCSTHA_10sec.tif', 9999, None),
    ('phh2o_10sec.tif', -9999, None),
    ('PHIHOX_10sec.tif', 255, None),
    ('PHIKCL_10sec.tif', 255, None),
    ('population_2015_30sec.tif', 3.4028235e+38, 0.0),
    ('population_2015_5min.tif', 3.4028235e+38, 0.0),
    ('sand_10sec.tif', -9999, None),
    ('silt_10sec.tif', -9999, None),
    ('slope_10sec.tif', None, None),
    ('soc_10sec.tif', -9999, None),
    ('tri_10sec.tif', 0, None),
    ('wind_speed_10sec.tif', -999, None),
]

LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)


@retrying.retry(wait_exponential_multiplier=100, wait_exponential_max=5000)
def download_gs(base_uri, target_path, skip_if_target_exists=False):
    """Download base to target."""
    try:
        if not(skip_if_target_exists and os.path.exists(target_path)):
            subprocess.run(
                f'/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil cp '
                f'{base_uri} {target_path}',
                check=True, shell=True)
    except Exception:
        LOGGER.exception(
            f'exception during download of {base_uri} to {target_path}')
        raise


def fetch_data(data_dir, task_graph):
    """Download all the global data needed to run this analysis.

    Args:
        bounding_box (list): minx, miny, maxx, maxy list to clip to
        data_dir (str): path to directory to copy clipped rasters
            to
        task_graph (TaskGraph): taskgraph object to schedule work.

    Returns:
        List of (file_path, nodata, nodata_replacement) tuples.

    """
    try:
        os.makedirs(data_dir)
    except OSError:
        pass

    downloaded_file_list = []
    for file_uri, nodata, nodata_replacement in \
            CARBON_EDGE_MODEL_DATA_NODATA + [
                (BACCINI_10s_2014_BIOMASS_URI, None, None)]:
        target_file_path = os.path.join(
            data_dir, os.path.basename(file_uri))
        _ = task_graph.add_task(
            func=download_gs,
            args=(file_uri, target_file_path),
            kwargs={'skip_if_target_exists': True},
            target_path_list=[target_file_path],
            task_name=f'download {file_uri} to {data_dir}')
        if file_uri != BACCINI_10s_2014_BIOMASS_URI:
            downloaded_file_list.append(
                (target_file_path, nodata, nodata_replacement))

    task_graph.join()
    return downloaded_file_list
