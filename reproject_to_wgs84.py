import logging
import os

import ecoshard

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))

RASTER_LIST = [
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/accessibility_to_cities_2015_30sec_compressed_md5_c8b0cede8a8f6b0f004c8b97586ea61a.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/altitude_10sec_compressed_md5_5f2c8b4e26ec969819134109181c3744.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_01_30sec_compressed_md5_6f0ba86674e14d3e2a11d9f66282df51.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_02_30sec_compressed_md5_4a7139ff1bcde6a384cc3824d93e3aeb.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_03_30sec_compressed_md5_b0d5cd27de607125451efa648cae58a7.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_04_30sec_compressed_md5_9cbe6c4a4c22ae3fda829a68ebb5c3ab.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_05_30sec_compressed_md5_f3e26f183e4add02cac1c984775618c3.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_06_30sec_compressed_md5_0b2ab91b48920df38ca4455b46313797.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_07_30sec_compressed_md5_6e3b749fb1ae93d7283a73b24a76e02c.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_08_30sec_compressed_md5_dacfe3568b4510d371b0dd4b719400a0.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_09_30sec_compressed_md5_40120ac7b65703b6f93eec2ba771be99.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_10_30sec_compressed_md5_cfc0444c884753ae9c01237dcdbddf67.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_11_30sec_compressed_md5_92f12d876ee52439fe403096764c7519.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_12_30sec_compressed_md5_1466feb920dd5defcbbe7afa6a713966.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_13_30sec_compressed_md5_c25eba18f88adb7576e4221293b79d46.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_14_30sec_compressed_md5_81716ca53ef4308c06d9334cbdd34fc2.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_15_30sec_compressed_md5_eff27479e3a40a134dc794c0f755ce85.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_16_30sec_compressed_md5_5617089f98f296129e27223c203778aa.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_17_30sec_compressed_md5_152fe6a9be238c8e125dbb304f1406fe.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_18_30sec_compressed_md5_13910428a50f5a3a2a0c49d4e35f68ff.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/bio_19_30sec_compressed_md5_34b16f0cc7d11e1c20c1d74280008f76.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/hillshade_10sec_compressed_md5_0973aa325db643290320ce8a2afbdf49.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/livestock_Bf_2010_5min_compressed_md5_9291ed6ebb8fa0784caaf756ff49e6a1.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/livestock_Ch_2010_5min_compressed_md5_7b9436a725ae19e78adca5f1a253ef68.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/livestock_Ct_2010_5min_compressed_md5_a1baa8737123585a1e2852e4ad27ab3b.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/livestock_Dk_2010_5min_compressed_md5_9e5d7ed72481011963d18a4cab6c59e8.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/livestock_Gt_2010_5min_compressed_md5_ffb9ff487f044dc3c799e510053f68b7.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/livestock_Ho_2010_5min_compressed_md5_b4454fc97c1fc6f9937e4488b8a03482.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/night_lights_10sec_compressed_md5_0d66b62beb113326848a49ebab369105.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/night_lights_5min_compressed_md5_f69d0392bd9cd5537417f3656dfa126d.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/nitrogen_10sec_compressed_md5_6994f7cbb2974ab2c6b07f0941e2d2ad.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/slope_10sec_compressed_md5_939da641aaa7f72bdd143c64a81cbad6.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/tri_10sec_compressed_md5_021caeb308060476b216c1bba57514d7.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/wind_speed_10sec_compressed_md5_ec54562e1a6d307e532b767989f48a13.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/population_2015_30sec_compressed_md5_676c2ff75cebe0a4fcd090dfecc7a037.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/population_2015_5min_compressed_md5_4c267e3cb681689acc08020fd64e023d.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/cec_0-5cm_mean_compressed_md5_fcb258ec64c03d494f6f37811e1953e7.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/cec_0-5cm_uncertainty_compressed_md5_da49cc29b7e92932636bef2fcb59f2bc.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/cec_5-15cm_mean_compressed_md5_2237766c8236006be2ae6b533c18ce1b.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/cec_5-15cm_uncertainty_compressed_md5_d5bbaf58ccce257fa9b6c848ebeb1438.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/cfvo_0-5cm_mean_compressed_md5_559e5694539eebc1c1812d097f51f264.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/cfvo_0-5cm_uncertainty_compressed_md5_3ceda87f2ff831a5bca7dcddbba8a0ec.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/cfvo_5-15cm_mean_compressed_md5_2d0ca616540fac16f337111d161044c7.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/cfvo_5-15cm_uncertainty_compressed_md5_c4708eda6aae30dbda28957419a4aeef.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/clay_0-5cm_mean_compressed_md5_8811e315c128b13d19b91eedd38f3289.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/clay_0-5cm_uncertainty_compressed_md5_d2d5413cad4be779f67633f055ec7edd.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/clay_5-15cm_mean_compressed_md5_f3034943f4c27c2e34cd8b1c3c9eae12.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/clay_5-15cm_uncertainty_compressed_md5_2bc6390a2f1be9a148364daf1d74cd1e.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/nitrogen_0-5cm_mean_compressed_md5_982afcaa250504dda1c74a9305bd4dfb.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/nitrogen_0-5cm_uncertainty_compressed_md5_0a764e8e11c095a6198cbe0d57ff17e9.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/nitrogen_5-15cm_mean_compressed_md5_40cc7a4f8dc6e3f20477b26e61a7fe14.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/nitrogen_5-15cm_uncertainty_compressed_md5_0a20d836d1d3e10ee369e100b864267b.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/phh2o_0-5cm_mean_compressed_md5_cf6d71bd6fb983f0b95da9a4a42daa5f.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/phh2o_0-5cm_uncertainty_compressed_md5_53cbff25993947a31426ad9eeeac89a8.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/phh2o_5-15cm_mean_compressed_md5_a4791fe139d07654cd51539a3ef4cee0.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/phh2o_5-15cm_uncertainty_compressed_md5_b44c67d80604a9cb678a907961f13b6b.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/sand_0-5cm_mean_compressed_md5_39ce08e191c75f63fd0b876d41b54688.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/sand_0-5cm_uncertainty_compressed_md5_e7034724303f6dce155ad29ad450102e.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/sand_5-15cm_uncertainty_compressed_md5_4ecbda7ddf5504e5cc6a86ab3986239d.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/silt_0-5cm_mean_compressed_md5_2d035d45e08cc5a442bbbffcc1ebc0c7.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/silt_0-5cm_uncertainty_compressed_md5_bd64b35062130e80bdf76ebde9f91f58.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/silt_5-15cm_mean_compressed_md5_8393319c8345da12c664d39b67a0afa4.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/silt_5-15cm_uncertainty_compressed_md5_8277d8b498e593cb352be7bf43592be6.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/soc_0-5cm_mean_compressed_md5_076f6fa676ab399577e5881d4aa3784e.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/soc_0-5cm_uncertainty_compressed_md5_e7f40d8b08e2ad9b128e916b93b346b5.tif',
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs/soc_5-15cm_mean_compressed_md5_6bf0892a2bbe8ee00f84fafd84302f29.tif',
]

WORKSPACE_DIR = 'workspace'
ORIGINAL_DIR = os.path.join(WORKSPACE_DIR, 'originals')
PROJECTED_DIR = os..path.join(WORKSPACE_DIR, 'projected')
WARP_PIXEL_SIZE = (0.002777777777777777884, -0.002777777777777777884)
WARP_PROJ_WKT = osr.SRS_WKT_WGS84_LAT_LONG
WARP_BB = [-179, -60, 179, 75]
for dir_path in [ORIGINAL_DIR, PROJECTED_DIR]:
    os.makedirs(dir_path)


def download_warp_and_hash(base_url, target_dir):
    """Download ``base_url`` but warp, clip, and hash it to ``target_dir``."""
    original_path = os.path.join(ORIGINAL_DIR, os.path.basename(base_url))
    ecoshard.download_url(base_url, original_path)
    warp_raster_path = os.path.join(
        PROJECTED_DIR, os.path.basename(original_path))
    pygeoprocessing.warp_raster(
        original_path, WARP_PIXEL_SIZE,
        warp_raster_path,
        'near', target_bb=WARP_BB,
        target_projection_wkt=WARP_PROJ_WKT)
    pre, _, suf = re.match(
        '(.*)(_md5.*)(\\.tif)', warp_raster_path).groups()

    os.rename(warp_raster_path, f'{pre}_wgs84_{suf}')
    warp_raster_path = f'{pre}_wgs84_{suf}'
    ecoshard.hash_file(warp_raster_path, rename=True)


if __name__ == '__main__':
    thread_list = []
    for raster_url in RASTER_LIST:
        download_thread = threading.thread(
            target=download_warp_and_hash,
            args=(raster_url, PROJECTED_DIR))
        download_thread.start()
        thread_list.append(download_thread)
    for thread in thread_list:
        thread.join()
