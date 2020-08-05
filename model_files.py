FOREST_REGRESSION_LASSO_TABLE_URI = (
    'gs://ecoshard-root/global_carbon_regression/'
    'lasso_interacted_not_forest_gs1to100_nonlinear_'
    'alpha0-0001_params_namefix.csv')
BACCINI_10s_2014_BIOMASS_URI = (
    'gs://ecoshard-root/global_carbon_regression/baccini_10s_2014'
    '_md5_5956a9d06d4dffc89517cefb0f6bb008.tif')

CARBON_EDGE_REGRESSION_MODEL_URI_LIST = [
    'gs://ecoshard-root/global_carbon_regression/inputs/ACDWRB_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/AWCh1_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/AWCh2_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/AWCh3_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/AWCtS_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/BDRICM_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/BDRLOG_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/BDTICM_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/BLDFIE_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/CECSOL_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/CLYPPT_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/CRFVOL_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/HISTPR_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/OCDENS_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/OCSTHA_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/PHIHOX_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/PHIKCL_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/accessibility_to_cities_2015_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/altitude_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bdod_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_01_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_02_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_03_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_04_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_05_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_06_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_07_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_08_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_09_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_10_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_11_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_12_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_13_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_14_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_15_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_16_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_17_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_18_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/bio_19_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/cec_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/cfvo_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/clay_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/ecozone_country_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/hillshade_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/is_cropland_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/is_urban_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/livestock_Bf_2010_5min.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/livestock_Ch_2010_5min.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/livestock_Ct_2010_5min.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/livestock_Dk_2010_5min.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/livestock_Gt_2010_5min.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/livestock_Ho_2010_5min.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/livestock_Pg_2010_5min.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/livestock_Sh_2010_5min.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_2014_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_cropland_10sec_gs1.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_cropland_10sec_gs10.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_cropland_10sec_gs100.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_cropland_10sec_gs2.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_cropland_10sec_gs20.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_cropland_10sec_gs3.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_cropland_10sec_gs30.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_cropland_10sec_gs5.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_cropland_10sec_gs50.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_urban_10sec_gs1.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_urban_10sec_gs10.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_urban_10sec_gs100.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_urban_10sec_gs2.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_urban_10sec_gs20.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_urban_10sec_gs3.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_urban_10sec_gs30.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_urban_10sec_gs5.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_is_urban_10sec_gs50.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_not_forest_10sec_gs1.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_not_forest_10sec_gs10.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_not_forest_10sec_gs100.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_not_forest_10sec_gs2.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_not_forest_10sec_gs20.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_not_forest_10sec_gs3.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_not_forest_10sec_gs30.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_not_forest_10sec_gs5.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/lulc_esa_smoothed_2014_10sec_not_forest_10sec_gs50.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/ndvcec015_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/night_lights_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/night_lights_5min.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/nitrogen_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/not_forest_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/ocd_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/ocs_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/phh2o_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/population_2015_30sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/population_2015_5min.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/results_2020-07-07b_md5_fe89d44cf181486b384beb432c253d47.zip',
    'gs://ecoshard-root/global_carbon_regression/inputs/sand_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/silt_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/slope_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/soc_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/tri_10sec.tif',
    'gs://ecoshard-root/global_carbon_regression/inputs/wind_speed_10sec.tif',
    ]
