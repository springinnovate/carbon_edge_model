CALL python train_regression_model.py sampled_points_2022_07_07_17_36_50_md5_9c3e8f_scrubbed.gpkg model_hansen.csv --interaction_ids gf_5.0_fc_stack_hansen_forest_cover2014_compressed --prefix hansen_
CALL python train_regression_model.py sampled_points_2022_07_07_17_36_50_md5_9c3e8f_scrubbed.gpkg model_hansen_and_esa.csv --interaction_ids gf_5.0_fc_stack_hansen_forest_cover2014_compressed gf_5.0_masked_forest_ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7 --prefix hansen_and_esa_
CALL python train_regression_model.py sampled_points_2022_07_07_17_36_50_md5_9c3e8f_scrubbed.gpkg model_lat_lng.csv --prefix lat_lng_
CALL python train_regression_model.py sampled_points_2022_07_07_17_36_50_md5_9c3e8f_scrubbed.gpkg model_no_forest_edge.csv --prefix no_forest_edge_
CALL python train_regression_model.py sampled_points_2022_07_07_17_36_50_md5_9c3e8f_scrubbed.gpkg model_esa.csv --interaction_ids gf_5.0_masked_forest_ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7 --prefix esa_
CALL python train_regression_model.py sampled_points_2022_07_07_17_36_50_md5_9c3e8f_scrubbed.gpkg model_hansen_lat.csv --interaction_ids gf_5.0_fc_stack_hansen_forest_cover2014_compressed --prefix hansen_lat_
