.. default-role:: code

Spatially Dependent Global Carbon Model
=======================================

This model predicts biomass density in a landscape given a georeferenced spatial map indicating regions of cropland, urban, forest, and other landcover types. The underlying model is complex, capturing the relationships between local and distant biological, geological, and human factors.

Training the model
------------------

The model trains off of a set of fixed predictors that will always be present in any model run (biophysical quantities) and a landcover derived set of predictors provided by the user when running predictions. The landcover raster will be filtered to extract forest, urban, and agriculture masks. Response carbon stock data is associated with the same year as the landcover predictor data.

Step 1 -- Align/project base rasters
************************************

Place all the rasters you want to use in the model including predictors and response variable in a folder.

``python align_rasters_for_carbon_model.py path_to_raster_folder/*.tif``

This creates a directory called ``processed_rasters`` which is a copy of all the rasters in the base folder but aligned and projected into the same dimensions.

Step 2 -- Extract landcover type masks
**************************************

In this step the landcover rasters are processed into forest, urban, and crop mask rasters. These are used to identify where the model should be run (forest only) but also in a later step to measure any distance weighted influence from those landcover types. To extract masks from any number of ESA landcover rasters run the following command

``python extract_landcover_masks.py processed_rasters/[landcover pattern].tif``

This script expects landcover rasters to be in the ESA format with the following classifications:
  * cropland: 10-41
  * urban 190
  * forest: 50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 160, 170

For each file matched in the pattern, 3 files are generated preprended with
``masked_cropland``, ``masked_urban``, or ``masked_forest``.

Step 3 -- Calculate distance weighted influence
***********************************************

The three masktypes in the previous step should be distance weighted based on
the expected influence of carbon edge densities using this command

``python gaussian_filter_rasters.py processed_rasters/masked_*.tif --kernel_distance_list 0.4 1.45``

The kernel distances of 0.4km, and 1.45km, were selected from experimental observation of expected maximum edge effect distances in two types of forest.

In this example, every file matched in the ``../masked_*.tif`` pattern, will generate two new files with the names ``gf_0.4_[original filename]`` and ``gf_1.4_[original filename]``. The ``gf`` stands for "Gaussian Filter" and the floating point refers to the "maximum expected edge effect". Practically, this sets the sigma and the radius of the maximum range of the kernel to the value of the maximum expected edge effect.

Step 4 -- Sample rasters into point dataset
*******************************************
Rasters are then sampled into a point dataset and separated by base points and holdback test set. The following command samples all the rasters in the ``processed_rasters`` directory:


``python sample_data.py --sample_rasters processed_rasters/*.tif --holdback_centers (1.270418,-56.916893)  (46.809309,-123.475978) ( 3.104814,27.546842) (15.833150,104.030651) --holdback_margin 2.0 --n_samples 100000 --sample_vector_path countries.gpkg``

Arguments are as follows:
  * ``--holdback_centers`` a list of coordinates in the format (lat,lng) to define holdback boxes for validation.
  * ``--holdback_margin`` with of margin around the holdback bounding box to avoid sampling in units of degrees.
  * ``--n_samples`` number of sample points to generate.
  * ``--sample_vector_path`` if provided, limits sample points to the areas within the polygon of this vector.

This creates a point GPKG file in the current directory named ``sampled_points_[bounding box]_md5_[hash of file].gpkg`` that has the fields
  * ``holdback`` true if in the holdback point set, false if not
  * n other fields named after the rasters in the ``--sample_rasters`` input argument. Note the name of this file for the next step.

Step 5 -- Create predictor response table
*****************************************

This step creates a table that's used to specify what data are predictors and responses, which data are actually a single dataset, any limits on the data to sample mean in the model, and any limits on what data to use in training the model. Any exclusions or inclusions refer to accepting or rejecting an entire sample **vector**, not just a particular data point. For example, if there is 1 predictror and 10 responses, a single sample will consist of 11 numerical values sampled from 11 different rasters. If any of the exclusions are triggered below (outside of min/max, in ``exclude``, or not in ``include``) the entire sample vector will be rejected as a point to train the model.

The CSV headers are explained in detail here:

* ``predictor`` - a text ID that's used to identify a field in the GPKG created in the previous step. This ID is also the base name of the raster that was sampled in Step 3.
* ``response`` - same as a ``predictor`` but will act as a response when training the model
* ``include`` - specifies that the value here is the only one included when sampling data. Useful in the case where there is a ``predictor`` that may be a mask only, such as a raster that indicates where forest is when you only want to train a model on forest.
* ``exclude`` - opposite of ``include`` and used to exclude any samples which include this value in the sample, useful for specifying nodata values.
* ``min`` and ``max`` - numeric minimums and maximums to limit the sampling. Any values that lie outside this range will not generate a sample vector.
* ``group`` and ``target`` - used to "group" differently named raster samples into a single sample pool. For example, if there are several ``response`` rasters but they refer to different years of data, all are needed to train the model as a single predictor, you would list every individual year as a separate response and give it a unique ``group`` per predictor, but an identical ``target``. For example, if you have 5 rasters as responses, and 10 predictors, but all 5 rasters have the same ``target``, there will be 5 samples of 11 units where all the predictors are identical but the responses are unique.
* ``filter_only`` - if set to 1 then this raster is not included in the sample vector, but is only used to accept or reject a sample vector given any of the filter options. This is useful if there is an input predictor that is a mask, such as forest, but otherwise always has the same value.

Below is an example of such a table:

+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| predictor                                                                        | response  | include  | exclude  | min  | max  | group  | target               | filter_only  |
+==================================================================================+===========+==========+==========+======+======+========+======================+==============+
| baccini_carbon_data_2003_2014_compressed_md5_11d1455ee8f091bf4be12c4f7ff9451b_1  |           |          |          | 50   | 400  | 2003   | baccini_carbon_data  |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| baccini_carbon_data_2003_2014_compressed_md5_11d1455ee8f091bf4be12c4f7ff9451b_2  |           |          |          | 50   | 400  | 2004   | baccini_carbon_data  |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| baccini_carbon_data_2003_2014_compressed_md5_11d1455ee8f091bf4be12c4f7ff9451b_3  |           |          |          | 50   | 400  | 2005   | baccini_carbon_data  |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| ESACCI-LC-L4-LCCS-Map-300m-P1Y-2003_forest                                       |           | 1        |          |      |      |        |                      | 1            |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| ESACCI-LC-L4-LCCS-Map-300m-P1Y-2004_forest                                       |           | 1        |          |      |      |        |                      | 1            |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| ESACCI-LC-L4-LCCS-Map-300m-P1Y-2005_forest                                       |           | 1        |          |      |      |        |                      | 1            |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| altitude_10sec_compressed_wgs84__md5_bfa771b1aef1b18e48962c315e5ba5fc            |           |          |          |      |      |        |                      |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| bio_02_30sec_compressed_wgs84__md5_7ad508baff5bbd8b2e7991451938a5a7              |           |          |          |      |      |        |                      |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| bio_03_30sec_compressed_wgs84__md5_a2de2d38c1f8b51f9d24f7a3a1e5f142              |           |          |          |      |      |        |                      |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| bio_05_30sec_compressed_wgs84__md5_bdd225e46613405c80a7ebf7e3b77249              |           |          |          |      |      |        |                      |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| bio_08_30sec_compressed_wgs84__md5_baf898dd624cfc9415092d7f37ae44ff              |           |          |          |      |      |        |                      |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| silt_5-15cm_mean_compressed_wgs84__md5_d0abb0769ebd015fdc12b50b20f8c51e          |           |          |          |      |      |        |                      |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| slope_10sec_compressed_wgs84__md5_e2bdd42cb724893ce8b08c6680d1eeaf               |           |          |          |      |      |        |                      |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| soc_0-5cm_mean_compressed_wgs84__md5_b5be42d9d0ecafaaad7cc592dcfe829b            |           |          |          |      |      |        |                      |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| soc_5-15cm_mean_compressed_wgs84__md5_4c489f6132cc76c6d634181c25d22d19           |           |          |          |      |      |        |                      |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| tri_10sec_compressed_wgs84__md5_258ad3123f05bc140eadd6246f6a078e                 |           |          |          |      |      |        |                      |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+
| wind_speed_10sec_compressed_wgs84__md5_7c5acc948ac0ff492f3d148ffc277908          |           |          |          |      |      |        |                      |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+------+------+--------+----------------------+--------------+


Utility to make model config table
==================================

The script at ``utils/build_skeleton_model_config.py`` can be used to generate a base CSV model config file that is easier to manipulate than writing from scratch. To use it:

``python utils/build_skeleton_model_config.py [path to .gpkg] --output_filename [model_config_table.csv]``

The generated CSV has all the necessary columns specified above and every data column in the geopackage (except ``holdback`` and ``geometry``) is listed in the ``predictor`` column. It is easier to manipulate that table than it would be to write it from scratch.

Step 6 -- Train the model
*************************

Given the sample point dataset and a model configuration file created above, a regression model can be trained with the following command:

``python train_regression_model.py [path to point .gpkg] [path to model configuration .csv]``

This will train the model with several techniques that will get logged to the console. Each model reports its fitness with a scatterplot figure located at ``fig_dir/[regression model]_[training|holdback].png`` which contains the fit line, R^2 and adjusted R^2 and information whether it was the training or holdback test set.

The saved model will be located at ``[prefix]_model.dat`` in the current
working directory and can be used in the next step.

Step 7 -- Run the model on custom forest cover
**********************************************

To run the model the user provides a path to the model created in Step 6,
ensures the base model data are local to the machine the model is run on, and provides a raster of forest mask projected in meters. Assuming these data are present the user can invoke the following command:

```
python .\run_model.py PATH_TO_MODEL.dat PATH_TO_CUSTOM_FOREST_RASTER.tif --predictor_raster_dir PATH_TO_DIRECTORY_THAT_CONTAINS_PREDICTOR_RASTERS_USED_TO_TRAIN_MODEL
```

The script will output three rasters:
    * ``CUSTOM_FOREST_RASTER_NAME_std_forest_edge_result.tif``: modeled forest carbon using standard model.
    * ``CUSTOM_FOREST_RASTER_NAME_full_forest_edge_result.tif``: modeled forest carbon if a full "edge" effect is present in all forest edge pixels.
    * ``CUSTOM_FOREST_RASTER_NAME_no_forest_edge_result.tif``: modeled forest carbon if no "edge" effect is present in all forest edge pixels.
