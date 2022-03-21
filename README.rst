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

``python sample_data.py --sample_rasters processed_rasters/*.tif --holdback_bb 15.0 -9 33.0 9 --holdback_margin 2.0 --n_samples 100000 --country_polygon_path countries.gpkg``

Arguments are as follows:
  * ``--holdback_bb`` a bounding box in (lng min, lat min, lng max, lat max) format to define as the holdback test set for validation. Note this bounding box must be in WGS84 even if input rasters are not.
  * ``--holdback_margin`` with of margin around the holdback bounding box to avoid sampling in units of degrees.
  * ``--n_samples`` number of sample points to generate.
  * ``--country_polygon_path`` if provided, limits sample points to the areas within the polygon of this vector.

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
* ``exclude`` - opposite of ``include`` and used to exclude any samples which include this value in the sample
* ``min`` and ``max`` - numeric minimums and maximums to limit the sampling. Any values that lie outside this range will not generate a sample vector.
* ``group`` and ``target`` - used to "group" differently named raster samples into a single sample pool. For example, if there are several ``response`` rasters but they refer to different years of data, all are needed to train the model as a single predictor, you would list every individual year as a separate response and give it a unique ``group`` per predictor, but an identical ``target``. For example, if you have 5 rasters as responses, and 10 predictors, but all 5 rasters have the same ``target``, there will be 5 samples of 11 units where all the predictors are identical but the responses are unique.
* ``filter_only`` - if set to 1 then this raster is not included in the sample vector, but is only used to accept or reject a sample vector given any of the filter options. This is useful if there is an input predictor that is a mask, such as forest, but otherwise always has the same value.

Below is an example of such a table

+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| predictor                                                                        | response  | include  | exclude  | min                  | max  | group  | target  | filter_only  |
+==================================================================================+===========+==========+==========+======================+======+========+=========+==============+
| baccini_carbon_data_2003_2014_compressed_md5_11d1455ee8f091bf4be12c4f7ff9451b_1  | 50        | 400      | 2003     | baccini_carbon_data  |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| baccini_carbon_data_2003_2014_compressed_md5_11d1455ee8f091bf4be12c4f7ff9451b_2  | 50        | 400      | 2004     | baccini_carbon_data  |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| baccini_carbon_data_2003_2014_compressed_md5_11d1455ee8f091bf4be12c4f7ff9451b_3  | 50        | 400      | 2005     | baccini_carbon_data  |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| ESACCI-LC-L4-LCCS-Map-300m-P1Y-2003_forest                                       | 1         | 1        |          |                      |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| ESACCI-LC-L4-LCCS-Map-300m-P1Y-2004_forest                                       | 1         | 1        |          |                      |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| ESACCI-LC-L4-LCCS-Map-300m-P1Y-2005_forest                                       | 1         | 1        |          |                      |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| altitude_10sec_compressed_wgs84__md5_bfa771b1aef1b18e48962c315e5ba5fc            |           |          |          |                      |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| bio_02_30sec_compressed_wgs84__md5_7ad508baff5bbd8b2e7991451938a5a7              |           |          |          |                      |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| bio_03_30sec_compressed_wgs84__md5_a2de2d38c1f8b51f9d24f7a3a1e5f142              |           |          |          |                      |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| bio_05_30sec_compressed_wgs84__md5_bdd225e46613405c80a7ebf7e3b77249              |           |          |          |                      |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| bio_08_30sec_compressed_wgs84__md5_baf898dd624cfc9415092d7f37ae44ff              |           |          |          |                      |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| silt_5-15cm_mean_compressed_wgs84__md5_d0abb0769ebd015fdc12b50b20f8c51e          |           |          |          |                      |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| slope_10sec_compressed_wgs84__md5_e2bdd42cb724893ce8b08c6680d1eeaf               |           |          |          |                      |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| soc_0-5cm_mean_compressed_wgs84__md5_b5be42d9d0ecafaaad7cc592dcfe829b            |           |          |          |                      |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| soc_5-15cm_mean_compressed_wgs84__md5_4c489f6132cc76c6d634181c25d22d19           |           |          |          |                      |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| tri_10sec_compressed_wgs84__md5_258ad3123f05bc140eadd6246f6a078e                 |           |          |          |                      |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+
| wind_speed_10sec_compressed_wgs84__md5_7c5acc948ac0ff492f3d148ffc277908          |           |          |          |                      |      |        |         |              |
+----------------------------------------------------------------------------------+-----------+----------+----------+----------------------+------+--------+---------+--------------+


Step 5 -- Train the model
*************************






Running the Model
-----------------

Step 1 -- Create a simple landcover classification
**************************************************

(This step is not necessary if you already have a raster defined as below)

This model uses 4 landcover types to help predict forest carbon:

 * 1: cropland
 * 2: urban
 * 3: forest
 * 4: other landcover types

This model includes a script ``utils/esa_to_carbon_model_landcover_types.py`` to help with this process. It can be called at the command line as follows:

``python utils/esa_to_carbon_model_landcover_types.py esa_lulc.tif carbon_model_landcover_types.tif --clipping_shapefile_path aoi.gpkg``

Here, ``esa_lulc.tif`` is the base ESA landcover map, ``carbon_model_landcover_types.tif`` is the desired output raster which is the conversion of the ESA landcover map to a 1-4 integer mask suitable for this model, and ``--clipping_shapefile_path aoi.gpkg`` is an optional argument to that can clip the base ``esa_lulc.tif`` raster to a smaller area of interest and/or reprojection.

Step 2 -- Run the Carbon Model
******************************

This step requires that you have a raster with the four landcover types described in Step 1. that raster is called ``carbon_model_landcover_types.tif`` the model can be run as follows:

``python carbon_edge_model.py --landcover_type_raster_path carbon_model_landcover_types.tif``

This script will make a directory in the current directory called ``carbon_model_workspace``. When complete, the root of this directory will contain the output file ``biomass_per_ha_stocks_{mask}.tif'`` where ``mask`` is the basename of the input landtype mask raster.

Note: this model requires several gigabytes of global data to operate. When the model is run for the first time it will automatically download these data to a subdirectory in the workspace named ``data``. As long as the same workspace is used on subsequent runs, the model will reuse those
data rather than re-download.

Installing Dependencies
-----------------------

The Python dependencies for this model are listed in ``requirements.txt`` but it also requires that the Google Cloud SDK be installed. To simplify this requirement we provide a Docker image that can be used to run the model without any additional dependency requirements. It can be run as follows:

(Windows)
*********

``docker run --rm -it -v "%CD%":/usr/local/workspace therealspring/inspring:latest carbon_edge_model.py mask.tif``

(Linux)
*******

``docker run --rm -it -v `pwd`:/usr/local/workspace therealspring/inspring:latest carbon_edge_model.py mask.tif``

Utility Scripts
---------------

The following utility scripts are available in ``./utils``

 * ``create_marginal_value.py`` used to subtract one raster from another of to create a marginal value map. Use as follows:

    ``python utils/create_marginal_value.py --base_value_raster_path base.tif --scenario_value_raster_path scenario.tif --target_marginal_value_path marginal_value.tif``

 * ``esa_to_carbon_model_landcover_types.py`` used to convert an ESA style landcover map into the 4 catagory landcover map used in this model. Described above in **Step 1 -- Create a simple landcover classification**.

Model Builder
-------------

The ``model_builder`` contains Python code to build the regression model used by ``carbon_edge_model.py`` it need not be run by an end user but instead is provided as reference.

Directories
-----------

 * ``model_base_data`` will be generated by the ``carbon_edge_model.py`` script and will contain base data for future runs to avoid large downloads per evaluation. It should not be modified by hand.
 * ``model_run_workspace`` is the root workspace for a particular model run defined by a given landcover scenario, this directory will contain
    * ``churn`` a directory to hold intermediate files that are not useful for human inspection, and
    * ``biomass_per_ha_stocks_{base_landcover_type_raster_path_id]}.tif`` -- the output of the model.

Model Analysis
--------------

The model was generated by randomly sampling forest pixel points distribued evenly on a sphere from 35N to 35S latitude (subtropics).

against 64,000 points using the method described above.

.. image:: images/global_point_samples.png
  :width: 400
  :alt: Global point samples (100,000 shown)

.. image:: images/points_in_brazil.png
  :width: 400
  :alt: Brazil point samples zoomed for detail

Goodness of fit
***************

The table below shows the results of several training runs. The first column was the number of points selected for the model using the method in the previous section. The `r_squared` vs `r_squared_test` are the R^2 scores calculated on the training data and the holdout data respectively. In each instance 80% of the points were used for training while 20% were held out for validation. We gain confidence that the model is accurate and not overfitting at around 320,000 points but we get a slightly better R^2 at 640,000.

.. list-table:: R^2 performance vs sample points
   :widths: 25 25 50 2
   :header-rows: 1

   * - n_points
     - r_squared
     - r_squared_test
     - model used
   * - 40000
     - 0.843442966
     - 0.810843161
     -
   * - 80000
     - 0.817260654
     - 0.816668817
     -
   * - 160000
     - 0.813861854
     - 0.814232628
     -
   * - 320000
     - 0.811147318
     - 0.769127539
     -
   * - 640000
     - 0.864520049
     - 0.810522805
     - **<----------**

Example in Local Area
---------------------

Below is modeled data compared against global Baccini biomass layer:


Baccini Biomass Layer:

.. image:: images/base_baccini.PNG
  :width: 400
  :alt: Base Baccini Biomass Layer in Brazil

Modeled Biomass Layer:

.. image:: images/modeled_carbon.png
  :width: 400
  :alt: Modeled Biomass Layer in Brazil

Error:

.. image:: images/bra_error.png
  :width: 400
  :alt: Modeled Biomass Layer in Brazil

Error legend:

.. image:: images/error_legend.png
  :alt: Error Legend


Data
****

Data required for this model are automatically fetched from public Google Bucket storage (located at the root ``https://storage.googleapis.com/ecoshard-root/global_carbon_regression/inputs``). This includes the rasters listed below as well a ``scikit.learn`` pickled regression model trained using the method above.

    * ``accessibility_to_cities_2015_30sec.tif``
    * ``ACDWRB_10sec.tif``
    * ``altitude_10sec.tif``
    * ``AWCh1_10sec.tif``
    * ``AWCh2_10sec.tif``
    * ``AWCh3_10sec.tif``
    * ``AWCtS_10sec.tif``
    * ``bdod_10sec.tif``
    * ``BDRICM_10sec.tif``
    * ``BDRLOG_10sec.tif``
    * ``BDTICM_10sec.tif``
    * ``bio_01_30sec.tif``
    * ``bio_02_30sec.tif``
    * ``bio_03_30sec.tif``
    * ``bio_04_30sec.tif``
    * ``bio_05_30sec.tif``
    * ``bio_06_30sec.tif``
    * ``bio_07_30sec.tif``
    * ``bio_08_30sec.tif``
    * ``bio_09_30sec.tif``
    * ``bio_10_30sec.tif``
    * ``bio_11_30sec.tif``
    * ``bio_12_30sec.tif``
    * ``bio_13_30sec.tif``
    * ``bio_14_30sec.tif``
    * ``bio_15_30sec.tif``
    * ``bio_16_30sec.tif``
    * ``bio_17_30sec.tif``
    * ``bio_18_30sec.tif``
    * ``bio_19_30sec.tif``
    * ``BLDFIE_10sec.tif``
    * ``cfvo_10sec.tif``
    * ``clay_10sec.tif``
    * ``CLYPPT_10sec.tif``
    * ``CRFVOL_10sec.tif``
    * ``hillshade_10sec.tif``
    * ``HISTPR_10sec.tif``
    * ``livestock_Bf_2010_5min.tif``
    * ``livestock_Ch_2010_5min.tif``
    * ``livestock_Ct_2010_5min.tif``
    * ``livestock_Dk_2010_5min.tif``
    * ``livestock_Gt_2010_5min.tif``
    * ``livestock_Ho_2010_5min.tif``
    * ``livestock_Pg_2010_5min.tif``
    * ``livestock_Sh_2010_5min.tif``
    * ``ndvcec015_10sec.tif``
    * ``night_lights_10sec.tif``
    * ``night_lights_5min.tif``
    * ``nitrogen_10sec.tif``
    * ``ocd_10sec.tif``
    * ``OCDENS_10sec.tif``
    * ``ocs_10sec.tif``
    * ``OCSTHA_10sec.tif``
    * ``phh2o_10sec.tif``
    * ``PHIHOX_10sec.tif``
    * ``PHIKCL_10sec.tif``
    * ``population_2015_30sec.tif``
    * ``population_2015_5min.tif``
    * ``sand_10sec.tif``
    * ``silt_10sec.tif``
    * ``slope_10sec.tif``
    * ``soc_10sec.tif``
    * ``tri_10sec.tif``
    * ``wind_speed_10sec.tif``
    * ``baccini_10s_2014_md5_5956a9d06d4dffc89517cefb0f6bb008.tif``

Coefficents
***********

Below is a truncated version of the normalized coefficients used in the 640,000 point model. A complete and searchable table of factors and can be found at: https://github.com/therealspring/carbon_edge_model/blob/master/images/coef_640000.csv


.. list-table:: Truncated Coefficient Table (full table at https://github.com/therealspring/carbon_edge_model/blob/master/images/coef_640000.csv)
   :widths: 25 25
   :header-rows: 1

   * - Coefficient
     - Feature Term
   * - `+1.084e+02`
     - `urban_gf*forest_gf`
   * - `+5.622e+01`
     - `cropland_gf^2`
   * - `+5.146e+01`
     - `cropland_gf*forest_gf`
   * - `+4.534e+01`
     - `forest_gf^2`
   * - `-3.438e+01`
     - `cropland_gf*urban_gf`
   * - `-1.537e+01`
     - `urban_gf^2`
   * - `-9.928e+00`
     - `phh2o_10sec`
   * - `-9.519e+00`
     - `bio_11_30sec*urban_gf`
   * - `+6.126e+00`
     - `AWCh3_10sec`
   * - `-5.611e+00`
     - `bio_02_30sec*forest_gf`
   * - `-5.002e+00`
     - `PHIKCL_10sec`
   * - `-4.264e+00`
     - `bio_10_30sec*urban_gf`
   * - `+4.188e+00`
     - `tri_10sec*urban_gf`

License
-------

This software is permissively licensed under The Apache 2.0 open source license.


inputs:
  landcover
  prob
  what to flip it to


raster a
raster b
{
  [for value in a
    if b > threshold, flip a to something, else something else]
}

landcover in a, value > threshold, value < threshold
