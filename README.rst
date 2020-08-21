.. default-role:: code

Spatially Dependent Global Carbon Model
=======================================

This model predicts CO2 stocks in a landscape given a georeferenced spatial
map indicating regions of cropland, urban, forest, and other landcover types.
The underlying model is complex, capturing the relationships between local
and distant biological, geological, and human factors.

Running the Model
-----------------

Step 1 -- Create a simple landcover classification
**************************************************

(This step is not necessary if you already have a raster defined as below)

This model uses 4 landcover types to help predict forest carbon:

 * 1: cropland
 * 2: urban
 * 3: forest
 * 4: other

This model includes a script ``esa_to_carbon_model_landcover_types.py`` to
help with this process. It can be called at the command line as follows:

``python esa_to_carbon_model_landcover_types.py esa_lulc.tif carbon_model_landcover_types.tif --clipping_shapefile_path aoi.gpkg``

Here, ``esa_lulc.tif`` is the base ESA landcover map, ``carbon_model_landcover_types.tif`` is the desired output raster which is the conversion of the ESA landcover map to a 1-4 integer mask suitable for this model, and ``--clipping_shapefile_path aoi.gpkg`` is an optional argument to that can clip the base ``esa_lulc.tif`` raster to a smaller area of interest and/or reprojection.

Step 2 -- Run the Carbon Model
******************************

This step requires that you have a raster with the four landcover types described in Step 1. that raster is called ``carbon_model_landcover_types.tif`` the model can be run as follows:

``python carbon_edge_model.py --landtype_mask_raster_path carbon_model_landcover_types.tif --workspace_dir path_to_workspace``

This script will make a directory in the current directory called
``path_to_workspace`` as specified above. When complete, the root of this directory will contain the output file
``biomass_per_ha_stocks_{mask}.tif'`` where ``mask`` is the basename of the input landtype mask raster.

Note: this model requires several gigabytes of global data to operate. On a
first run this model will automatically download these data to a subdirectory in the workspace named ``data``. So long as
the same workspace is used on subsequent runs, the model will reuse those
data rather than re-download.

Installing Dependencies
-----------------------

The Python dependencies for this model are listed in ``requirements.txt`` but
it also requires that the Google Cloud SDK be installed. To simplify this
requirement we provide a Docker image that can be used to run the model
without any additional dependency requirements. It can be run as follows:

(Windows)
*********

``docker run --rm -it -v "%CD%":/usr/local/workspace therealspring/inspring:latest carbon_edge_model.py mask.tif``

(Linux)
*******

``docker run --rm -it -v `pwd`:/usr/local/workspace therealspring/inspring:latest carbon_edge_model.py mask.tif``
