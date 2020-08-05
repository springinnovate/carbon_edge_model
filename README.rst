.. default-role:: code

Spatially Dependent Global Carbon Model
=======================================

This model predicts CO2 stocks in a landscape given a georeferenced spatial
map indicating regions of cropland, urban, forest, and other landcover types.
The underlying model is complex, capturing the relationships between local
and distant biological, geological, and human factors.

Running the Model
-----------------

The input to this model is a single raster that must be georeferenced on
Earth with four pixel types describing:

 * 1: cropland
 * 2: urban
 * 3: forest
 * 4: other

Assuming that raster is called ``carbon_model_landcover_mask_type.tif`` the
model can be run as follows:

``python carbon_edge_model.py mask.tif``

This script will make a directory in the current directory called
``carbon_model_workspace`` (unless overridden with the ``--workspace_dir``
flag). When complete, the root of this directory will contain the output file
``c_stocks_{mask}.tif'`` where ``mask`` is the basename of the input mask.

Note: this model requires several gigabytes of global data to operate. On a
first run this model will automatically download these data and so long as
the same workspace is used, on subsequent runs the model will reuse those
data rather than re-download. These data can be found in the ``data``
subdirectory of the workspace.

(Additional Flags)
*****************

 * ``--workspace_dir [local dir]`` override the default workspace directory
 * ``--co2`` calculate carbon stocks in CO2 instead of C.

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


ESA To Mask Script (Optional)
-----------------------------

A script is provided that can translate an ESA style landcover map to the
input necessary to run this model. It can also be provided with a shapefile
to locally clip and project a region of interest. This script is not
necessary to run this model, but provided as a convenience to the user. It
can be run as follows:

``python esa_to_carbon_edge_mask.py [esa_landcover.tif] [carbon_model_landcover_mask_type.tif] --clipping_shapefile_path [region_of_interest.gpkg]``

The first argument ``esa_landcover.tif`` is a path to an ESA styled landcover
map, it can be in any projection. The second argument,
``carbon_model_landcover_mask_type.tif`` is the desired output file which
will be used as an input to the model. The last argument,
``--clipping_shapefile_path`` is optional and if provided points to a
shapefile that will be used to clip and project the output. If not provided
the output mask will be the same size as projection as ``esa_landcover.tif``
