import pygeoprocessing

raster_path = r"./workspace/ecoshards/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif"
raster_info = pygeoprocessing.get_raster_info(raster_path)
pygeoprocessing.warp_raster(
    raster_path, raster_info['pixel_size'], 'input_lulc.tif', 'near',
    target_bb=[-64, -4, -55, 3])
