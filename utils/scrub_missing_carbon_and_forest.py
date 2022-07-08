"""Create a skeleton model training CSV config."""
import argparse
import os

import geopandas


def main():
    parser = argparse.ArgumentParser(description='Carbon Model Scrub')
    parser.add_argument('geopandas_data_path', type=str, help=(
        'path to geopandas structure to scrub'))
    args = parser.parse_args()
    gdf = geopandas.read_file(args.geopandas_data_path)

    cog_cols = [col for col in gdf.columns if col.startswith('cog_')]
    print('scrubbing')
    gdf = gdf.drop(columns=cog_cols)
    gdf = gdf[gdf.baccini_carbon_data_2014_compressed <= 400]
    gdf = gdf[gdf.baccini_carbon_data_2014_compressed > 0]
    #gdf = gdf[gdf['masked_forest_ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7'] == 1]

    gdf = gdf.loc[(gdf[[col for col in gdf.columns if col != 'holdback']] != 0).all(axis=1)]
    gdf['lon'] = gdf.centroid.x
    gdf['lat'] = gdf.centroid.y

    print('saving')

    gdf.to_file(
        '%s_scrubbed%s' % os.path.splitext(args.geopandas_data_path),
        driver='GPKG')
    print('done')

if __name__ == '__main__':
    main()
