"""Detect outliers in geopandas data."""
import argparse
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)
logging.getLogger('fiona').setLevel(logging.WARN)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARN)

import geopandas
import numpy

def list_outliers(data, m=100.):
    """List outliers in numpy array within m standard deviations of normal."""
    p99 = numpy.percentile(data, 99)
    p1 = numpy.percentile(data, 1)
    p50 = numpy.median(data)
    # p50 to p99 is 2.32635 sigma
    rSig = (p99-p1)/(2*2.32635)
    return numpy.unique(data[numpy.abs(data - p50) > rSig*m])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='outlier test')
    parser.add_argument('geopandas_data', type=str, help=(
        'path to geopandas structure to train on'))
    parser.add_argument('--n_rows', type=int, help='number of rows to load')
    parser.add_argument('--m', type=float, help='n deviations to cutoff')
    args = parser.parse_args()
    gdf = geopandas.read_file(args.geopandas_data, rows=args.n_rows)
    for column_id in gdf.columns:
        if gdf[column_id].dtype in (int, float, complex):
            outliers = list_outliers(gdf[column_id].to_numpy(), m=args.m)
            if len(outliers) > 0:
                LOGGER.debug(f'{column_id}: {outliers}')
                LOGGER.debug(gdf[column_id].isin(outliers))
                gdf[column_id][gdf[column_id].isin(outliers)] = 0
    print(min(gdf['baccini_carbon_data_2003_2014_compressed_md5_11d1455ee8f091bf4be12c4f7ff9451b']))
    print('second pass')
    for column_id in gdf.columns:
        if gdf[column_id].dtype in (int, float, complex):
            outliers = list_outliers(gdf[column_id].to_numpy(), m=args.m)
            if len(outliers) > 0:
                LOGGER.debug(f'{column_id}: {outliers}')
