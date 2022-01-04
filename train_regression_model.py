"""Framework to build regression model based on geopandas structure."""
import argparse
import collections
import logging
import os
import pickle

import matplotlib.pyplot as plt
import pandas
import sklearn.metrics
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.preprocessing import PolynomialFeatures


logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)
logging.getLogger('fiona').setLevel(logging.WARN)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARN)

from osgeo import gdal
from osgeo import osr
from utils import esa_to_carbon_model_landcover_types
import geopandas
import numpy

gdal.SetCacheMax(2**27)

#BOUNDING_BOX = [-64, -4, -55, 3]
BOUNDING_BOX = [-179, -60, 179, 60]
POLY_ORDER = 2

WORKSPACE_DIR = 'workspace'
FIG_DIR = os.path.join(WORKSPACE_DIR, 'fig_dir')
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
ALIGN_DIR = os.path.join(WORKSPACE_DIR, f'align{"_".join([str(v) for v in BOUNDING_BOX])}')
CHURN_DIR = os.path.join(WORKSPACE_DIR, f'churn{"_".join([str(v) for v in BOUNDING_BOX])}')
CHECKPOINT_DIR = 'model_checkpoints'
for dir_path in [
        WORKSPACE_DIR, ECOSHARD_DIR, ALIGN_DIR, CHURN_DIR, CHECKPOINT_DIR, FIG_DIR]:
    os.makedirs(dir_path, exist_ok=True)
RASTER_LOOKUP_PATH = os.path.join(WORKSPACE_DIR, f'raster_lookup{"_".join([str(v) for v in BOUNDING_BOX])}.dat')

URL_PREFIX = (
    'https://storage.googleapis.com/ecoshard-root/global_carbon_regression_2/'
    'inputs/')

RESPONSE_RASTER_FILENAME = 'baccini_carbon_data_2003_2014_compressed_md5_11d1455ee8f091bf4be12c4f7ff9451b.tif'

MASK_TYPES = [
    ('cropland', esa_to_carbon_model_landcover_types.CROPLAND_LULC_CODES),
    ('urban', esa_to_carbon_model_landcover_types.URBAN_LULC_CODES),
    ('forest', esa_to_carbon_model_landcover_types.FOREST_CODES)]

EXPECTED_MAX_EDGE_EFFECT_KM_LIST = [0.5]  # changed given guidance from reviewer [1.0, 3.0, 10.0]

CELL_SIZE = (0.004, -0.004)  # in degrees
PROJECTION_WKT = osr.SRS_WKT_WGS84_LAT_LONG
SAMPLE_RATE = 0.001

MAX_TIME_INDEX = 11


def load_data(
        geopandas_data, invalid_values, n_rows, predictor_response_table_path,
        allowed_set):
    """
    Load and process data from geopandas data structure.

    Args:
        geopandas_data (str): path to geopandas file containing at least
            the fields defined in the predictor response table and a
            "holdback" field to indicate the test data.
        invalid_values (list): list of (fieldname, value) tuples to
            invalidate any fieldname entries that have that value.
        n_rows (int): number of rows to load.
        predictor_response_table_path (str): path to a csv file containing
            headers 'predictor' and 'response'. Any non-null values
            underneath these headers are used for predictor and response
            variables.
        allowed_set (set): if predictor in this set, allow it in the data
            otherwise skip

    Return:
        pytorch dataset tuple of (train, test) DataSets.
    """
    # load data
    #LOGGER.info(f'reading geopandas file {geopandas_data}')
    if geopandas_data.endswith('gpkg'):
        gdf = geopandas.read_file(geopandas_data)
    else:
        with open(geopandas_data, 'rb') as geopandas_file:
            gdf = pickle.load(geopandas_file).copy()
    if n_rows is not None:
        gdf = gdf.sample(n=n_rows, replace=False).copy()
    #LOGGER.info(f'gdf columns: {gdf.columns}')
    for invalid_value_tuple in invalid_values:
        key, value = invalid_value_tuple.split(',')
        gdf.drop(gdf[key] != float(value), inplace=True)

    rejected_outliers = {}
    gdf.to_csv('dropped_base.csv')
    for column_id in gdf.columns:
        if gdf[column_id].dtype in (int, float, complex):
            outliers = list_outliers(gdf[column_id].to_numpy())
            if len(outliers) > 0:
                #LOGGER.debug(f'{column_id}: {outliers}')
                gdf.replace({column_id: outliers}, 0, inplace=True)
                #gdf.loc[column_id, gdf[column_id].isin(outliers)] = 0
                rejected_outliers[column_id] = outliers
    gdf.to_csv('dropped.csv')

    # load predictor/response table
    predictor_response_table = pandas.read_csv(predictor_response_table_path)
    # drop any not in the base set
    predictor_response_table = predictor_response_table[
        predictor_response_table['predictor'].isin(
            allowed_set.union(set([numpy.nan])))]
    dataset_map = {}
    fields_to_drop_list = []
    for train_holdback_type, train_holdback_val in [
            ('holdback', [True, 'TRUE']), ('train', [False, 'FALSE'])]:
        gdf_filtered = gdf[gdf['holdback'].isin(train_holdback_val)]

        # drop fields that request it
        for index, row in predictor_response_table[~predictor_response_table['include'].isnull()].iterrows():
            LOGGER.debug(column_id)
            column_id = row['predictor']
            if not isinstance(column_id, str):
                column_id = row['response']
            if not isinstance(column_id, str):
                column_id = row['filter']

            if row['filter_only'] == 1:
                fields_to_drop_list.append(column_id)

        # restrict based on "include"
        index_filter_series = None
        for index, row in predictor_response_table[~predictor_response_table['include'].isnull()].iterrows():
            LOGGER.debug(column_id)
            column_id = row['predictor']
            if not isinstance(column_id, str):
                column_id = row['response']
            if not isinstance(column_id, str):
                column_id = row['filter']
            #LOGGER.debug(f'column id to filter: {column_id}')
            #LOGGER.debug(f"going to keep value {row['include']} from column {column_id}")

            keep_indexes = (gdf_filtered[column_id]==float(row['include']))
            if index_filter_series is None:
                index_filter_series = keep_indexes
            else:
                index_filter_series &= keep_indexes

            #LOGGER.debug(index_filter_series)

        # restrict based on "exclude"
        for index, row in predictor_response_table[~predictor_response_table['exclude'].isnull()].iterrows():
            column_id = row['predictor']
            if not isinstance(column_id, str):
                column_id = row['response']
            if not isinstance(column_id, str):
                column_id = row['filter']
            #LOGGER.debug(f'column id to filter: {column_id}')
            #LOGGER.debug(f"going to drop value {row['exclude']} from column {column_id}")
            #LOGGER.debug(f"row['predictor']: {row['predictor']} row['response']: {row['response']} ")

            keep_indexes = (gdf_filtered[column_id]!=float(row['exclude']))
            if index_filter_series is None:
                index_filter_series = keep_indexes
            else:
                index_filter_series &= keep_indexes

            LOGGER.debug(index_filter_series)

        # restrict based on min/max
        if 'max' in predictor_response_table:
            for index, row in predictor_response_table[~predictor_response_table['max'].isnull()].iterrows():
                column_id = row['predictor']
                if not isinstance(column_id, str):
                    column_id = row['response']
                if not isinstance(column_id, str):
                    column_id = row['filter']
                #LOGGER.debug(f'column id to filter: {column_id}')
                #LOGGER.debug(f"going to drop value > {row['max']} from column {column_id}")
                #LOGGER.debug(f"row['predictor']: {row['predictor']} row['response']: {row['response']} ")

                keep_indexes = (gdf_filtered[column_id] <= float(row['max']))
                if index_filter_series is None:
                    index_filter_series = keep_indexes
                else:
                    index_filter_series &= keep_indexes



                #LOGGER.debug(index_filter_series)

        # restrict based on min/max
        if 'min' in predictor_response_table:
            for index, row in predictor_response_table[~predictor_response_table['min'].isnull()].iterrows():
                column_id = row['predictor']
                if not isinstance(column_id, str):
                    column_id = row['response']
                if not isinstance(column_id, str):
                    column_id = row['filter']
                # skip if not allowed
                #LOGGER.debug(f'column id to filter: {column_id}')
                #LOGGER.debug(f"going to drop value < {row['min']} from column {column_id}")
                #LOGGER.debug(f"row['predictor']: {row['predictor']} row['response']: {row['response']} ")

                keep_indexes = (gdf_filtered[column_id] >= float(row['min']))
                if index_filter_series is None:
                    index_filter_series = keep_indexes
                else:
                    index_filter_series &= keep_indexes

                #LOGGER.debug(index_filter_series)

        #LOGGER.debug(index_filter_series)

        if index_filter_series is not None:
            gdf_filtered = gdf_filtered[index_filter_series]

        if 'group' in predictor_response_table:
            unique_groups = predictor_response_table['group'].dropna().unique()
        #else:
        #    unique_groups = [numpy.nan]

        #for field_to_drop in fields_to_drop_list:
        #    gdf_filtered.drop(field_to_drop, axis=1, inplace=True)

        # gdf filtered is the table with data indexed by the raw parameter
        # names that will show up in the predictor response table
        # ['predictor'] or ['response'] columns
        #LOGGER.debug(gdf_filtered.columns)
        # predictor response table is the model parameter table indexed by
        #   predictor, response, include, exclude, min, max, group, target
        #LOGGER.debug(predictor_response_table['predictor'])
        #LOGGER.debug(predictor_response_table['response'])
        parameter_stats = {}
        group_collection = collections.defaultdict(lambda: collections.defaultdict(list))
        index = 0
        for group_id in unique_groups:
            # this will collect the correct target ids for the resulting
            # grouped variables, every ID list should be the same for each
            # group id
            id_list_by_parameter_type = collections.defaultdict(list)
            # set up one group run, this map will collect vector of predictor
            # and response for training for that group
            predictor_response_map = collections.defaultdict(list)
            for parameter_type in ['predictor', 'response']:
                for parameter_id, predictor_response_group_id, target_id in \
                        zip(predictor_response_table[parameter_type],
                            predictor_response_table['group'],
                            predictor_response_table['target']):
                    if parameter_id in fields_to_drop_list:
                        continue
                    # this loop gets at a particular parameter
                    # (crop, slope, etc)
                    if not isinstance(parameter_id, str):
                        # parameter might not be a predictor or a response
                        # (n/a in the model table column)
                        continue

                    if not numpy.isnan(predictor_response_group_id):
                        # this predictor class has a group defined with it,
                        # use it if the group matches the current group id
                        if predictor_response_group_id != group_id:
                            continue
                    else:
                        # if the predictor response is not defined then it's
                        # used in every group
                        target_id = parameter_id

                    # LOGGER.debug(
                    #    f'{train_holdback_type} {parameter_type} {group_id} {parameter_id}: '
                    #    f'{target_id}')
                    # if this parameter id has groups, it also has a target,
                    # we want to uniquely stack the n groups
                    # if there is no group then we can duplicate the stack n times

                    # gdf_filtered only has raw data
                    # the predictor_response table tells us how the fields in
                    # gdf_filtered group to a higher concept
                    if isinstance(parameter_id, str):
                        id_list_by_parameter_type[parameter_type].append(
                            target_id)
                        if parameter_id == 'geometry.x':
                            predictor_response_map[parameter_type].append(
                                gdf_filtered['geometry'].x)
                        elif parameter_id == 'geometry.y':
                            predictor_response_map[parameter_type].append(
                                gdf_filtered['geometry'].y)
                        else:
                            predictor_response_map[parameter_type].append(
                                gdf_filtered[parameter_id])
                        if parameter_type == 'predictor':
                            parameter_stats[(index, target_id)] = (
                                gdf_filtered[parameter_id].mean(),
                                gdf_filtered[parameter_id].std())
                            index += 1

                group_collection[group_id] = (
                    predictor_response_map, id_list_by_parameter_type)
        # group_collection is sorted by group
        x_tensor = None
        for key, (parameters, id_list) in group_collection.items():
            #LOGGER.debug(
            #    f"{key}:\n{id_list['predictor'][0]}\n"
            #    f"{(parameters['predictor'][0])}")

            local_x_tensor = numpy.array(
                predictor_response_map['predictor'], dtype=numpy.float32)
            local_y_tensor = numpy.array(
                predictor_response_map['response'], dtype=numpy.float32)
            if x_tensor is None:
                x_tensor = local_x_tensor
                y_tensor = local_y_tensor
            else:
                x_tensor = numpy.concatenate(
                    (x_tensor, local_x_tensor), axis=1)
                y_tensor = numpy.concatenate(
                    (y_tensor, local_y_tensor), axis=1)
            #LOGGER.debug(x_tensor.shape)
        #LOGGER.debug(x_tensor.shape)
        #LOGGER.debug(y_tensor.shape)
        dataset_map[train_holdback_type] = (x_tensor.T, y_tensor.T)
        dataset_map[f'{train_holdback_type}_params'] = parameter_stats

    #gdf_filtered.to_csv('gdf_filtered.csv')
    return (
        predictor_response_table['predictor'].count(),
        predictor_response_table['response'].count(),
        id_list_by_parameter_type['predictor'],
        id_list_by_parameter_type['response'],
        dataset_map['train'], dataset_map['holdback'], rejected_outliers,
        dataset_map['train_params'])


def list_outliers(data, m=100.):
    """List outliers in numpy array within m standard deviations of normal."""
    p99 = numpy.percentile(data, 99)
    p1 = numpy.percentile(data, 1)
    p50 = numpy.median(data)
    # p50 to p99 is 2.32635 sigma
    rSig = (p99-p1)/(2*2.32635)
    return numpy.unique(data[numpy.abs(data - p50) > rSig*m])


def r2_analysis(
        geopandas_data_path, invalid_values, n_rows,
        predictor_response_table_path, allowed_set, reg):
    """Calculate adjusted R2 given the allowed set."""
    (n_predictors, n_response, predictor_id_list, response_id_list,
     trainset, testset, rejected_outliers,
     parameter_stats) = load_data(
        geopandas_data_path, invalid_values, n_rows,
        predictor_response_table_path, allowed_set)
    LOGGER.info(f'got {n_predictors} predictors, doing fit')
    LOGGER.info(f'these are the predictors:\n{predictor_id_list}')
    model = reg.fit(trainset[0], trainset[1])
    expected_values = trainset[1].flatten()
    LOGGER.info('fit complete, calculate r2')
    modeled_values = model.predict(trainset[0]).flatten()

    r2 = sklearn.metrics.r2_score(expected_values, modeled_values)
    k = trainset[0].shape[1]
    n = trainset[0].shape[0]
    r2_adjusted = 1-(1-r2)*(n-1)/(n-k-1)
    return r2_adjusted, reg, predictor_id_list


def _write_coeficient_table(poly_features, predictor_id_list, prefix, name, reg):
    poly_feature_id_list = poly_features.get_feature_names_out(
        predictor_id_list)
    with open(os.path.join(
            f"{prefix}coef_{name}.csv"), 'w') as table_file:
        table_file.write('id,coef\n')
        for feature_id, coef in zip(poly_feature_id_list, reg[-1].coef_.flatten()):
            table_file.write(f"{feature_id.replace(' ', '*')},{coef}\n")

def main():
    parser = argparse.ArgumentParser(description='DNN model trainer')
    parser.add_argument('geopandas_data', type=str, help=(
        'path to geopandas structure to train on'))
    parser.add_argument('predictor_response_table', type=str, help=(
        'path to csv table with fields "predictor" and "response", the '
        'fieldnames underneath are used to sample the geopandas datastructure '
        'for training'))
    parser.add_argument(
        '--n_rows', type=int,
        help='number of samples to train on from the dataset')
    parser.add_argument(
        '--invalid_values', type=str, nargs='*', default=list(),
        help='values to mask out of dataframe write as fieldname,value pairs')
    parser.add_argument(
        '--prefix', type=str, default='', help='add prefix to output files')
    args = parser.parse_args()

    predictor_response_table = pandas.read_csv(args.predictor_response_table)
    allowed_set = set(predictor_response_table['predictor'].dropna())

    poly_features = PolynomialFeatures(POLY_ORDER, interaction_only=False)
    max_iter = 50000
    # lasso_reg = make_pipeline(
    #     poly_features, StandardScaler(), linear_model.Lasso(
    #         alpha=0.1, max_iter=max_iter))
    # LOGGER.info('doing initial r2 analysis')
    # last_adjusted_r2, reg, predictor_id_list = r2_analysis(
    #     args.geopandas_data, args.invalid_values, args.n_rows,
    #     args.predictor_response_table, allowed_set, lasso_reg)
    # print(last_adjusted_r2)
    # _write_coeficient_table(
    #     poly_features, predictor_id_list, args.prefix, os.path.basename(
    #         os.path.splitext(args.predictor_response_table)[0]), reg)
    # return

    # LOGGER.debug(last_adjusted_r2)
    # pool = multiprocessing.pool.Pool(3)

    # while True:
    #     result_list = []
    #     for predictor_id in reversed(sorted(allowed_set)):
    #         local_set = set(allowed_set)
    #         local_set.remove(predictor_id)
    #         #LOGGER.debug(predictor_id)
    #         result = pool.apply_async(
    #             r2_analysis, (
    #                 args.geopandas_data, args.invalid_values, args.n_rows,
    #                 args.predictor_response_table, local_set, lasso_reg))
    #         result_list.append((predictor_id, result))

    #         #r2_adjusted = r2_analysis(
    #         #    args.geopandas_data, args.invalid_values, args.n_rows,
    #         #    args.predictor_response_table, local_set, lasso_reg)
    #     pool.close()
    #     pool.join()
    #     r2_by_predictor_list = []
    #     for predictor_id, result in result_list:
    #         LOGGER.debug(f'waiting on {predictor_id}')
    #         adjusted_r2 = result.get()
    #         LOGGER.info(
    #             f'got {adjusted_r2} for {predictor_id} increase of '
    #             f'{adjusted_r2-last_adjusted_r2}')
    #         r2_by_predictor_list.append((adjusted_r2, predictor_id))
    #     LOGGER.debug(f'previous r2: {last_adjusted_r2}')
    #     LOGGER.debug('\n'.join([
    #         f'{predictor_id} (new {r2} vs previous {last_adjusted_r2}): increase {last_adjusted_r2-r2}' for r2, predictor_id in
    #         sorted(r2_by_predictor_list)
    #         if last_adjusted_r2-r2 > 0]))
    # #   return

    (n_predictors, n_response, predictor_id_list, response_id_list,
     trainset, testset, rejected_outliers, parameter_stats) = load_data(
        args.geopandas_data, args.invalid_values, args.n_rows,
        args.predictor_response_table, allowed_set)


    #LOGGER.debug(parameter_stats)
    #mean_vector = []
    #sample_vector = trainset[:][0][0].detach().numpy()
    #predictor_vector = trainset[0]
    #response_vector = trainset[1]
    # for (parameter_index, parameter_id), (mean, std) in sorted(
    #         parameter_stats.items()):
    #     LOGGER.debug(
    #         f'{parameter_index}, {parameter_id}, {mean}, {std}, '
    #         f'{predictor_vector[parameter_index]}')
    #     mean_vector.append(mean)

    #LOGGER.debug(predictor_vector.shape)

    # sensitivity_test = []
    # # seed with a 'random' sample and the mean
    # parameter_id_list = ['random sample', 'training set mean']
    # sensitivity_test.append(trainset[0][:, 0])
    # sensitivity_test.append(numpy.array(mean_vector))
    # for (index, parameter_id), (mean, std) in sorted(parameter_stats.items()):

    #     LOGGER.debug(f'{index}, {parameter_id}, {mean}, {std}')
    #     local_sample_vector = numpy.array(sensitivity_test[-1])
    #     parameter_id_list.append(f'{parameter_id}_sample-std')
    #     local_sample_vector[index] -= std
    #     sensitivity_test.append(numpy.array(local_sample_vector))

    #     local_sample_vector = numpy.array(sensitivity_test[-1])
    #     parameter_id_list.append(f'{parameter_id}_sample+std')
    #     local_sample_vector[index] += std
    #     sensitivity_test.append(numpy.array(local_sample_vector))

    #     sensitivity_vector = numpy.array(mean_vector)
    #     parameter_id_list.append(f'{parameter_id}_mean-std')
    #     sensitivity_vector[index] = mean-std
    #     sensitivity_test.append(numpy.array(sensitivity_vector))

    #     sensitivity_vector = numpy.array(mean_vector)
    #     parameter_id_list.append(f'{parameter_id}_mean+std')
    #     sensitivity_vector[index] = mean+std
    #     sensitivity_test.append(numpy.array(sensitivity_vector))

    #sensitivity_test = numpy.array(sensitivity_test)
    #LOGGER.debug(sensitivity_test.shape)
    # mean vector is a vector of mean values for the parameters

    LOGGER.debug(f'predictor id list: {predictor_id_list}')
    LOGGER.debug(f'response id list: {response_id_list}')

    for name, reg in [
            #('ols', make_pipeline(poly_features, StandardScaler(), linear_model.LinearRegression())),
            ('svm', make_pipeline(poly_features, StandardScaler(), LinearSVR(max_iter=max_iter, loss='squared_epsilon_insensitive', dual=False))),
            ('lasso', make_pipeline(poly_features, StandardScaler(), linear_model.Lasso(alpha=0.1, max_iter=max_iter))),
            ('lasso lars', make_pipeline(poly_features, StandardScaler(), linear_model.LassoLars(alpha=.1, normalize=False, max_iter=max_iter))),
            ]:
        LOGGER.debug(f'{name}: {trainset[0].shape}, {trainset[1].shape}')
        model = reg.fit(trainset[0], trainset[1])
        # sensitivity_result = model.predict(sensitivity_test)
        # # result is pairs of 4 - mean-std for parameter, mean+std, sample-std, sample+std
        # with open(f'sensitivity_{name}.csv', 'w') as sensitivity_table:
        #     sensitivity_table.write('parameter,result,val-sample,val-mean\n')
        #     # do the mean first
        #     for index, val in enumerate(sensitivity_result):
        #         if isinstance(val, numpy.ndarray):
        #             val = val[0]
        #         if index == 0:
        #             sample = val
        #         if index == 1:
        #             mean = val
        #         if index <= 1:
        #             sensitivity_table.write(f'{parameter_id_list[index]},{val},n/a,n/a\n')
        #         elif (index-2) % 4 < 2:
        #             # sample row
        #             sensitivity_table.write(f'{parameter_id_list[index]},{val},{val-sample},n/a\n')
        #         else:
        #             # mean row
        #             sensitivity_table.write(f'{parameter_id_list[index]},{val},n/a,{val-mean}\n')
        _write_coeficient_table(
            poly_features, predictor_id_list, args.prefix, name, reg)

        k = trainset[0].shape[1]
        for expected_values, modeled_values, n, prefix in [
                (testset[1].flatten(), model.predict(testset[0]).flatten(), testset[0].shape[0], 'holdback'),
                (trainset[1].flatten(), model.predict(trainset[0]).flatten(), trainset[0].shape[0], 'training'),
                ]:

            try:
                z = numpy.polyfit(expected_values, modeled_values, 1)
            except ValueError as e:
                # this guards against a poor polyfit line
                print(e)
            trendline_func = numpy.poly1d(z)
            plt.xlabel('expected values')
            plt.ylabel('model output')
            plt.plot(
                expected_values,
                trendline_func(expected_values),
                "r--", linewidth=1.5)
            plt.scatter(expected_values, modeled_values, c='g', s=0.25)
            plt.ylim(
                min(expected_values), max(expected_values))
            r2 = sklearn.metrics.r2_score(expected_values, modeled_values)
            r2_adjusted = 1-(1-r2)*(n-1)/(n-k-1)
            plt.title(
                f'{args.prefix}{prefix} {name}\n$R^2={r2:.3f}$\nAdjusted $R^2={r2_adjusted:.3f}$')
            plt.savefig(os.path.join(
                FIG_DIR, f'{args.prefix}{name}_{prefix}.png'))
            plt.close()

        model_structure = {
            'model': model,
            'predictor_id_list': predictor_id_list,
        }

    # start with a list of predictors and response
    # train model and record adjusted R^2
    # loop through each predictor to drop it and compare against R^2
    # drop the predictor that gives the largest R^2 increase to the base, report to .txt, then iterate
    predictor_set = set()
    previous_r_2 = 0.0
    #while True:


    LOGGER.debug('all done')


if __name__ == '__main__':
    main()
