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
from sklearn.decomposition import PCA
from sklearn.preprocessing import SplineTransformer

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
import geopandas
import numpy

gdal.SetCacheMax(2**27)

#BOUNDING_BOX = [-64, -4, -55, 3]
BOUNDING_BOX = [-179, -60, 179, 60]
POLY_ORDER = 2

FIG_DIR = os.path.join('fig_dir')
CHECKPOINT_DIR = 'model_checkpoints'
for dir_path in [
        FIG_DIR]:
    os.makedirs(dir_path, exist_ok=True)


def load_data(
        geopandas_data, n_rows, predictor_response_table_path,
        allowed_set):
    """
    Load and process data from geopandas data structure.

    Args:
        geopandas_data (str): path to geopandas file containing at least
            the fields defined in the predictor response table and a
            "holdback" field to indicate the test data.
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
    if geopandas_data.endswith('gpkg'):
        gdf = geopandas.read_file(geopandas_data)
    else:
        with open(geopandas_data, 'rb') as geopandas_file:
            gdf = pickle.load(geopandas_file).copy()

    rejected_outliers = {}
    gdf.to_csv('dropped_base.csv')
    for column_id in gdf.columns:
        if gdf[column_id].dtype in (int, float, complex):
            outliers = list_outliers(gdf[column_id].to_numpy())
            if len(outliers) > 0:
                gdf.replace({column_id: outliers}, 0, inplace=True)
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
            column_id = row['predictor']
            if not isinstance(column_id, str):
                column_id = row['response']
            if not isinstance(column_id, str):
                column_id = row['filter']

            keep_indexes = (gdf_filtered[column_id]==float(row['include']))
            if index_filter_series is None:
                index_filter_series = keep_indexes
            else:
                index_filter_series &= keep_indexes

        # restrict based on "exclude"
        for index, row in predictor_response_table[~predictor_response_table['exclude'].isnull()].iterrows():
            column_id = row['predictor']
            if not isinstance(column_id, str):
                column_id = row['response']
            if not isinstance(column_id, str):
                column_id = row['filter']

            keep_indexes = (gdf_filtered[column_id]!=float(row['exclude']))
            if index_filter_series is None:
                index_filter_series = keep_indexes
            else:
                index_filter_series &= keep_indexes

        # restrict based on min/max
        if 'max' in predictor_response_table:
            for index, row in predictor_response_table[~predictor_response_table['max'].isnull()].iterrows():
                column_id = row['predictor']
                if not isinstance(column_id, str):
                    column_id = row['response']
                if not isinstance(column_id, str):
                    column_id = row['filter']

                keep_indexes = (gdf_filtered[column_id] <= float(row['max']))
                if index_filter_series is None:
                    index_filter_series = keep_indexes
                else:
                    index_filter_series &= keep_indexes

        # restrict based on min/max
        if 'min' in predictor_response_table:
            for index, row in predictor_response_table[~predictor_response_table['min'].isnull()].iterrows():
                column_id = row['predictor']
                if not isinstance(column_id, str):
                    column_id = row['response']
                if not isinstance(column_id, str):
                    column_id = row['filter']

                keep_indexes = (gdf_filtered[column_id] >= float(row['min']))
                if index_filter_series is None:
                    index_filter_series = keep_indexes
                else:
                    index_filter_series &= keep_indexes

        if index_filter_series is not None:
            gdf_filtered = gdf_filtered[index_filter_series]

        if 'group' in predictor_response_table:
            unique_groups = predictor_response_table['group'].dropna().unique()
        if unique_groups.size == 0:
            unique_groups = [numpy.nan]

        parameter_stats = {}
        group_collection = collections.defaultdict(
            lambda: collections.defaultdict(list))
        index = 0
        for group_id in unique_groups:
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
                    if (isinstance(predictor_response_group_id, str) or
                            not numpy.isnan(predictor_response_group_id)):
                        # this predictor class has a group defined with it,
                        # use it if the group matches the current group id
                        if predictor_response_group_id != group_id:
                            continue
                    else:
                        # if the predictor response is not defined then it's
                        # used in every group
                        target_id = parameter_id

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
        dataset_map[train_holdback_type] = (x_tensor.T, y_tensor.T)
        dataset_map[f'{train_holdback_type}_params'] = parameter_stats

    gdf_filtered.to_csv('gdf_filtered.csv')
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
        geopandas_data_path, n_rows,
        predictor_response_table_path, allowed_set, reg):
    """Calculate adjusted R2 given the allowed set."""
    (n_predictors, n_response, predictor_id_list, response_id_list,
     trainset, testset, rejected_outliers,
     parameter_stats) = load_data(
        geopandas_data_path, n_rows,
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
        table_file.write('id,coef,pca,scale,mean,\n')
        for feature_id, coef, pca, scale, mean in zip(poly_feature_id_list, reg[-1].coef_.flatten(), reg[-2].singular_values_, reg[-3].scale_.flatten(), reg[-3].mean_.flatten()):
            table_file.write(f"{feature_id.replace(' ', '*')},{coef},{pca},{scale},{mean}\n")


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
        '--prefix', type=str, default='', help='add prefix to output files')
    args = parser.parse_args()

    predictor_response_table = pandas.read_csv(args.predictor_response_table)
    allowed_set = set(predictor_response_table['predictor'].dropna())

    poly_features = PolynomialFeatures(
        POLY_ORDER, interaction_only=False, include_bias=False)
    spline_features = SplineTransformer(degree=2, n_knots=3)
    max_iter = 50000
    (n_predictors, n_response, predictor_id_list, response_id_list,
     trainset, testset, rejected_outliers, parameter_stats) = load_data(
        args.geopandas_data, args.n_rows,
        args.predictor_response_table, allowed_set)
    n_components = int(n_predictors**2*.3)
    n_components = n_predictors**2
    #n_components = n_predictors
    for name, reg in [
            #('ols', make_pipeline(poly_features, StandardScaler(), linear_model.LinearRegression())),
            #('svm', make_pipeline(poly_features, StandardScaler(), LinearSVR(max_iter=max_iter, loss='squared_epsilon_insensitive', dual=False))),
            #('lasso', make_pipeline(poly_features, StandardScaler(), linear_model.Lasso(alpha=0.1, max_iter=max_iter))),
            #('lasso lars', make_pipeline(poly_features, StandardScaler(), linear_model.LassoLars(alpha=.1, normalize=False, max_iter=max_iter))),
            # ('LinearSVR', make_pipeline(spline_features, StandardScaler(), PCA(n_components=n_components), LinearSVR(max_iter=max_iter, loss='squared_epsilon_insensitive', epsilon=1e-3, dual=False))),
            # ('LassoLarsCV', make_pipeline(spline_features, StandardScaler(), PCA(n_components=n_components), linear_model.LassoLarsCV(max_iter=max_iter, cv=10, eps=1e-3, normalize=False))),
            # ('LassoLars', make_pipeline(spline_features, StandardScaler(), PCA(n_components=n_components), linear_model.LassoLars(alpha=.1, normalize=False, max_iter=max_iter, eps=1e-3))),
            ('LinearSVR', make_pipeline(poly_features, PCA(n_components=n_components), StandardScaler(), LinearSVR(max_iter=max_iter, loss='squared_epsilon_insensitive', epsilon=1e-3, dual=False))),
            ('LassoLarsCV', make_pipeline(poly_features, PCA(n_components=n_components), StandardScaler(), linear_model.LassoLarsCV(max_iter=max_iter, cv=10, eps=1e-3, normalize=False))),
            ('LassoLars', make_pipeline(poly_features, PCA(n_components=n_components), StandardScaler(), linear_model.LassoLars(alpha=.1, normalize=False, max_iter=max_iter, eps=1e-3))),
            ]:
        LOGGER.info(f'fitting data with {name}')
        model = reg.fit(trainset[0], trainset[1])
        model_filename = f'{name}_model.dat'
        LOGGER.info(f'saving model to {model_filename}')
        with open(model_filename, 'wb') as model_file:
            model_to_pickle = {
                'model': model,
                'predictor_list': predictor_id_list
            }
            model_file.write(pickle.dumps(model_to_pickle))

        LOGGER.info(f'saving coefficient table for {name}')
        #_write_coeficient_table(
        #    poly_features, predictor_id_list, args.prefix, name, reg)

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
            if prefix == 'holdback':
                LOGGER.info(f'{name}-{prefix} adjusted R^2: {r2_adjusted:.3f}')
            plt.title(
                f'{args.prefix}{prefix} {name}\n$R^2={r2:.3f}$ -- Adjusted $R^2={r2_adjusted:.3f}$')
            plt.savefig(os.path.join(
                FIG_DIR, f'{args.prefix}{name}_{prefix}.png'))
            plt.close()

        model_structure = {
            'model': model,
            'predictor_id_list': predictor_id_list,
        }

    LOGGER.debug('all done')


if __name__ == '__main__':
    main()
