"""Framework to build regression model based on geopandas structure."""
import argparse
import collections
import logging
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib
import pandas
import sklearn.metrics
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsCV
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.preprocessing import SplineTransformer
from sklearn.compose import TransformedTargetRegressor

from CustomInteraction import CustomInteraction

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)
logging.getLogger('fiona').setLevel(logging.WARN)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARN)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARN)

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

    fig, ax = plt.subplots(figsize=(20, 10))
    for train_holdback_id, plot_color in [
            (1, 'b'),
            (2, 'g'),
            (3, 'r'),
            (4, 'c'),
            (5, 'm'),
            (6, 'y')]:
        gdf_filtered = gdf[gdf['holdback_id'] == train_holdback_id]
        gdf_filtered.plot(color=plot_color, ax=ax, markersize=2)

    gdf_filtered = gdf[gdf['holdback'].isin([False, 'FALSE'])]
    gdf_filtered.plot(color='k', ax=ax, markersize=1)

    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('Sample points with color coded holdback sets')
    #matplotlib.pyplot.show()
    plt.savefig(os.path.join(FIG_DIR, 'global_sample_points.png'))
    plt.close()

    # load predictor/response table
    predictor_response_table = pandas.read_csv(predictor_response_table_path)
    # drop any not in the base set
    predictor_response_table = predictor_response_table[
        predictor_response_table['predictor'].isin(
            allowed_set.union(set([numpy.nan])))]
    LOGGER.debug(predictor_response_table)
    dataset_map = {}
    fields_to_drop_list = []
    holdback_area_list = []
    for train_holdback_type, train_holdback_val, train_holdback_id in [
            ('train', [False, 'FALSE'], None),
            ('holdback', [True, 'TRUE'], None),
            ('holdback_1', [True, 'TRUE'], 1),
            ('holdback_2', [True, 'TRUE'], 2),
            ('holdback_3', [True, 'TRUE'], 3),
            ('holdback_4', [True, 'TRUE'], 4),
            ('holdback_5', [True, 'TRUE'], 5),
            ('holdback_6', [True, 'TRUE'], 6),
            ]:
        gdf_filtered = gdf[gdf['holdback'].isin(train_holdback_val)]
        if train_holdback_id:
            gdf_filtered = gdf_filtered[
                gdf_filtered['holdback_id'] == train_holdback_id]

        # drop fields that request it
        for index, row in predictor_response_table[~predictor_response_table['include'].isnull()].iterrows():
            column_id = row['predictor']
            if not isinstance(column_id, str):
                column_id = row['response']
            if not isinstance(column_id, str):
                column_id = row['filter']

            LOGGER.info(f'xxxxxxxxxxxxxxxxxxxxxxx {row["filter_only"]}')
            if row['filter_only'] in [1, '1']:
                LOGGER.info(f'******************* dropping {column_id}')
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
                        LOGGER.info(f'xxxxxxxxxxxxxx actively dropped {parameter_id}')
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

        if train_holdback_id is not None:
            holdback_area_list.append(dataset_map[train_holdback_type])

    gdf_filtered.to_csv('gdf_filtered.csv')
    return (
        predictor_response_table['predictor'].count(),
        predictor_response_table['response'].count(),
        id_list_by_parameter_type['predictor'],
        id_list_by_parameter_type['response'],
        dataset_map['train'], dataset_map['holdback'], holdback_area_list,
        rejected_outliers,
        dataset_map['train_params'])


def list_outliers(data, m=100.):
    """List outliers in numpy array within m standard deviations of normal."""
    p99 = numpy.percentile(data, 99)
    p1 = numpy.percentile(data, 1)
    p50 = numpy.median(data)
    # p50 to p99 is 2.32635 sigma
    rSig = (p99-p1)/(2*2.32635)
    return numpy.unique(data[numpy.abs(data - p50) > rSig*m])


def _write_coeficient_table(
        poly_features, predictor_id_list, prefix, name, reg):
    poly_feature_id_list = poly_features.get_feature_names_out(
        predictor_id_list)
    with open(os.path.join(
            FIG_DIR,
            f"{prefix}coef_{name}.csv"), 'w') as table_file:
        intercept = reg[-1].intercept_
        try:
            intercept = intercept[0]
        except Exception:
            pass
        if len(reg) == 3:
            table_file.write('id,coef,scale,mean,term1,term2\n')
            table_file.write(f"intercept,{intercept},1,1,x,x\n")
            for feature_id, coef, scale, mean in zip(poly_feature_id_list, reg[-1].coef_.flatten(), reg[-2].scale_.flatten(), reg[-2].mean_.flatten()):
                if '**2' in feature_id:
                    term1 = feature_id.split('*')[0]
                    term2 = term1
                elif '*' not in feature_id:
                    term1 = feature_id
                    term2 = term1
                else:
                    term1, term2 = feature_id.split('*')
                table_file.write(f"{feature_id.replace(' ', '*')},{coef},{scale},{mean},{term1},{term2}\n")
        else:
            table_file.write('id,coef,pca,scale,mean,\n')
            table_file.write(f"intercept,{intercept}\n")
            for feature_id, coef, pca, scale, mean in zip(poly_feature_id_list, reg[-1].coef_.flatten(), reg[-2].singular_values_, reg[-3].scale_.flatten(), reg[-3].mean_.flatten()):
                table_file.write(f"{feature_id.replace(' ', '*')},{coef},{pca},{scale},{mean}\n")

def regression_results(y_true, y_pred, n, k):
    # Regression metrics
    explained_variance = sklearn.metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = sklearn.metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = sklearn.metrics.median_absolute_error(y_true, y_pred)
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    r2_adjusted = 1-(1-r2)*(n-1)/(n-k-1)


    return r2, r2_adjusted, explained_variance, mean_absolute_error, mse, mean_squared_log_error, median_absolute_error

    # print('explained_variance: ', round(explained_variance,4))
    # print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    # print('r2: ', round(r2,4))
    # print('MAE: ', round(mean_absolute_error,4))
    # print('MSE: ', round(mse,4))
    # print('RMSE: ', round(np.sqrt(mse),4))


def clip_to_range(series, min_val, max_val):
    series[series < min_val] = min_val
    series[series > max_val] = max_val
    return series

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
    parser.add_argument(
        '--interaction_ids', type=str, nargs='+', help=(
            'if selected creates interactions only between these columns and '
            'all other fields, if not selected all fields are interacted with '
            'each other in a second order polynomial'))
    parser.add_argument(
        '--gf_forest_mask_id', type=str,
        help='gaussian filterd forest mask id in table')
    parser.add_argument(
        '--gf_size', type=float, help='gaussian filter size in km')
    parser.add_argument('--polynomial', action='store_true')
    args = parser.parse_args()

    predictor_response_table = pandas.read_csv(args.predictor_response_table)
    allowed_set = set(predictor_response_table['predictor'].dropna())

    # spline_features = SplineTransformer(degree=2, n_knots=3)
    max_iter = 50000
    (n_predictors, n_response, predictor_id_list, response_id_list,
     trainset, holdbackset, holdback_area_list, rejected_outliers,
     parameter_stats) = load_data(
        args.geopandas_data, args.n_rows,
        args.predictor_response_table, allowed_set)

    if args.gf_forest_mask_id not in predictor_id_list:
        LOGGER.warn(f'{args.gf_forest_mask_id} not in {predictor_id_list}')
    LOGGER.info(f'these are the predictors:\n{predictor_id_list}')
    if args.interaction_ids:
        interaction_indexes = [
            predictor_id_list.index(predictor_id)
            for predictor_id in args.interaction_ids]
        poly_features = CustomInteraction(
            interaction_col_indexes=interaction_indexes)
    else:
        if args.polynomial:
            order = POLY_ORDER
        else:
            order = 1
        poly_features = PolynomialFeatures(
            order, interaction_only=False, include_bias=False)

    for name, reg in [
            #('LinearSVR_v2', make_pipeline(poly_features, StandardScaler(), LinearSVR(max_iter=max_iter, loss='epsilon_insensitive', epsilon=1e-4, dual=True))),
            ('LinearSVR_v3', make_pipeline(poly_features, StandardScaler(), LinearSVR(max_iter=max_iter, loss='squared_epsilon_insensitive', epsilon=0, dual=False))),
            #('LassoLarsCV', make_pipeline(poly_features, StandardScaler(),  LassoLarsCV(max_iter=max_iter, cv=10, eps=1e-3, normalize=False))),
            #('LassoLars', make_pipeline(poly_features, StandardScaler(),  LassoLars(alpha=.1, normalize=False, max_iter=max_iter, eps=1e-3))),
            ]:

        LOGGER.info(f'fitting data with {name}')
        kwargs = {
            reg.steps[-1][0] + '__sample_weight': (trainset[1].flatten()/max(trainset[1]))**1
            }
        LOGGER.debug(kwargs)
        LOGGER.debug(f'trainset 0: :::::: {trainset[0]}')
        LOGGER.debug(f'trainset 0: :::::: {trainset[0].shape}')
        LOGGER.debug(trainset[1])
        model = reg.fit(trainset[0], trainset[1], **kwargs)
        model_filename = f'{args.prefix}_{name}_model.dat'
        LOGGER.info(f'saving model to {model_filename}')
        with open(model_filename, 'wb') as model_file:
            model_to_pickle = {
                'model': model,
                'predictor_list': predictor_id_list,
                'gf_forest_id': args.gf_forest_mask_id,
                'gf_size': args.gf_size,
            }
            model_file.write(pickle.dumps(model_to_pickle))

        LOGGER.info(f'saving coefficient table for {name}')
        _write_coeficient_table(
            poly_features, predictor_id_list, args.prefix, name, reg)

        k = trainset[0].shape[1]

        r2_table = open(os.path.join(FIG_DIR, f'r2_summary.csv'), 'a')
        r2_table.write('model,r2,r2_adjusted,explained_variance,mean_absolute_error,mse,mean_squared_log_error,median_absolute_error\n')
        for expected_values, modeled_values, n, prefix in [
                (trainset[1].flatten(), clip_to_range(model.predict(trainset[0]).flatten(), 10, 400), trainset[0].shape[0], 'training'),
                (holdbackset[1].flatten(), clip_to_range(model.predict(holdbackset[0]).flatten(), 10, 400), holdbackset[0].shape[0], 'holdback'),
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
            plt.scatter(expected_values, modeled_values, c='k', s=0.25)
            plt.xlim(0, 400)
            plt.ylim(0, 400)
            #plt.ylim(min(expected_values), max(expected_values))
            #r2 = sklearn.metrics.r2_score(expected_values, modeled_values, multioutput='variance_weighted', force_finite=False)
            #r2_adjusted = 1-(1-r2)*(n-1)/(n-k-1)
            r2, r2_adjusted, explained_variance, mean_absolute_error, mse, mean_squared_log_error, median_absolute_error = regression_results(expected_values, modeled_values, n, k)
            LOGGER.info(f'{args.prefix} {name}-{prefix} adjusted R^2: {r2_adjusted:.3f} r^2: {r2}')
            plt.title(
                f'{args.prefix}{prefix} {name}\n$R^2={r2:.3f}$ -- Adjusted $R^2={r2_adjusted:.3f}$')
            plt.savefig(os.path.join(
                FIG_DIR, f'{args.prefix}{name}_{prefix}.png'))
            plt.close()
            r2_table.write(f'{prefix}_{args.prefix},{r2},{r2_adjusted},{explained_variance},{mean_absolute_error},{mse},{mean_squared_log_error},{median_absolute_error}\n')
            #r2_table.write(f'{prefix}_{args.prefix},{r2},{r2_adjusted}\n')


        plt.xlabel('expected values')
        plt.ylabel('model output')
        plt.title(
            f'Separate Holdback {args.prefix} {name}\n$R^2={r2:.3f}$ -- Adjusted $R^2={r2_adjusted:.3f}$')
        for expected_values, modeled_values, n, prefix, color in [
                    (local_hb_set[1].flatten(), clip_to_range(model.predict(local_hb_set[0]).flatten(), 10, 400), local_hb_set[0].shape[0], f'holdback_{index+1}', color)
                     for index, (local_hb_set, color) in enumerate(zip(holdback_area_list, ['b', 'g', 'r', 'c', 'm', 'y']))]:
            try:
                z = numpy.polyfit(expected_values, modeled_values, 1)
            except ValueError as e:
                # this guards against a poor polyfit line
                print(e)
            trendline_func = numpy.poly1d(z)

            plt.plot(
                expected_values,
                trendline_func(expected_values),
                "r--", linewidth=0.5, c=color)
            plt.scatter(expected_values, modeled_values, c=color, s=0.25)
            #plt.ylim(min(expected_values), max(expected_values))
            r2 = sklearn.metrics.r2_score(expected_values, modeled_values, multioutput='variance_weighted', force_finite=False)
            r2_adjusted = 1-(1-r2)*(n-1)/(n-k-1)
            LOGGER.info(f'{name}-{prefix} adjusted R^2: {r2_adjusted:.3f} r^2: {r2}')
            #r2_table.write(f'{args.prefix}{prefix},{r2},{r2_adjusted}\n')

        plt.xlim(0, 400)
        plt.ylim(0, 400)
        plt.savefig(os.path.join(
            FIG_DIR, f'{args.prefix}{name}_holdback_inidividual.png'))
        plt.close()

        r2_table.close()

    LOGGER.debug('all done')


if __name__ == '__main__':
    main()
