"""Script to download everything needed to train the models."""
from datetime import datetime
import argparse
import collections
import functools
import logging
import math
import multiprocessing
import os
import pickle
import threading
import time

import matplotlib.pyplot as plt
import pandas
import ray
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import sklearn.metrics

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
from osgeo import ogr
from utils import esa_to_carbon_model_landcover_types
import ecoshard
import pygeoprocessing
import geopandas
import numpy
import scipy
import taskgraph
import torch
from sklearn.model_selection import train_test_split
torch.autograd.set_detect_anomaly(True)

gdal.SetCacheMax(2**27)
torch.set_num_threads(multiprocessing.cpu_count())
torch.set_num_interop_threads(multiprocessing.cpu_count())


#BOUNDING_BOX = [-64, -4, -55, 3]
BOUNDING_BOX = [-179, -60, 179, 60]

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


class NeuralNetwork(torch.nn.Module):
    """Flexible neural net."""
    def __init__(self, M, l1, l2):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(M, int(l1)),
            torch.nn.Linear(int(l1), int(l2)),
            torch.nn.Linear(int(l2), 1),
            #torch.nn.Dropout(p=0.1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def model_predict(
            model, lulc_raster_path, forest_mask_raster_path,
            aligned_predictor_list, predicted_biomass_raster_path):
    """Predict biomass given predictors."""
    pygeoprocessing.new_raster_from_base(
        lulc_raster_path, predicted_biomass_raster_path, gdal.GDT_Float32,
        [-1])
    predicted_biomass_raster = gdal.OpenEx(
        predicted_biomass_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    predicted_biomass_band = predicted_biomass_raster.GetRasterBand(1)

    predictor_band_nodata_list = []
    raster_list = []
    # simple lookup to map predictor band/nodata to a list
    for predictor_path, nodata in aligned_predictor_list:
        predictor_raster = gdal.OpenEx(predictor_path, gdal.OF_RASTER)
        raster_list.append(predictor_raster)
        predictor_band = predictor_raster.GetRasterBand(1)

        if nodata is None:
            nodata = predictor_band.GetNoDataValue()
        predictor_band_nodata_list.append((predictor_band, nodata))
    forest_raster = gdal.OpenEx(forest_mask_raster_path, gdal.OF_RASTER)
    forest_band = forest_raster.GetRasterBand(1)

    for offset_dict in pygeoprocessing.iterblocks(
            (lulc_raster_path, 1), offset_only=True):
        forest_array = forest_band.ReadAsArray(**offset_dict)
        valid_mask = (forest_array == 1)
        x_vector = None
        array_list = []
        for band, nodata in predictor_band_nodata_list:
            array = band.ReadAsArray(**offset_dict)
            if nodata is None:
                nodata = band.GetNoDataValue()
            if nodata is not None:
                valid_mask &= array != nodata
            array_list.append(array)
        if not numpy.any(valid_mask):
            continue
        for array in array_list:
            if x_vector is None:
                x_vector = array[valid_mask].astype(numpy.float32)
                x_vector = numpy.reshape(x_vector, (-1, x_vector.size))
            else:
                valid_array = array[valid_mask].astype(numpy.float32)
                valid_array = numpy.reshape(valid_array, (-1, valid_array.size))
                x_vector = numpy.append(x_vector, valid_array, axis=0)
        y_vector = model(torch.from_numpy(x_vector.T))
        result = numpy.full(forest_array.shape, -1)
        result[valid_mask] = (y_vector.detach().numpy()).flatten()
        predicted_biomass_band.WriteArray(
            result,
            xoff=offset_dict['xoff'],
            yoff=offset_dict['yoff'])
    predicted_biomass_band = None
    predicted_biomass_raster = None


def init_weights(m):
    if type(m) == torch.nn.Linear:
        #m.weight.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)


def load_data(
        geopandas_data, invalid_values, n_rows, m_std_reject, predictor_response_table):
    """
    Load and process data from geopandas data structure.

    Args:
        geopandas_data (str): path to geopandas file containing at least
            the fields defined in the predictor response table and a
            "holdback" field to indicate the test data.
        invalid_values (list): list of (fieldname, value) tuples to
            invalidate any fieldname entries that have that value.
        n_rows (int): number of rows to load.
        m_std_reject (float): number of standard deviations to reject as
            an outlier (set to 0).
        predictor_response_table (str): path to a csv file containing
            headers 'predictor' and 'response'. Any non-null values
            underneath these headers are used for predictor and response
            variables.

    Return:
        pytorch dataset tuple of (train, test) DataSets.
    """
    # load data
    LOGGER.info(f'reading geopandas file {geopandas_data}')
    gdf = geopandas.read_file(geopandas_data, rows=n_rows)
    LOGGER.info(f'gdf columns: {gdf.columns}')
    for invalid_value_tuple in invalid_values:
        key, value = invalid_value_tuple.split(',')
        gdf = gdf[gdf[key] != float(value)]

    rejected_outliers = {}
    for column_id in gdf.columns:
        if gdf[column_id].dtype in (int, float, complex):
            outliers = list_outliers(gdf[column_id].to_numpy(), m=m_std_reject)
            if len(outliers) > 0:
                LOGGER.debug(f'{column_id}: {outliers}')
                gdf[column_id][gdf[column_id].isin(outliers)] = 0
                rejected_outliers[column_id] = outliers

    # load predictor/response table
    predictor_response_table = pandas.read_csv(predictor_response_table)
    dataset_map = {}
    for train_holdback_type, train_holdback_val in [
            ('holdback', True), ('train', False)]:
        predictor_response_map = collections.defaultdict(list)
        gdf_filtered = gdf[gdf['holdback']==train_holdback_val]
        for parameter_type in ['predictor', 'response']:
            for parameter_id in predictor_response_table[parameter_type]:
                if isinstance(parameter_id, str):
                    if parameter_id == 'geometry.x':
                        predictor_response_map[parameter_type].append(
                            gdf_filtered['geometry'].x)
                    elif parameter_id == 'geometry.y':
                        predictor_response_map[parameter_type].append(
                            gdf_filtered['geometry'].y)
                    else:
                        predictor_response_map[parameter_type].append(
                            gdf_filtered[parameter_id])

        x_tensor = torch.from_numpy(numpy.array(
            predictor_response_map['predictor'], dtype=numpy.float32).T)
        y_tensor = torch.from_numpy(numpy.array(
            predictor_response_map['response'], dtype=numpy.float32).T)
        dataset_map[train_holdback_type] = torch.utils.data.TensorDataset(
            x_tensor, y_tensor)
    return (
        predictor_response_table['predictor'].count(), dataset_map['train'],
        dataset_map['holdback'], rejected_outliers)


def train_cifar_ray(
        config, n_predictors, trainset, testset, n_epochs, working_dir,
        checkpoint_dir=None, ):
    model = NeuralNetwork(n_predictors, config["l1"], config["l2"])
    device = 'cpu'
    model.to(device)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = torch.utils.data.random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True)

    last_epoch = 0
    n_train_samples = len(train_loader)

    training_loss_list = []
    validation_loss_list = []

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        training_running_loss = 0.0
        epoch_steps = 0

        for i, (predictor_t, response_t) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(predictor_t)
            training_loss = loss_fn(outputs, response_t)
            training_loss.backward()
            optimizer.step()

            # print statistics
            training_running_loss += training_loss.item()

            epoch_steps += 1
            if i % 100 == 0:
                LOGGER.debug(f'[{epoch+1+last_epoch}/{n_epochs+last_epoch}, {i+1}/{int(numpy.ceil(n_train_samples/config["batch_size"]))}]')

        training_loss_list.append(training_running_loss/epoch_steps)

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for i, (predictor_t, response_t) in enumerate(val_loader):
            with torch.no_grad():
                outputs = model(predictor_t)
                loss = loss_fn(outputs, response_t)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        validation_loss_list.append(val_loss/val_steps)

        with ray.tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        outputs = model(trainset[:][0])
        ray.tune.report(
            loss=(val_loss / val_steps),
            r2=sklearn.metrics.r2_score(outputs.detach(), trainset[:][1]))
        #ray.tune.report(r2=sklearn.metrics.r2_score(outputs.detach(), trainset[:][1]))

        figure_prefix = os.path.join(
            working_dir, 'figdir',
            f'{config["l1"]}_{config["l2"]}_{config["lr"]}')

        expected_values = trainset[:][1].numpy().flatten()
        actual_values = outputs.detach().numpy().flatten()

        plot_metrics(
            figure_prefix,
            expected_values, actual_values, training_loss_list,
            validation_loss_list)
    print("Finished Training")


def plot_metrics(
        figure_prefix,
        expected_values, actual_values, training_loss_list,
        validation_loss_list):
    # plot model correlation graph
    fig, ax = plt.subplots(figsize=(12, 10))

    # equivalent but more general
    ax1 = plt.subplot(2, 1, 1)

    # add a subplot with no frame
    ax2 = plt.subplot(2, 1, 2, frameon=False)

    ax1.scatter(expected_values, actual_values, c='b', s=0.25)
    z = numpy.polyfit(expected_values, actual_values, 1)
    trendline_func = numpy.poly1d(z)

    r2 = sklearn.metrics.r2_score(actual_values, expected_values)

    ax1.set_xlabel('expected values')
    ax1.set_ylabel('actual values')
    ax1.set_ylim(-100, 200)
    ax1.plot(
        expected_values,
        trendline_func(expected_values),
        "r--", linewidth=1.5)
    ax1.set_title(f'$R^2={r2:.3f}$')

    ax2.set_xlabel('epoch values')
    ax2.set_ylabel('loss')
    ax2.plot(
        range(len(training_loss_list)),
        training_loss_list,
        "b-", linewidth=1.5, label='training loss')
    ax2.plot(
        range(len(validation_loss_list)),
        validation_loss_list,
        "r-", linewidth=1.5, label='validation loss')
    ax2.legend()
    ax2.set_title(f'Loss Function Model Trained with lat/lng coordinates')
    plt.savefig(f'{figure_prefix}_model_loss_{len(validation_loss_list):02d}.png')
    plt.close()


def list_outliers(data, m=100.):
    """List outliers in numpy array within m standard deviations of normal."""
    p99 = numpy.percentile(data, 99)
    p1 = numpy.percentile(data, 1)
    p50 = numpy.median(data)
    # p50 to p99 is 2.32635 sigma
    rSig = (p99-p1)/(2*2.32635)
    return numpy.unique(data[numpy.abs(data - p50) > rSig*m])


def main():
    parser = argparse.ArgumentParser(description='DNN model trainer')
    parser.add_argument('geopandas_data', type=str, help=(
        'path to geopandas structure to train on'))
    parser.add_argument('predictor_response_table', type=str, help=(
        'path to csv table with fields "predictor" and "response", the '
        'fieldnames underneath are used to sample the geopandas datastructure '
        'for training'))
    parser.add_argument('--n_epochs', required=True, type=int, help=(
        'number of iterations to run trainer'))
    parser.add_argument('--learning_rate', required=True, type=float, help=(
        'learning rate of initial epoch'))
    parser.add_argument('--momentum', required=True, type=float, help=(
        'momentum, default 0.9'), default=0.9)
    parser.add_argument('--last_epoch', help='last epoch to pick up at')
    parser.add_argument(
        '--n_rows', type=int,
        help='number of samples to train on from the dataset')
    parser.add_argument('--m_std_reject', type=float, default=100)
    parser.add_argument(
        '--invalid_values', type=str, nargs='*', default=list(),
        help='values to mask out of dataframe write as fieldname,value pairs')
    parser.add_argument(
        '--num_samples', type=int, default=10,
        help='number of times to do a sample to see best structure')
    args = parser.parse_args()

    n_predictors, trainset, testset, rejected_outliers = load_data(
        args.geopandas_data, args.invalid_values, args.n_rows, 100,
        args.predictor_response_table)

    config = {
        #"l1": ray.tune.sample_from(lambda _: 2 ** numpy.random.randint(2, 9)),
        #"l2": ray.tune.sample_from(lambda _: 2 ** numpy.random.randint(2, 9)),
        "l1": ray.tune.loguniform(0.1*n_predictors, 10*n_predictors),
        "l2": ray.tune.loguniform(0.1*n_predictors, 10*n_predictors),
        "lr": ray.tune.loguniform(args.learning_rate*1e-2, args.learning_rate*1e2),
        "batch_size": ray.tune.choice([1000, 500, 250, 100])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=args.n_epochs,
        grace_period=10,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration", "r2"])

    result = ray.tune.run(
        functools.partial(
            train_cifar_ray, trainset=trainset, testset=testset,
            n_epochs=args.n_epochs, n_predictors=n_predictors,
            working_dir=os.getcwd()),
        resources_per_trial={"cpu": 2},
        config=config,
        num_samples=args.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation R^2: {}".format(
        best_trial.last_result["r2"]))

    LOGGER.debug('all done')


if __name__ == '__main__':
    main()
