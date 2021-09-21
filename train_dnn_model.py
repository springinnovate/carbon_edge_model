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

TIME_PREDICTOR_LIST = [
    #('baccini_carbon_error_compressed_wgs84__md5_77ea391e63c137b80727a00e4945642f.tif', None),
]

LULC_TIME_LIST = [
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2003-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2004-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2005-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2006-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2007-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2008-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2009-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2010-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2011-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2012-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2013-v2.0.7_smooth_compressed.tif', None),
    ('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2014-v2.0.7_smooth_compressed.tif', None)]

PREDICTOR_LIST = [
    ('accessibility_to_cities_2015_30sec_compressed_wgs84__md5_a6a8ffcb6c1025c131f7663b80b3c9a7.tif', -9999),
    ('altitude_10sec_compressed_wgs84__md5_bfa771b1aef1b18e48962c315e5ba5fc.tif', None),
    ('bio_01_30sec_compressed_wgs84__md5_3f851546237e282124eb97b479c779f4.tif', -9999),
    ('bio_02_30sec_compressed_wgs84__md5_7ad508baff5bbd8b2e7991451938a5a7.tif', -9999),
    ('bio_03_30sec_compressed_wgs84__md5_a2de2d38c1f8b51f9d24f7a3a1e5f142.tif', -9999),
    ('bio_04_30sec_compressed_wgs84__md5_94cfca6af74ffe52316a02b454ba151b.tif', -9999),
    ('bio_05_30sec_compressed_wgs84__md5_bdd225e46613405c80a7ebf7e3b77249.tif', -9999),
    ('bio_06_30sec_compressed_wgs84__md5_ef252a4335eafb7fe7b4dc696d5a70e3.tif', -9999),
    ('bio_07_30sec_compressed_wgs84__md5_1db9a6cdce4b3bd26d79559acd2bc525.tif', -9999),
    ('bio_08_30sec_compressed_wgs84__md5_baf898dd624cfc9415092d7f37ae44ff.tif', -9999),
    ('bio_09_30sec_compressed_wgs84__md5_180c820aae826529bfc824b458165eee.tif', -9999),
    ('bio_10_30sec_compressed_wgs84__md5_d720d781970e165a40a1934adf69c80e.tif', -9999),
    ('bio_11_30sec_compressed_wgs84__md5_f48a251c54582c22d9eb5d2158618bbe.tif', -9999),
    ('bio_12_30sec_compressed_wgs84__md5_23cb55c3acc544e5a941df795fcb2024.tif', -9999),
    ('bio_13_30sec_compressed_wgs84__md5_b004ebe58d50841859ea485c06f55bf6.tif', -9999),
    ('bio_14_30sec_compressed_wgs84__md5_7cb680af66ff6c676441a382519f0dc2.tif', -9999),
    ('bio_15_30sec_compressed_wgs84__md5_edc8e5af802448651534b7a0bd7113ac.tif', -9999),
    ('bio_16_30sec_compressed_wgs84__md5_a9e737a926f1f916746d8ce429c06fad.tif', -9999),
    ('bio_17_30sec_compressed_wgs84__md5_0bc4db0e10829cd4027b91b7bbfc560f.tif', -9999),
    ('bio_18_30sec_compressed_wgs84__md5_76cf3d38eb72286ba3d5de5a48bfadd4.tif', -9999),
    ('bio_19_30sec_compressed_wgs84__md5_a91b8b766ed45cb60f97e25bcac0f5d2.tif', -9999),
    ('cec_0-5cm_mean_compressed_wgs84__md5_b3b4285906c65db596a014d0c8a927dd.tif', None),
    ('cec_0-5cm_uncertainty_compressed_wgs84__md5_f0f4eb245fd2cc4d5a12bd5f37189b53.tif', None),
    ('cec_5-15cm_mean_compressed_wgs84__md5_55c4d960ca9006ba22c6d761d552c82f.tif', None),
    ('cec_5-15cm_uncertainty_compressed_wgs84__md5_880eac199a7992f61da6c35c56576202.tif', None),
    ('cfvo_0-5cm_mean_compressed_wgs84__md5_7abefac8143a706b66a1b7743ae3cba1.tif', None),
    ('cfvo_0-5cm_uncertainty_compressed_wgs84__md5_3d6b883fba1d26a6473f4219009298bb.tif', None),
    ('cfvo_5-15cm_mean_compressed_wgs84__md5_ae36d799053697a167d114ae7821f5da.tif', None),
    ('cfvo_5-15cm_uncertainty_compressed_wgs84__md5_1f2749cd35adc8eb1c86a67cbe42aebf.tif', None),
    ('clay_0-5cm_mean_compressed_wgs84__md5_9da9d4017b691bc75c407773269e2aa3.tif', None),
    ('clay_0-5cm_uncertainty_compressed_wgs84__md5_f38eb273cb55147c11b48226400ae79a.tif', None),
    ('clay_5-15cm_mean_compressed_wgs84__md5_c136adb39b7e1910949b749fcc16943e.tif', None),
    ('clay_5-15cm_uncertainty_compressed_wgs84__md5_0acc36c723aa35b3478f95f708372cc7.tif', None),
    ('hillshade_10sec_compressed_wgs84__md5_192a760d053db91fc9e32df199358b54.tif', None),
    ('night_lights_10sec_compressed_wgs84__md5_54e040d93463a2918a82019a0d2757a3.tif', None),
    ('night_lights_5min_compressed_wgs84__md5_e36f1044d45374c335240777a2b94426.tif', None),
    ('nitrogen_0-5cm_mean_compressed_wgs84__md5_6adecc8d790ccca6057a902e2ddd0472.tif', None),
    ('nitrogen_0-5cm_uncertainty_compressed_wgs84__md5_4425b4bd9eeba0ad8a1092d9c3e62187.tif', None),
    ('nitrogen_10sec_compressed_wgs84__md5_1aed297ef68f15049bbd987f9e98d03d.tif', None),
    ('nitrogen_5-15cm_mean_compressed_wgs84__md5_9487bc9d293effeb4565e256ed6e0393.tif', None),
    ('nitrogen_5-15cm_uncertainty_compressed_wgs84__md5_2de5e9d6c3e078756a59ac90e3850b2b.tif', None),
    ('phh2o_0-5cm_mean_compressed_wgs84__md5_00ab8e945d4f7fbbd0bddec1cb8f620f.tif', None),
    ('phh2o_0-5cm_uncertainty_compressed_wgs84__md5_8090910adde390949004f30089c3ae49.tif', None),
    ('phh2o_5-15cm_mean_compressed_wgs84__md5_9b187a088ecb955642b9a86d56f969ad.tif', None),
    ('phh2o_5-15cm_uncertainty_compressed_wgs84__md5_6809da4b13ebbc747750691afb01a119.tif', None),
    ('sand_0-5cm_mean_compressed_wgs84__md5_6c73d897cdef7fde657386af201a368d.tif', None),
    ('sand_0-5cm_uncertainty_compressed_wgs84__md5_efd87fd2062e8276148154c4a59c9b25.tif', None),
    ('sand_5-15cm_uncertainty_compressed_wgs84__md5_03bc79e2bfd770a82c6d15e36a65fb5c.tif', None),
    ('silt_0-5cm_mean_compressed_wgs84__md5_1d141933d8d109df25c73bd1dcb9d67c.tif', None),
    ('silt_0-5cm_uncertainty_compressed_wgs84__md5_ac5ec50cbc3b9396cf11e4e431b508a9.tif', None),
    ('silt_5-15cm_mean_compressed_wgs84__md5_d0abb0769ebd015fdc12b50b20f8c51e.tif', None),
    ('silt_5-15cm_uncertainty_compressed_wgs84__md5_cc125c85815db0d1f66b315014907047.tif', None),
    ('slope_10sec_compressed_wgs84__md5_e2bdd42cb724893ce8b08c6680d1eeaf.tif', None),
    ('soc_0-5cm_mean_compressed_wgs84__md5_b5be42d9d0ecafaaad7cc592dcfe829b.tif', None),
    ('soc_0-5cm_uncertainty_compressed_wgs84__md5_33c1a8c3100db465c761a9d7f4e86bb9.tif', None),
    ('soc_5-15cm_mean_compressed_wgs84__md5_4c489f6132cc76c6d634181c25d22d19.tif', None),
    ('tri_10sec_compressed_wgs84__md5_258ad3123f05bc140eadd6246f6a078e.tif', None),
    ('wind_speed_10sec_compressed_wgs84__md5_7c5acc948ac0ff492f3d148ffc277908.tif', None),
]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, M, l1=100, l2=100):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(M, l1),
            torch.nn.Linear(l1, l2),
            torch.nn.Linear(l2, 1),
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

    parser.add_argument(
        '--invalid_values', type=str, nargs='*', default=list(),
        help='values to mask out of dataframe write as fieldname,value pairs')
    parser.add_argument(
        '--num_samples', type=int, default=10,
        help='number of times to do a sample to see best structure')
    args = parser.parse_args()

    n_predictors, trainset, testset, rejected_outliers = load_data(
        args.geopandas_data, args.invalid_values, args.n_rows,
        args.predictor_response_table)

    config = {
        #"l1": ray.tune.sample_from(lambda _: 2 ** numpy.random.randint(2, 9)),
        #"l2": ray.tune.sample_from(lambda _: 2 ** numpy.random.randint(2, 9)),
        "l1": ray.tune.choice([100]),
        "l2": ray.tune.choice([100]),
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
