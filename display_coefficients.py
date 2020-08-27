"""Tracer code for regression models."""
import argparse
import collections
import logging
import pickle
import os
import sys
import time

from osgeo import gdal
import numpy
import pygeoprocessing
import rtree
from sklearn.linear_model import LassoLarsCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.covariance import EmpiricalCovariance
import sklearn.preprocessing
import taskgraph

import carbon_model_data
from carbon_model_data import BASE_DATA_DIR
from utils import esa_to_carbon_model_landcover_types

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dump coeffients')
    parser.add_argument('model_path', help='path to pickled model')
    args = parser.parse_args()

    with open(args.model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    print(model[2])
