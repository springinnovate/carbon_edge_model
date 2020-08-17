"""Tracer code for regression models."""
import argparse
import glob
import logging
import multiprocessing
import os
import sys

import model_files

import retrying
import sklearn
import taskgraph

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)

LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)

BASE_DATA_DIR = 'base_data_no_humans_please'

MODEL_FIT_WORKSPACE = 'carbon_model'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model maker')
    parser.add_argument(
        '--raster_directory', help=(
            'Path to directory containing rasters to build a model from'))
    args = parser.parse_args()

    task_graph = taskgraph.TaskGraph(
        BASE_DATA_DIR,
        1, #multiprocessing.cpu_count(),
        5.0)

    model_files.fetch_data(BASE_DATA_DIR, task_graph)

    LOGGER.debug('all done!')

    task_graph.join()
    task_graph.close()
