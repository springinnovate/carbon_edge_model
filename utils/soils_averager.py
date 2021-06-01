"""Average soils layers."""
import collections
import glob
import os
import re

import pygeoprocessing
import numpy


def average(array_a, array_b):
    result = numpy.empty(array_a.shape, dtype=numpy.float32)
    valid_array = (array_a >= 0) & (array_b >= 0)

    return result


if __name__ == '__main__':
    raster_pairs = collections.defaultdict(list)
    for path in glob.glob('soil_layers/soils21/*.tif'):
        print(path)
        prefix = os.path.basename(path).split('_')[0]
        match = re.match('([^_]*)_[^_]*_([^_]*)_.*', os.path.basename(path))
        prefix = f'{match.group(1)}_{match.group(2)}'
        print(prefix)
        raster_pairs[prefix].append(path)
    print(raster_pairs)
