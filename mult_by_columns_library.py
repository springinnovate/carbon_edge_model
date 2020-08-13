"""Demo of how to use pandas to multiply one table by another."""
import logging
import multiprocessing
import os
import re
import sys

from osgeo import gdal
import pandas
import pygeoprocessing
import pygeoprocessing.multiprocessing
import numpy

gdal.SetCacheMax(2**27)

# treat this one column name as special for the y intercept
INTERCEPT_COLUMN_ID = 'intercept'
OPERATOR_FN = {
    '+': numpy.add,
    '*': numpy.multiply,
    '^': numpy.power,
}
N_CPUS = multiprocessing.cpu_count()

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)

LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)


def raster_rpn_calculator_op(*args_list):
    """Calculate RPN expression.

    Args:
        args_list (list): a length list of N+4 long where:
            - the first N elements are array followed by nodata
            - the N+1th element is the target nodata
            - the N+2nd  element is an RPN stack containing either
              symbols, numeric values, or an operator in OPERATOR_SET.
            - N+3rd value is a dict mapping the symbol to a dict with
              "index" in it showing where index*2 location it is in the
              args_list.
            - N+4th value is a set of symbols that if present should set their
              nodata to 0.

    Returns:
        evaluation of the RPN calculation
    """
    n = len(args_list)-4
    result = numpy.empty(args_list[0].shape, dtype=numpy.float32)
    result[:] = args_list[n]  # target nodata
    rpn_stack = list(args_list[n+1])
    info_dict = args_list[n+2]
    zero_nodata_indexes = args_list[n+3]

    valid_mask = numpy.ones(args_list[0].shape, dtype=numpy.bool)
    # build up valid mask where all pixel stacks are defined
    for index in range(0, n, 2):
        nodata_value = args_list[index+1]
        if nodata_value is not None and index//2 not in zero_nodata_indexes:
            valid_mask &= \
                ~numpy.isclose(args_list[index], args_list[index+1])

    # process the rpn stack
    accumulator_stack = []
    while rpn_stack:
        val = rpn_stack.pop(0)
        if val in OPERATOR_FN:
            operator = val
            operand_b = accumulator_stack.pop()
            operand_a = accumulator_stack.pop()
            val = OPERATOR_FN[operator](operand_a, operand_b)
            accumulator_stack.append(val)
        else:
            if isinstance(val, str):
                arg_index = info_dict[val]['index']
                if arg_index in zero_nodata_indexes:
                    nodata_mask = numpy.isclose(
                        args_list[2*arg_index],  args_list[2*arg_index+1])
                    args_list[2*arg_index][nodata_mask] = 0.0
                accumulator_stack.append(args_list[2*arg_index][valid_mask])
            else:
                accumulator_stack.append(val)

    result[valid_mask] = accumulator_stack.pop(0)
    if accumulator_stack:
        raise RuntimeError(
            f'accumulator_stack not empty: {accumulator_stack}')
    return result


def evaluate_table_expression_as_raster(
        lasso_table_path, data_dir, workspace_dir,
        base_convolution_raster_id, target_raster_id,
        pixel_size, target_result_path, task_graph, n_workers,
        zero_nodata_symbols=None, target_nodata=numpy.finfo('float32').min):
    """Calculate large regression.

    Args:
        lasso_table_path (str): path to lasso table
        data_dir (str): path to directory containing rasters in lasso
            table path
        workspace_dir (str): path to output directory, will contain
            "result.tif" after completion
        base_convolution_raster_id (str): The convolution columns in
            the lasso table have the form  [base]_[mask_type]_gs[kernel_size],
            this parameter matches [base] so it can be replaced with a
            filename of the form [target_raster_id]_[mask_type]_[kernel_size].
        target_raster_id (str): this is the base of the target raster that
            to use in the table.
        pixel_size (tuple): desired target pixel size in raster units
        target_result_path (str): path to desired output raster
        task_graph (TaskGraph): TaskGraph object that can be used for
            scheduling.
        zero_nodata_symbols (set): set of symbols whose nodata values should be
            treated as 0.
        target_nodata (float): desired target nodata value
        n_workers (int): number of workers to allocate to raster eval

    Returns:
        None

    """
    lasso_df = pandas.read_csv(lasso_table_path, header=None)
    LOGGER.debug(f"parsing through {lasso_table_path}")
    # built a reverse polish notation stack for the operations and their order
    # that they need to be executed in
    rpn_stack = []
    first_term = True
    for row_index, row in lasso_df.iterrows():
        header = row[0]
        if header == INTERCEPT_COLUMN_ID:
            # special case of the intercept, just push it
            rpn_stack.append(float(row[1]))
        else:
            # it's an expression/coefficient row
            LOGGER.debug(f'{row_index}: {row}')
            coefficient = float(row[1])
            # put on the coefficient first since it's there, we'll multiply
            # it later
            rpn_stack.append(coefficient)

            # split out all the multiplcation terms
            product_list = header.split('*')
            for product in product_list:
                if product.startswith(base_convolution_raster_id):
                    LOGGER.debug(f'parsing out base and gs in {product}')
                    match = re.match(
                        fr'{base_convolution_raster_id}(.*)',
                        product)
                    suffix = match.group(1)
                    product = \
                        f'''{target_raster_id}{suffix}'''
                # for each multiplication term split out an exponent if exists
                if '^' in product:
                    rpn_stack.extend(product.split('^'))
                    # cast the exponent to an integer so can operate directly
                    rpn_stack[-1] = int(rpn_stack[-1])
                    # push the ^ to exponentiate the last two operations
                    rpn_stack.append('^')
                else:
                    # otherwise it's a single value
                    rpn_stack.append(product)
                # multiply this term and the last
                rpn_stack.append('*')

        # if it's not the first term we want to add the rest
        if first_term:
            first_term = False
        else:
            rpn_stack.append('+')

    LOGGER.debug(rpn_stack)

    # find the unique symbols in the expression
    raster_id_list = [
        x for x in set(rpn_stack)-set(OPERATOR_FN)
        if not isinstance(x, (int, float))]

    LOGGER.debug(raster_id_list)

    # translate symbols into raster paths and get relevant raster info
    raster_id_to_info_map = {}
    missing_raster_path_list = []
    for index, raster_id in enumerate(raster_id_list):
        raster_path = os.path.join(data_dir, f'{raster_id}.tif')
        if not os.path.exists(raster_path):
            missing_raster_path_list.append(raster_path)
            continue
        else:
            raster_info = pygeoprocessing.get_raster_info(raster_path)
            raster_id_to_info_map[raster_id] = {
                'path': raster_path,
                'nodata': raster_info['nodata'][0],
                'index': index,
            }

    if missing_raster_path_list:
        raise ValueError(
            f'expected the following '
            f'{"rasters" if len(missing_raster_path_list) > 1 else "raster"} '
            f'given the entries in the table, but could not find them '
            f'locally:\n' + "\n".join(missing_raster_path_list))

    LOGGER.info(f'raster paths:\n{str(raster_id_to_info_map)}')

    LOGGER.info('construct raster calculator raster path band list')
    raster_path_band_list = []
    LOGGER.debug(raster_id_list)
    LOGGER.debug(raster_id_to_info_map)
    for index, raster_id in enumerate(raster_id_list):
        raster_path_band_list.append(
            (raster_id_to_info_map[raster_id]['path'], 1))
        raster_path_band_list.append(
            (raster_id_to_info_map[raster_id]['nodata'], 'raw'))
        if index != raster_id_to_info_map[raster_id]['index']:
            raise RuntimeError(
                f"indexes dont match: {index} {raster_id} "
                f"{raster_id_to_info_map}")

    zero_nodata_indexes = {
        raster_id_to_info_map[raster_id]['index']
        for raster_id in zero_nodata_symbols
        if raster_id in raster_id_to_info_map}

    raster_path_band_list.append((target_nodata, 'raw'))
    raster_path_band_list.append((rpn_stack, 'raw'))
    raster_path_band_list.append((raster_id_to_info_map, 'raw'))
    raster_path_band_list.append((zero_nodata_indexes, 'raw'))
    LOGGER.debug(rpn_stack)

    # wait for rasters to align
    task_graph.join()

    LOGGER.debug(raster_path_band_list)
    pygeoprocessing.multiprocessing.raster_calculator(
        raster_path_band_list, raster_rpn_calculator_op, target_result_path,
        gdal.GDT_Float32, float(target_nodata), n_workers)
    LOGGER.debug('all done with mult by raster')


def evaluate_table_expression_at_point(
        lasso_table_path, vector_path, fid, data_dir, workspace_dir,
        base_convolution_raster_id, target_raster_id,
        target_result_table_path):
    """Calculate table regression at a single point.

    Args:
        lasso_table_path (str): path to lasso table
        vector_path (str): path to point vector used to evaluate a single point
        fid (int): this fid is the feature in `vector_path` used to evalute
            the mode
        data_dir (str): path to directory containing rasters in lasso
            table path
        workspace_dir (str): path to output directory, will contain
            "result.tif" after completion
        base_convolution_raster_id (str): The convolution columns in
            the lasso table have the form  [base]_[mask_type]_gs[kernel_size],
            this parameter matches [base] so it can be replaced with a
            filename of the form [target_raster_id]_[mask_type]_[kernel_size].
        target_raster_id (str): this is the base of the target raster that
            to use in the table.
        target_result_table_path (str): path to desired output table

    Returns:
        None

    """
    lasso_df = pandas.read_csv(lasso_table_path, header=None)
    LOGGER.debug(f"parsing through {lasso_table_path}")
    # built a reverse polish notation stack for the operations and their order
    # that they need to be executed in
    rpn_stack = []
    first_term = True
    for row_index, row in lasso_df.iterrows():
        header = row[0]
        if header == INTERCEPT_COLUMN_ID:
            # special case of the intercept, just push it
            rpn_stack.append(float(row[1]))
        else:
            # it's an expression/coefficient row
            LOGGER.debug(f'{row_index}: {row}')
            coefficient = float(row[1])
            # put on the coefficient first since it's there, we'll multiply
            # it later
            rpn_stack.append(coefficient)

            # split out all the multiplcation terms
            product_list = header.split('*')
            for product in product_list:
                if product.startswith(base_convolution_raster_id):
                    LOGGER.debug(f'parsing out base and gs in {product}')
                    match = re.match(
                        fr'{base_convolution_raster_id}(.*)',
                        product)
                    suffix = match.group(1)
                    product = \
                        f'''{target_raster_id}{suffix}'''
                # for each multiplication term split out an exponent if exists
                if '^' in product:
                    rpn_stack.extend(product.split('^'))
                    # cast the exponent to an integer so can operate directly
                    rpn_stack[-1] = int(rpn_stack[-1])
                    # push the ^ to exponentiate the last two operations
                    rpn_stack.append('^')
                else:
                    # otherwise it's a single value
                    rpn_stack.append(product)
                # multiply this term and the last
                rpn_stack.append('*')

        # if it's not the first term we want to add the rest
        if first_term:
            first_term = False
        else:
            rpn_stack.append('+')

    LOGGER.debug(rpn_stack)

    # find the unique symbols in the expression
    raster_id_list = [
        x for x in set(rpn_stack)-set(OPERATOR_FN)
        if not isinstance(x, (int, float))]

    LOGGER.debug(raster_id_list)

    # translate symbols into raster paths and get relevant raster info
    raster_id_to_info_map = {}
    missing_raster_path_list = []
    for index, raster_id in enumerate(raster_id_list):
        raster_path = os.path.join(data_dir, f'{raster_id}.tif')
        if not os.path.exists(raster_path):
            missing_raster_path_list.append(raster_path)
            continue
        else:
            raster_info = pygeoprocessing.get_raster_info(raster_path)
            raster_id_to_info_map[raster_id] = {
                'path': raster_path,
                'nodata': raster_info['nodata'][0],
                'index': index,
            }

    if missing_raster_path_list:
        raise ValueError(
            f'expected the following '
            f'{"rasters" if len(missing_raster_path_list) > 1 else "raster"} '
            f'given the entries in the table, but could not find them '
            f'locally:\n' + "\n".join(missing_raster_path_list))

    LOGGER.info(f'raster paths:\n{str(raster_id_to_info_map)}')

    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    feature = layer.GetFeature(fid)
    feature_geom = feature.GetGeometryRef()
    x = feature_geom.GetX()
    y = feature_geom.GetY()
    feature_geom = None
    feature = None
    layer = None
    vector = None

    base_raster_path = next(iter(raster_id_to_info_map.values()))['path']
    base_raster_info = pygeoprocessing.get_raster_info(base_raster_path)
    inv_geotransform = gdal.InvGeoTransform(base_raster_info['geotransform'])
    n_cols, n_rows = base_raster_info['raster_size']
    col, row = [
        int(coord) for coord in gdal.ApplyGeoTransform(inv_geotransform, x, y)]

    LOGGER.debug(f"\n\n\n\nquerying at coord {x} {y} boundingbox {base_raster_info['bounding_box']}\n{inv_geotransform}\n\n")
    LOGGER.debug(rpn_stack)
    with open(target_result_table_path, 'w') as target_table_file:
        val_accumulator_stack = []
        symbol_accumulator_stack = []
        accumulator_stack = []
        while rpn_stack:
            val = rpn_stack.pop(0)
            LOGGER.debug(f'pop: {val}')
            if val in OPERATOR_FN:
                operator = val
                if operator == '+':
                    # newline!
                    LOGGER.debug(f'writing: {symbol_accumulator_stack}')
                    target_table_file.write(
                        f'{symbol_accumulator_stack.pop()}\',')
                    target_table_file.write(
                        f'{val_accumulator_stack.pop()}\',')
                    target_table_file.write(
                        f'{accumulator_stack.pop()}\n')
                    continue

                symbol_b = symbol_accumulator_stack.pop()
                symbol_a = symbol_accumulator_stack.pop()
                symbol_accumulator_stack.append(
                    f'{symbol_a}{operator}{symbol_b}')

                val_b = val_accumulator_stack.pop()
                val_a = val_accumulator_stack.pop()
                val_accumulator_stack.append(
                    f'{val_a}{operator}{val_b}')

                operand_b = accumulator_stack.pop()
                operand_a = accumulator_stack.pop()
                val = OPERATOR_FN[operator](operand_a, operand_b)
                accumulator_stack.append(val)
            else:
                if isinstance(val, str):
                    raster_path = raster_id_to_info_map[val]['path']
                    raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
                    band = raster.GetRasterBand(1)
                    raster_val = band.ReadAsArray(col, row, 1, 1)[0, 0]
                    accumulator_stack.append(raster_val)
                    val_accumulator_stack.append(raster_val)
                else:
                    accumulator_stack.append(val)
                    val_accumulator_stack.append(val)
                symbol_accumulator_stack.append(f'{val}')

    LOGGER.debug('all done with eval at point')
