"""Compile a scikit.learn model to GEE."""
import hashlib
import numpy
import argparse
import pickle


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='compile scikit.learn model to GEE')
    parser.add_argument(
        'model_path', type=str, help='path to scikit.learn regression model')
    args = parser.parse_args()

    with open(args.model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # model['model'] has 3 steps:
    #   * polynomal features
    #   * standard scalar
    #   * model w/ coefficients
    feature_names = model['model'][0].get_feature_names_out(model['predictor_list'])
    feature_means = model['model'][1].mean_
    feature_scale = model['model'][1].scale_
    intercept = model['model'][2].intercept_
    feature_coef = model['model'][2].coef_

    # map features to a hash
    name_to_hash = {}
    hashes = {}
    for feature_name in feature_names:
        m = hashlib.sha256()
        m.update(feature_name.encode('utf-8'))
        hash_val = f'x{m.hexdigest()[:10]}'
        if hash_val in hashes:
            m2 = hashlib.sha256()
            m2.update(hashes[hash_val].encode('utf-8'))
            raise ValueError(f'duplciate hash\n\tt{hashes[hash_val]}\n\t{feature_name}\n\t{m.hexdigest()} vs {m2.hexdigest()}')
        hashes[hash_val] = feature_name
        name_to_hash[feature_name] = hash_val

    print('var null_image = ee.Image()')

    for predictor_id in model['predictor_list']:
        #print(f"var {name_to_hash[predictor_id]}: ee.Image.loadGeoTIFF('gs://cog/cog_{predictor_id}.tif')")
        print(f"var {name_to_hash[predictor_id]} = ee.Image.loadGeoTIFF('gs://ecoshard-root/cog/cog_downstream_bene_2017_50000.tif');")


    expression_list = []
    for index, (mean, scale, coef, term_expression) in enumerate(zip(feature_means, feature_scale, feature_coef, feature_names)):
        if '^' not in term_expression:
            names = term_expression.split(' ')
            term_expression = '*'.join([name_to_hash[name] for name in names])
        else:
            name, exp = term_expression.split('^')
            term_expression = f'{name_to_hash[name]}**{exp}'
            names = [name]

        scaled_coef = coef*scale+mean
        if numpy.isclose(scaled_coef, 0):
            continue
        load_string = ','.join([f"'{name_to_hash[name]}': {name_to_hash[name]}" for name in names])
        new_expression = (
            f"var term{index} = null_image.expression("
            f"'{coef/scale}*({term_expression}-{mean})', " + '{'
            f"{load_string}"+"});")
        print(new_expression)

    for coef in feature_coef:
        if numpy.isclose(coef, 0.0):
            continue
        print(coef)


if __name__ == '__main__':
    main()
