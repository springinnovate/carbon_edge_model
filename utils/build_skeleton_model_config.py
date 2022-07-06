"""Create a skeleton model training CSV config."""
import argparse
import os

import geopandas


def main():
    parser = argparse.ArgumentParser(description='Construct model skeleton')
    parser.add_argument('geopandas_data', type=str, help=(
        'path to geopandas structure to train on'))
    parser.add_argument(
        '--output_filename', type=str, help='define output file')
    args = parser.parse_args()
    gdf = geopandas.read_file(args.geopandas_data)

    if args.output_filename is not None:
        output_filename = args.output_filename
    else:
        output_filename = f'''model_config_{
            os.path.basename(os.path.splitext(args.geopandas_data)[0])}.csv'''

    with open(output_filename, 'w') as csv_file:
        csv_file.write(
            'predictor,response,include,exclude,min,max,group,'
            'target,filter_only\n')

        for column in gdf.columns:
            if column not in ['geometry', 'holdback']:
                csv_file.write(f'{column}\n')


if __name__ == '__main__':
    main()
