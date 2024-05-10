import matplotlib.pyplot as plt

STUDY_GROUP_FIELD = 'Study'
PERCENTILE = 0.99
BACCINI_NODATA = 32767
REGRESSION_NODATA = -1
YEAR_THRESHOLD = 2005 # 1990
YEAR_FIELD = 'Date (most recent)'
OBSERVED_CARBON_THRESHOLD = 10

REGRESSION_MODEL = 'regression_carbon_esa_compressed_md5_c867a0'
import pandas
import numpy as np
IPCC_MODEL = 'ipcc_carbon_esa_compressed_md5_5b4803'
BACCINI_MODEL = 'baccini_carbon_data_2014_compressed'
OBSERVED = 'Observed_C'


def main():
    carbon_validation_table_path = "data/ForC_carbon_validation_051024.csv"
    carbon_validation_table = pandas.read_csv(carbon_validation_table_path)

    carbon_validation_table = carbon_validation_table[carbon_validation_table[YEAR_FIELD] >= YEAR_THRESHOLD ]
    #carbon_validation_table = carbon_validation_table[carbon_validation_table[YEAR_FIELD] == YEAR_THRESHOLD ]

    carbon_validation_table = carbon_validation_table[carbon_validation_table[OBSERVED] >= OBSERVED_CARBON_THRESHOLD ]

    study_counts = carbon_validation_table[STUDY_GROUP_FIELD].value_counts()
    nth_percentile = study_counts.quantile(PERCENTILE)
    top_studies = (study_counts[study_counts >= nth_percentile].index).unique()

    study_rows = [
        (study_id, carbon_validation_table[
            carbon_validation_table[STUDY_GROUP_FIELD] == study_id])
        for study_id in top_studies]

    non_top_studies_rows = carbon_validation_table[
        ~carbon_validation_table[STUDY_GROUP_FIELD].isin(top_studies)]
    study_rows += [(f'< than {PERCENTILE}', non_top_studies_rows)]

    print('group,n samples,model,rmse,R2,observed min,observed max,model min,model max')
    for study_group, study_row in study_rows:
        nodata_mask = (study_row[BACCINI_MODEL] == BACCINI_NODATA) | (study_row[REGRESSION_MODEL] == REGRESSION_NODATA)
        study_row = study_row[~nodata_mask]
        if len(study_row) == 0:
            continue
        max_value = study_row[OBSERVED].max()
        plt.figure(figsize=(8, 6))
        for model_id in [BACCINI_MODEL, REGRESSION_MODEL, IPCC_MODEL,]:
            max_value = max(max_value, study_row[model_id].max())
            # Create a scatter plot
            #plt.clf()
            residuals = study_row[OBSERVED] - study_row[model_id]*0.47
            rmse = np.sqrt(np.mean(residuals**2))

            plt.scatter(study_row[OBSERVED], study_row[model_id]*0.47, alpha=0.5, label=f'{model_id} RMSE: {rmse:.2f} ')


            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((study_row[OBSERVED] - np.mean(study_row[OBSERVED]))**2)
            r_squared = 1 - (ss_res / ss_tot)

            print(f'{study_group},{len(study_row)},{model_id} vs {OBSERVED},{rmse},{r_squared},{study_row[OBSERVED].min()},{study_row[OBSERVED].max()},{study_row[model_id].min()},{study_row[model_id].max()}')
        plt.plot([0, max_value], [0, max_value], 'r-', label='1:1 Line')
        plt.title(f'Scatter Plot of OBSERVED vs. modeled on {study_group} ({len(study_row)})')
        plt.xlabel(OBSERVED)
        plt.ylabel('model')
        plt.xlim(0, max_value)
        plt.ylim(0, max_value)
        plt.grid(True)
        plt.legend(title='Model ID')
        plt.show()



if __name__ == '__main__':
    main()
