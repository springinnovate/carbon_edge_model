import matplotlib.pyplot as plt
import pandas

def main():
    table = pandas.read_csv(r"D:\repositories\combine_forest_plots_from_ForC_dev\simplified (new)\ForC_simplified.csv")
    table = table[table['variable.name'].isin(['biomass_ag', 'biomass'])]
    def filter_by_year(year):
        year = str(year).split('.')[0]
        try:
            year = int(year)
            if year < 2000:
                return False
        except Exception:
            pass
        return True

    def filter_by_precision(val):
        decimal = str(val).split('.')[1]
        if len(decimal) < 3:
            return False
        repeating_dec_12ths = [
            '0833', '1666', '3333', '4166', '5833', '6666', '8333', '9166']
        if decimal in repeating_dec_12ths:
            return False
        return True

    table = table[table['date'].map(filter_by_year)]
    table = table[table['lat'].map(filter_by_precision)]
    table = table[table['lon'].map(filter_by_precision)]

    #unique_by_site_lat_lon = (table.groupby(['sites.sitename', 'lat', 'lon']).mean()).reset_index()
    #unique_by_site_lat_lon_date = (table.groupby(['date', 'sites.sitename', 'lat', 'lon']).mean()).reset_index()

    mean_forest_carbon_plots_by_lat_long = table[[
        'sites.sitename',
        'date',
        'lat',
        'lon',
        'mean',
        'stand.age',
        'distmrs.type',
        ]].groupby(['date', 'sites.sitename', 'lat', 'lon']).mean().reset_index()
    mean_forest_carbon_plots_by_lat_long.to_csv('simplified_ForC_filtered.csv')

    # ax = table.hist(column='mean', by='variable.name', bins=25, grid=True, figsize=(8,10), layout=(4,1), sharex=True, color='#86bf91', zorder=2, rwidth=0.9)
    # #ax = table['mean'].hist()
    # plt.show()


if __name__ == '__main__':
    main()
