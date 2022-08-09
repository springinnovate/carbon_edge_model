"""
********
Start with two LULC maps:

1) restoration_limited
2) ESACCI-LC-L4-LCCS

build two forest masks off of these:

3) restoration_limited_forest_mask
4) ESACCI-LC-L4-LCCS_forest_mask
5) ESA_to_restoration_new_forest_mask

Build ESA carbon map since the change is just static and covert to co2

6) restoration_limited_new_forest_co2
7) ESACCI-LC-L4-LCCS_new_forest_co2

Build regression carbon maps and convert to co2

8) restoration_limited_regression_co2
9) ESACCI-LC-L4-LCCS_regression_co2

Build ESA marginal value map:

10) (calculate 6-7) ESA_marginal_value_co2_new_forest

Build regression marginal value:

* calculate convolution on new forest mask to show how many pixels from the
  new forest benefit the pixel under question
  11) new_forest_coverage_5km

* calculate 8-9 to find marginal value as a whole
  12) regression_marginal_value_raw

* calculate 12/11 to determine average coverage
  13) regression_marginal_value_average

* convolve 13 to 5km to estimate local marginal value benefit
  14) regression_marginal_value_co2_raw

* mask 14 to new forest
  15) regression_marginal_value_co2_new_forest
"""


def main():
    """Entry point."""
    pass


if __name__ == '__main__':
    main()
