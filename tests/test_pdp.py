
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas.testing import assert_frame_equal
import pandas as pd

from pdpbox.pdp import pdp_isolate, pdp_interact


def test_pdp_isolate_binary(titanic_model, titanic_data, titanic_features):
    # feature_type: binary
    pdp_isolate_out = pdp_isolate(
        model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Sex',
        num_grid_points=10, grid_type='percentile', percentile_range=None, grid_range=None, cust_grid_points=None,
        memory_limit=0.5, n_jobs=1, predict_kwds=None, data_transformer=None)

    assert pdp_isolate_out._type == 'PDPIsolate_instance'
    assert pdp_isolate_out.n_classes == 2
    assert pdp_isolate_out.which_class is None
    assert pdp_isolate_out.feature == 'Sex'
    assert pdp_isolate_out.feature_type == 'binary'
    assert_array_equal(pdp_isolate_out.feature_grids, np.array([0, 1]))
    assert pdp_isolate_out.percentile_info == []
    assert pdp_isolate_out.display_columns == ['Sex_0', 'Sex_1']

    ice_lines_expected = pd.DataFrame({0: {0: 0.527877688407898, 200: 0.426897794008255, 400: 0.3850491940975189,
                                           600: 0.8612496852874756, 800: 0.9099752902984619},
                                       1: {0: 0.1041787713766098, 200: 0.12183572351932526, 400: 0.15937349200248718,
                                           600: 0.14207805693149567, 800: 0.1512364000082016}})
    assert_frame_equal(pdp_isolate_out.ice_lines.iloc[[0, 200, 400, 600, 800]], ice_lines_expected,
                       check_like=True, check_dtype=False)

    assert_array_equal(pdp_isolate_out.pdp, np.array([0.680056, 0.22317506], dtype=np.float32))

    count_data_expected = pd.DataFrame({'count': {0: 314, 1: 577},
                                        'count_norm': {0: 0.35241301907968575, 1: 0.6475869809203143},
                                        'x': {0: 0, 1: 1}})
    assert_frame_equal(pdp_isolate_out.count_data, count_data_expected, check_like=True, check_dtype=False)

    assert pdp_isolate_out.hist_data is None

    # feature_type: onehot
    pdp_isolate_out = pdp_isolate(
        model=titanic_model, dataset=titanic_data, model_features=titanic_features,
        feature=['Embarked_C', 'Embarked_S', 'Embarked_Q'], num_grid_points=10, grid_type='percentile',
        percentile_range=None, grid_range=None, cust_grid_points=None, memory_limit=0.5, n_jobs=1,
        predict_kwds=None, data_transformer=None)

    assert pdp_isolate_out.feature == ['Embarked_C', 'Embarked_S', 'Embarked_Q']
    assert pdp_isolate_out.feature_type == 'onehot'
    assert_array_equal(pdp_isolate_out.feature_grids, np.array(['Embarked_C', 'Embarked_S', 'Embarked_Q'], dtype='|S10'))
    assert pdp_isolate_out.percentile_info == []
    assert pdp_isolate_out.display_columns == ['Embarked_C', 'Embarked_S', 'Embarked_Q']

    ice_lines_expected = pd.DataFrame({'Embarked_C': {0: 0.19760717451572418, 200: 0.18956533074378967,
                                                      400: 0.1699531376361847, 600: 0.9045996069908142,
                                                      800: 0.16102053225040436},
                                       'Embarked_Q': {0: 0.11906618624925613, 200: 0.1521308720111847,
                                                      400: 0.14191272854804993, 600: 0.890386700630188,
                                                      800: 0.12191008776426315},
                                       'Embarked_S': {0: 0.1041787713766098, 200: 0.12183572351932526,
                                                      400: 0.15937349200248718, 600: 0.8612496852874756,
                                                      800: 0.1512364000082016}})
    assert_frame_equal(pdp_isolate_out.ice_lines.iloc[[0, 200, 400, 600, 800]], ice_lines_expected,
                       check_like=True, check_dtype=False)

    assert_array_equal(pdp_isolate_out.pdp, np.array([0.43266338, 0.3675584, 0.3948751], dtype=np.float32))

    count_data_expected = pd.DataFrame({'count': {0: 168, 1: 646, 2: 77},
                                        'count_norm': {0: 0.18855218855218855, 1: 0.7250280583613917,
                                                       2: 0.08641975308641975},
                                        'index': {0: 'Embarked_C', 1: 'Embarked_S', 2: 'Embarked_Q'},
                                        'x': {0: 0, 1: 1, 2: 2}})
    assert_frame_equal(pdp_isolate_out.count_data, count_data_expected, check_like=True, check_dtype=False)
    assert pdp_isolate_out.hist_data is None

    # feature_type: numeric
    pdp_isolate_out = pdp_isolate(
        model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Fare', num_grid_points=10,
        grid_type='percentile', percentile_range=None, grid_range=None, cust_grid_points=None, memory_limit=0.5,
        n_jobs=1, predict_kwds=None, data_transformer=None)

    assert pdp_isolate_out.feature == 'Fare'
    assert pdp_isolate_out.feature_type == 'numeric'
    assert_array_almost_equal(pdp_isolate_out.feature_grids,
                              np.array([0., 7.73284444, 7.8958, 8.6625, 13., 16.7, 26., 35.11111111, 73.5, 512.3292]),
                              decimal=8)
    assert_array_equal(pdp_isolate_out.percentile_info,
                       np.array(['(0.0)', '(11.11)', '(22.22)', '(33.33)', '(44.44)', '(55.56)', '(66.67)', '(77.78)',
                                 '(88.89)', '(100.0)'], dtype=object))
    assert pdp_isolate_out.display_columns == [0.0, 7.733, 7.896, 8.662, 13.0, 16.7, 26.0, 35.111, 73.5, 512.329]

    ice_lines_expected = pd.DataFrame(
        {7.732844444444444: {0: 0.10785847157239914, 500: 0.10733785480260849, 800: 0.08447999507188797},
         16.7: {0: 0.11272012442350388, 500: 0.1367582231760025, 800: 0.16041819751262665},
         512.3292: {0: 0.22303232550621033, 500: 0.26529940962791443, 800: 0.3425298035144806}})
    assert_frame_equal(pdp_isolate_out.ice_lines.loc[[0, 500, 800], pdp_isolate_out.ice_lines.columns.values[[1, 5, 9]]],
                       ice_lines_expected, check_like=True, check_dtype=False)

    assert_array_equal(pdp_isolate_out.pdp[[1, 5, 9]], np.array([0.34042153, 0.3876946, 0.48316196], dtype=np.float32))

    count_data_expected = pd.DataFrame({'count': {0: 99, 4: 108, 8: 102},
                                        'count_norm': {0: 0.1111111111111111, 4: 0.12121212121212122,
                                                       8: 0.11447811447811448},
                                        'x': {0: 0, 4: 4, 8: 8},
                                        'xticklabels': {0: '[0, 7.73)', 4: '[13, 16.7)', 8: '[73.5, 512.33]'}})
    assert_frame_equal(pdp_isolate_out.count_data.iloc[[0, 4, 8]], count_data_expected,
                       check_like=True, check_dtype=False)

    assert len(pdp_isolate_out.hist_data) == titanic_data.shape[0]
    assert_array_equal(pdp_isolate_out.hist_data[[0, 200, 400, 600, 800]], np.array([7.25, 9.5, 7.925, 27., 13.]))

    # use cust_grid_points
    pdp_isolate_out = pdp_isolate(
        model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Fare', num_grid_points=10,
        grid_type='percentile', percentile_range=None, grid_range=None, cust_grid_points=range(0, 100, 5),
        memory_limit=0.5, n_jobs=1, predict_kwds=None, data_transformer=None)

    assert_array_equal(pdp_isolate_out.feature_grids, range(0, 100, 5))
    assert pdp_isolate_out.percentile_info == []
    assert pdp_isolate_out.display_columns == range(0, 100, 5)

    ice_lines_expected = pd.DataFrame(
        {5: {0: 0.08728538453578949, 500: 0.08203054219484329, 800: 0.05646398663520813},
         25: {0: 0.10983016341924667, 500: 0.1353892683982849, 800: 0.1680663526058197},
         45: {0: 0.13062693178653717, 500: 0.15881720185279846, 800: 0.22301127016544342}})
    assert_frame_equal(pdp_isolate_out.ice_lines.loc[[0, 500, 800], pdp_isolate_out.ice_lines.columns.values[[1, 5, 9]]],
                       ice_lines_expected, check_like=True, check_dtype=False)

    assert_array_equal(pdp_isolate_out.pdp[[1, 5, 9]], np.array([0.29875883, 0.35867965, 0.38914606], dtype=np.float32))

    count_data_expected = pd.DataFrame({'count': {0: 16, 5: 94, 10: 20, 15: 21},
                                        'count_norm': {0: 0.017957351290684626, 5: 0.10549943883277217,
                                                       10: 0.02244668911335578, 15: 0.02356902356902357},
                                        'x': {0: 0, 5: 5, 10: 10, 15: 15},
                                        'xticklabels': {0: '[0, 5)', 5: '[25, 30)', 10: '[50, 55)', 15: '[75, 80)'}})
    assert_frame_equal(pdp_isolate_out.count_data.iloc[[0, 5, 10, 15]], count_data_expected,
                       check_like=True, check_dtype=False)



