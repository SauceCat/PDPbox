
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas.testing import assert_frame_equal
import pandas as pd
import matplotlib

from pdpbox.pdp import pdp_isolate, pdp_plot


class TestPDPIsolateBinary(object):
    def test_pdp_isolate_binary_feature(self, titanic_model, titanic_data, titanic_features):
        # feature_type: binary
        pdp_isolate_out = pdp_isolate(
            model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Sex',
            num_grid_points=10, grid_type='percentile', percentile_range=None, grid_range=None, cust_grid_points=None,
            memory_limit=0.5, n_jobs=1, predict_kwds={}, data_transformer=None)

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
                                           1: {0: 0.1041787713766098, 200: 0.12183572351932526,
                                               400: 0.15937349200248718,
                                               600: 0.14207805693149567, 800: 0.1512364000082016}})
        assert_frame_equal(pdp_isolate_out.ice_lines.iloc[[0, 200, 400, 600, 800]], ice_lines_expected,
                           check_like=True, check_dtype=False)

        assert_array_almost_equal(pdp_isolate_out.pdp, np.array([0.680056, 0.22317506]), decimal=6)

        count_data_expected = pd.DataFrame({'count': {0: 314, 1: 577},
                                            'count_norm': {0: 0.35241301907968575, 1: 0.6475869809203143},
                                            'x': {0: 0, 1: 1}})
        assert_frame_equal(pdp_isolate_out.count_data, count_data_expected, check_like=True, check_dtype=False)

        assert pdp_isolate_out.hist_data is None

    def test_pdp_isolate_onehot_feature(self, titanic_model, titanic_data, titanic_features):
        # feature_type: onehot
        pdp_isolate_out = pdp_isolate(
            model=titanic_model, dataset=titanic_data, model_features=titanic_features,
            feature=['Embarked_C', 'Embarked_S', 'Embarked_Q'], num_grid_points=10, grid_type='percentile',
            percentile_range=None, grid_range=None, cust_grid_points=None, memory_limit=0.5, n_jobs=1,
            predict_kwds={}, data_transformer=None)

        assert pdp_isolate_out._type == 'PDPIsolate_instance'
        assert pdp_isolate_out.n_classes == 2
        assert pdp_isolate_out.which_class is None
        assert pdp_isolate_out.feature == ['Embarked_C', 'Embarked_S', 'Embarked_Q']
        assert pdp_isolate_out.feature_type == 'onehot'
        assert_array_equal(pdp_isolate_out.feature_grids,
                           np.array(['Embarked_C', 'Embarked_S', 'Embarked_Q']))
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

        assert_array_almost_equal(pdp_isolate_out.pdp, np.array([0.43266338, 0.3675584, 0.3948751]), decimal=7)

        count_data_expected = pd.DataFrame({'count': {0: 168, 1: 646, 2: 77},
                                            'count_norm': {0: 0.18855218855218855, 1: 0.7250280583613917,
                                                           2: 0.08641975308641975},
                                            'index': {0: 'Embarked_C', 1: 'Embarked_S', 2: 'Embarked_Q'},
                                            'x': {0: 0, 1: 1, 2: 2}})
        assert_frame_equal(pdp_isolate_out.count_data, count_data_expected, check_like=True, check_dtype=False)
        assert pdp_isolate_out.hist_data is None

    def test_pdp_isolate_numeric_feature(self, titanic_model, titanic_data, titanic_features):
        # feature_type: numeric
        pdp_isolate_out = pdp_isolate(
            model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Fare',
            num_grid_points=10,
            grid_type='percentile', percentile_range=None, grid_range=None, cust_grid_points=None, memory_limit=0.5,
            n_jobs=1, predict_kwds={}, data_transformer=None)

        assert pdp_isolate_out._type == 'PDPIsolate_instance'
        assert pdp_isolate_out.n_classes == 2
        assert pdp_isolate_out.which_class is None
        assert pdp_isolate_out.feature == 'Fare'
        assert pdp_isolate_out.feature_type == 'numeric'
        assert_array_almost_equal(
            pdp_isolate_out.feature_grids,
            np.array([0., 7.73284444, 7.8958, 8.6625, 13., 16.7, 26., 35.11111111, 73.5, 512.3292]), decimal=8)
        assert_array_equal(
            pdp_isolate_out.percentile_info, np.array(['(0.0)', '(11.11)', '(22.22)', '(33.33)', '(44.44)', '(55.56)',
                                                       '(66.67)', '(77.78)', '(88.89)', '(100.0)'], dtype=object))
        assert pdp_isolate_out.display_columns == \
               ['0', '7.73', '7.9', '8.66', '13', '16.7', '26', '35.11', '73.5', '512.33']

        ice_lines_expected = pd.DataFrame(
            {7.732844444444444: {0: 0.10785847157239914, 500: 0.10733785480260849, 800: 0.08447999507188797},
             16.7: {0: 0.11272012442350388, 500: 0.1367582231760025, 800: 0.16041819751262665},
             512.3292: {0: 0.22303232550621033, 500: 0.26529940962791443, 800: 0.3425298035144806}})
        assert_frame_equal(
            pdp_isolate_out.ice_lines.loc[[0, 500, 800], pdp_isolate_out.ice_lines.columns.values[[1, 5, 9]]],
            ice_lines_expected, check_like=True, check_dtype=False)

        assert_array_almost_equal(
            pdp_isolate_out.pdp[[1, 5, 9]], np.array([0.34042153, 0.3876946, 0.48316196]), decimal=7)

        count_data_expected = pd.DataFrame({'count': {0: 99, 4: 108, 8: 102},
                                            'count_norm': {0: 0.1111111111111111, 4: 0.12121212121212122,
                                                           8: 0.11447811447811448},
                                            'x': {0: 0, 4: 4, 8: 8},
                                            'xticklabels': {0: '[0, 7.73)', 4: '[13, 16.7)', 8: '[73.5, 512.33]'}})
        assert_frame_equal(
            pdp_isolate_out.count_data.iloc[[0, 4, 8]], count_data_expected, check_like=True, check_dtype=False)

        assert len(pdp_isolate_out.hist_data) == titanic_data.shape[0]
        assert_array_equal(pdp_isolate_out.hist_data[[0, 200, 400, 600, 800]], np.array([7.25, 9.5, 7.925, 27., 13.]))

    def test_pdp_isolate_cust_grid_points(self, titanic_model, titanic_data, titanic_features):
        # use cust_grid_points
        pdp_isolate_out = pdp_isolate(
            model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Fare',
            num_grid_points=10,
            grid_type='percentile', percentile_range=None, grid_range=None, cust_grid_points=range(0, 100, 5),
            memory_limit=0.5, n_jobs=1, predict_kwds={}, data_transformer=None)

        assert pdp_isolate_out._type == 'PDPIsolate_instance'
        assert pdp_isolate_out.n_classes == 2
        assert pdp_isolate_out.which_class is None
        assert pdp_isolate_out.feature == 'Fare'
        assert pdp_isolate_out.feature_type == 'numeric'
        assert_array_equal(pdp_isolate_out.feature_grids, range(0, 100, 5))
        assert pdp_isolate_out.percentile_info == []
        assert pdp_isolate_out.display_columns == ['0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50',
                                                   '55', '60', '65', '70', '75', '80', '85', '90', '95']

        ice_lines_expected = pd.DataFrame(
            {5: {0: 0.08728538453578949, 500: 0.08203054219484329, 800: 0.05646398663520813},
             25: {0: 0.10983016341924667, 500: 0.1353892683982849, 800: 0.1680663526058197},
             45: {0: 0.13062693178653717, 500: 0.15881720185279846, 800: 0.22301127016544342}})
        assert_frame_equal(
            pdp_isolate_out.ice_lines.loc[[0, 500, 800], pdp_isolate_out.ice_lines.columns.values[[1, 5, 9]]],
            ice_lines_expected, check_like=True, check_dtype=False)

        assert_array_almost_equal(pdp_isolate_out.pdp[[1, 5, 9]],
                                  np.array([0.29875883, 0.35867965, 0.38914606]), decimal=8)

        count_data_expected = pd.DataFrame({'count': {0: 16, 5: 94, 10: 20, 15: 21},
                                            'count_norm': {0: 0.017957351290684626, 5: 0.10549943883277217,
                                                           10: 0.02244668911335578, 15: 0.02356902356902357},
                                            'x': {0: 0, 5: 5, 10: 10, 15: 15},
                                            'xticklabels': {0: '[0, 5)', 5: '[25, 30)', 10: '[50, 55)',
                                                            15: '[75, 80)'}})
        assert_frame_equal(pdp_isolate_out.count_data.iloc[[0, 5, 10, 15]], count_data_expected,
                           check_like=True, check_dtype=False)

        assert len(pdp_isolate_out.hist_data) == titanic_data.shape[0]
        assert_array_equal(pdp_isolate_out.hist_data[[0, 200, 400, 600, 800]], np.array([7.25, 9.5, 7.925, 27., 13.]))


class TestPDPIsolateRegression(object):
    def test_pdp_isolate_regression(self, ross_model, ross_data, ross_features):
        pdp_isolate_out = pdp_isolate(
            model=ross_model, dataset=ross_data, model_features=ross_features, feature='SchoolHoliday',
            num_grid_points=10, grid_type='percentile', percentile_range=None, grid_range=None, cust_grid_points=None,
            memory_limit=0.5, n_jobs=1, predict_kwds={}, data_transformer=None)

        assert pdp_isolate_out._type == 'PDPIsolate_instance'
        assert pdp_isolate_out.n_classes == 0
        assert pdp_isolate_out.which_class is None
        assert pdp_isolate_out.feature == 'SchoolHoliday'
        assert pdp_isolate_out.feature_type == 'binary'
        assert_array_equal(pdp_isolate_out.feature_grids, np.array([0, 1]))
        assert pdp_isolate_out.percentile_info == []
        assert pdp_isolate_out.display_columns == ['SchoolHoliday_0', 'SchoolHoliday_1']

        ice_lines_expected = pd.DataFrame(
            {0: {0: 8530.777753293662, 200000: 5482.39135417475, 400000: 11795.332507513558,
                 600000: 5725.370367820668, 800000: 4780.186722072591},
             1: {0: 8802.91008056077, 200000: 5352.321273982288, 400000: 7933.070072150073,
                 600000: 5493.134809064146, 800000: 4877.434213535265}})
        assert_frame_equal(pdp_isolate_out.ice_lines.iloc[[0, 200000, 400000, 600000, 800000]], ice_lines_expected,
                           check_like=True, check_dtype=False)

        assert_array_almost_equal(pdp_isolate_out.pdp, np.array([6956.36677514, 6986.85014497]), decimal=8)

        count_data_expected = pd.DataFrame({'count': {0: 680935, 1: 163457},
                                            'count_norm': {0: 0.8064204776928251, 1: 0.19357952230717487},
                                            'x': {0: 0, 1: 1}})
        assert_frame_equal(pdp_isolate_out.count_data, count_data_expected, check_like=True, check_dtype=False)

        assert pdp_isolate_out.hist_data is None

    def test_pdp_isolate_n_jobs(self, ross_model, ross_data, ross_features):
        # test n_jobs > 1
        _ = pdp_isolate(
            model=ross_model, dataset=ross_data, model_features=ross_features, feature='SchoolHoliday',
            num_grid_points=10, grid_type='percentile', percentile_range=None, grid_range=None, cust_grid_points=None,
            memory_limit=0.5, n_jobs=2, predict_kwds={}, data_transformer=None)


def test_pdp_isolate_multiclass(otto_model, otto_data, otto_features):
    pdp_isolate_out = pdp_isolate(
        model=otto_model, dataset=otto_data, model_features=otto_features, feature='feat_67',
        num_grid_points=10, grid_type='percentile', percentile_range=None, grid_range=None, cust_grid_points=None,
        memory_limit=0.5, n_jobs=1, predict_kwds={}, data_transformer=None)

    assert len(pdp_isolate_out) == 9
    assert pdp_isolate_out[0]._type == 'PDPIsolate_instance'
    assert pdp_isolate_out[0].n_classes == 9

    for i in range(9):
        assert pdp_isolate_out[i].which_class == i

    assert pdp_isolate_out[0].feature == 'feat_67'
    assert pdp_isolate_out[0].feature_type == 'numeric'

    assert_array_equal(pdp_isolate_out[0].feature_grids, np.array([0., 1., 2., 3., 4., 7., 104.]))
    assert_array_equal(pdp_isolate_out[0].percentile_info,
                       np.array(['(0.0, 11.11, 22.22, 33.33)', '(44.44)', '(55.56)', '(66.67)',
                                 '(77.78)', '(88.89)', '(100.0)'], dtype=object))
    assert pdp_isolate_out[0].display_columns == ['0', '1', '2', '3', '4', '7', '104']

    ice_lines_expected_0 = pd.DataFrame({0.0: {0: 0.56, 30000: 0.0, 60000: 0.08},
                                         1.0: {0: 0.56, 30000: 0.01, 60000: 0.04},
                                         2.0: {0: 0.56, 30000: 0.02, 60000: 0.03},
                                         3.0: {0: 0.56, 30000: 0.02, 60000: 0.01},
                                         4.0: {0: 0.57, 30000: 0.01, 60000: 0.01},
                                         7.0: {0: 0.6, 30000: 0.07, 60000: 0.01},
                                         104.0: {0: 0.47, 30000: 0.08, 60000: 0.01}})

    ice_lines_expected_4 = pd.DataFrame({0.0: {0: 0.0, 30000: 0.9, 60000: 0.0},
                                         1.0: {0: 0.0, 30000: 0.84, 60000: 0.0},
                                         2.0: {0: 0.0, 30000: 0.81, 60000: 0.0},
                                         3.0: {0: 0.0, 30000: 0.8, 60000: 0.0},
                                         4.0: {0: 0.0, 30000: 0.77, 60000: 0.0},
                                         7.0: {0: 0.0, 30000: 0.68, 60000: 0.0},
                                         104.0: {0: 0.0, 30000: 0.66, 60000: 0.0}})

    ice_lines_expected_8 = pd.DataFrame({0.0: {0: 0.0, 30000: 0.0, 60000: 0.75},
                                         1.0: {0: 0.0, 30000: 0.01, 60000: 0.85},
                                         2.0: {0: 0.0, 30000: 0.01, 60000: 0.88},
                                         3.0: {0: 0.0, 30000: 0.03, 60000: 0.9},
                                         4.0: {0: 0.0, 30000: 0.03, 60000: 0.91},
                                         7.0: {0: 0.0, 30000: 0.02, 60000: 0.93},
                                         104.0: {0: 0.08, 30000: 0.05, 60000: 0.95}})

    assert_frame_equal(pdp_isolate_out[0].ice_lines.iloc[[0, 30000, 60000]], ice_lines_expected_0,
                       check_like=True, check_dtype=False)
    assert_frame_equal(pdp_isolate_out[4].ice_lines.iloc[[0, 30000, 60000]], ice_lines_expected_4,
                       check_like=True, check_dtype=False)
    assert_frame_equal(pdp_isolate_out[8].ice_lines.iloc[[0, 30000, 60000]], ice_lines_expected_8,
                       check_like=True, check_dtype=False)

    assert_array_almost_equal(pdp_isolate_out[0].pdp,
                              np.array([0.03248117, 0.0330906 , 0.03270371, 0.03500663, 0.03582194,
                                        0.04053024, 0.05414461]), decimal=7)
    assert_array_almost_equal(pdp_isolate_out[4].pdp,
                              np.array([0.04597692, 0.04540612, 0.04516969, 0.04450903, 0.04441773,
                                        0.04367804, 0.02873638]), decimal=8)
    assert_array_almost_equal(pdp_isolate_out[8].pdp,
                              np.array([0.06646805, 0.07285966, 0.07858092, 0.08221888, 0.08401015,
                                        0.0905304, 0.12443566]), decimal=7)

    count_data_expected = pd.DataFrame({'count': {0: 23930, 1: 9878, 2: 6689, 3: 4434, 4: 8985, 5: 7962},
                                        'count_norm': {0: 0.38672872426387406, 1: 0.15963670448301495,
                                                       2: 0.10809980930217525, 3: 0.07165713177542907,
                                                       4: 0.14520508096577137, 5: 0.12867254920973528},
                                        'x': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
                                        'xticklabels': {0: '[0, 1)', 1: '[1, 2)', 2: '[2, 3)',
                                                        3: '[3, 4)', 4: '[4, 7)', 5: '[7, 104]'}})
    assert_frame_equal(pdp_isolate_out[0].count_data, count_data_expected, check_like=True, check_dtype=False)

    assert_array_equal(pdp_isolate_out[0].hist_data[[0, 10000, 20000, 30000, 40000, 50000, 60000]],
                       np.array([7, 0, 0, 0, 1, 4, 10]))


class TestPDPPlotSingle(object):
    @pytest.fixture
    def pdp_sex(self, titanic_data, titanic_model, titanic_features):
        result = pdp_isolate(model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Sex')
        return result

    def test_pdp_plot_single_default(self, pdp_sex):
        # single chart without data dist plot
        fig, axes = pdp_plot(pdp_sex, 'sex')
        assert type(fig) == matplotlib.figure.Figure
        assert sorted(axes.keys()) == ['pdp_ax', 'title_ax']
        assert type(axes['pdp_ax']) == matplotlib.axes._subplots.Subplot
        assert type(axes['title_ax']) == matplotlib.axes._subplots.Subplot

    def test_pdp_plot_single_distplot(self, pdp_sex):
        # single chart with data dist plot
        fig, axes = pdp_plot(pdp_sex, 'sex', plot_pts_dist=True)
        assert sorted(axes.keys()) == ['pdp_ax', 'title_ax']
        assert sorted(axes['pdp_ax'].keys()) == ['_count_ax', '_pdp_ax']
        assert type(axes['pdp_ax']['_pdp_ax']) == matplotlib.axes._subplots.Subplot
        assert type(axes['pdp_ax']['_count_ax']) == matplotlib.axes._subplots.Subplot
        assert type(axes['title_ax']) == matplotlib.axes._subplots.Subplot


class TestPDPPlotMulti(object):
    @pytest.fixture
    def pdp_feat_67_rf(self, otto_data, otto_model, otto_features):
        result = pdp_isolate(model=otto_model, dataset=otto_data, model_features=otto_features, feature='feat_67')
        return result

    def test_pdp_plot_multi_default(self, pdp_feat_67_rf):
        # multi charts without data dist plot
        fig, axes = pdp_plot(pdp_isolate_out=pdp_feat_67_rf, feature_name='feat_67', center=True, x_quantile=True)
        assert type(fig) == matplotlib.figure.Figure
        assert sorted(axes.keys()) == ['pdp_ax', 'title_ax']
        assert len(axes['pdp_ax']) == 9
        assert type(axes['title_ax']) == matplotlib.axes._subplots.Subplot
        assert type(axes['pdp_ax'][0]) == matplotlib.axes._subplots.Subplot

    def test_pdp_plot_multi_which_classes(self, pdp_feat_67_rf):
        # change which classes
        fig, axes = pdp_plot(pdp_feat_67_rf, 'feat_67', center=True, x_quantile=True, ncols=2, which_classes=[0, 3, 7])
        assert len(axes['pdp_ax']) == 3

    def test_pdp_plot_multi_one_class(self, pdp_feat_67_rf):
        # only keep 1 class
        fig, axes = pdp_plot(pdp_feat_67_rf, 'feat_67', center=True, x_quantile=True, ncols=2, which_classes=[5])
        assert type(axes['pdp_ax']) == matplotlib.axes._subplots.Subplot

    def test_pdp_plot_multi_distplot(self, pdp_feat_67_rf):
        # multi charts with data dist plot
        fig, axes = pdp_plot(pdp_isolate_out=pdp_feat_67_rf, feature_name='feat_67', center=True,
                             x_quantile=True, plot_pts_dist=True)
        assert sorted(axes.keys()) == ['pdp_ax', 'title_ax']
        assert len(axes['pdp_ax']) == 9
        assert sorted(axes['pdp_ax'][0].keys()) == ['_count_ax', '_pdp_ax']
        assert type(axes['pdp_ax'][0]['_count_ax']) == matplotlib.axes._subplots.Subplot
        assert type(axes['pdp_ax'][0]['_pdp_ax']) == matplotlib.axes._subplots.Subplot
