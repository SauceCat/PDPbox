
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

        count_data_expected = pd.DataFrame({'count': {0: 835488, 1: 181721},
                                            'count_norm': {0: 0.821353, 1: 0.178647},
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
