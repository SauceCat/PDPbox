
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pandas as pd
from pdpbox.utils import _check_feature, _check_target, _get_grids, _get_grid_combos, \
    _make_bucket_column_names, _make_bucket_column_names_percentile


class TestCheckFeature(object):
    def test_feature_binary(self, titanic_data):
        feature_type = _check_feature(feature='Sex', df=titanic_data)
        assert feature_type == 'binary'

    def test_feature_onehot(self, titanic_data):
        feature_type = _check_feature(feature=['Embarked_C', 'Embarked_S', 'Embarked_Q'], df=titanic_data)
        assert feature_type == 'onehot'

    def test_feature_numeric(self, titanic_data):
        feature_type = _check_feature(feature='Fare', df=titanic_data)
        assert feature_type == 'numeric'

    def test_feature_not_exist(self, titanic_data):
        with pytest.raises(ValueError):
            _ = _check_feature(feature='gender', df=titanic_data)

    def test_feature_onehot_not_exist(self, titanic_data):
        with pytest.raises(ValueError):
            _ = _check_feature(feature=['Embarked_C', 'Embarked_S', 'Embarked_Q', 'Embarked_F'], df=titanic_data)

    def test_feature_onehot_incomplete(self, titanic_data):
        with pytest.raises(ValueError):
            _ = _check_feature(feature=['Embarked_C'], df=titanic_data)


def test_percentile_range():
    from pdpbox.utils import _check_percentile_range

    with pytest.raises(ValueError):
        _check_percentile_range(percentile_range=5)
        _check_percentile_range(percentile_range=(5))
        _check_percentile_range(percentile_range=(5, 105))
        _check_percentile_range(percentile_range=(5, 105, 110))


# @pytest.mark.skip(reason="slow")
class TestCheckTarget(object):
    def test_target_binary(self, titanic_data):
        target_type = _check_target(target='Survived', df=titanic_data)
        assert target_type == 'binary'

    def test_target_regression(self, ross_data):
        target_type = _check_target(target='Sales', df=ross_data)
        assert target_type == 'regression'

    def test_target_multi_class(self, otto_data):
        target_list = ['target_0', 'target_1', 'target_2', 'target_3', 'target_4',
                       'target_5', 'target_6', 'target_7', 'target_8']
        target_type = _check_target(target=target_list, df=otto_data)
        assert target_type == 'multi-class'

    def test_target_multi_class_not_exist(self, otto_data):
        with pytest.raises(ValueError):
            _ = _check_target(target=['target'], df=otto_data)

    def test_target_multi_class_outbound(self, otto_data):
        with pytest.raises(ValueError):
            _ = _check_target(target=['target_9'], df=otto_data)

    def test_target_not_exist(self, titanic_data):
        with pytest.raises(ValueError):
            _ = _check_target(target='survived', df=titanic_data)


def test_check_dataset():
    from pdpbox.utils import _check_dataset

    with pytest.raises(ValueError):
        _check_dataset(df=np.random.rand(5, 5))


def test_make_list():
    from pdpbox.utils import _make_list

    assert _make_list([1, 2]) == [1, 2]
    assert _make_list(1) == [1]


def test_expand_default():
    from pdpbox.utils import _expand_default

    assert _expand_default(x=None, default=10) == [10, 10]
    assert _expand_default(x=[1, 2], default=10) == [1, 2]


# @pytest.mark.skip(reason="slow")
def test_check_model(titanic_model, otto_model, ross_model):
    from pdpbox.utils import _check_model

    assert (_check_model(model=titanic_model)) == (2, titanic_model.predict_proba)
    assert (_check_model(model=otto_model)) == (9, otto_model.predict_proba)
    assert (_check_model(model=ross_model)) == (0, ross_model.predict)


def test_check_grid_type():
    from pdpbox.utils import _check_grid_type

    with pytest.raises(ValueError):
        _check_grid_type(grid_type='quantile')


def test_check_classes():
    from pdpbox.utils import _check_classes

    with pytest.raises(ValueError):
        _check_classes(classes_list=[1, 2, 3], n_classes=3)
        _check_classes(classes_list=[-1, 2], n_classes=3)


def test_check_memory_limit():
    from pdpbox.utils import _check_memory_limit

    with pytest.raises(ValueError):
        _check_memory_limit(memory_limit=1)
        _check_memory_limit(memory_limit=5)
        _check_memory_limit(memory_limit=-1)


def test_check_frac_to_plot():
    from pdpbox.utils import _check_frac_to_plot

    with pytest.raises(ValueError):
        _check_frac_to_plot(frac_to_plot=1.5)
        _check_frac_to_plot(frac_to_plot=0)
        _check_frac_to_plot(frac_to_plot=-1)
        _check_frac_to_plot(frac_to_plot='1')


def test_calc_memory_usage(titanic_data):
    from pdpbox.utils import _calc_memory_usage

    assert _calc_memory_usage(df=titanic_data, total_units=10, n_jobs=1, memory_limit=0.5) == 1


class TestGetGrids(object):
    def test_grids_default(self, titanic_data):
        # default setting
        feature_grids, percentile_info = _get_grids(
            feature_values=titanic_data['Fare'].values, num_grid_points=10, grid_type='percentile',
            percentile_range=None, grid_range=None)

        expected_feature_grids = np.array([0.0, 13.0, 35.11111111])
        expected_percentile_info = np.array(['(0.0)', '(44.44)', '(77.78)'])

        assert_array_almost_equal(feature_grids[[0, 4, 7]], expected_feature_grids, decimal=8)
        assert_array_equal(percentile_info[[0, 4, 7]], expected_percentile_info)

    def test_grids_15(self, titanic_data):
        # num_grid_points=15
        feature_grids, percentile_info = _get_grids(
            feature_values=titanic_data['Fare'].values, num_grid_points=15, grid_type='percentile',
            percentile_range=None, grid_range=None)

        expected_feature_grids = np.array([0.0, 8.05, 14.4542, 37.0042])
        expected_percentile_info = np.array(['(0.0)', '(28.57)', '(50.0)', '(78.57)'])

        assert_array_equal(feature_grids[[0, 4, 7, 11]], expected_feature_grids)
        assert_array_equal(percentile_info[[0, 4, 7, 11]], expected_percentile_info)

    def test_grids_percentile_range(self, titanic_data):
        # percentile_range=(5, 95)
        feature_grids, percentile_info = _get_grids(
            feature_values=titanic_data['Fare'].values, num_grid_points=10, grid_type='percentile',
            percentile_range=(5, 95), grid_range=None)

        expected_feature_grids = np.array([7.225, 13.0, 31.0])
        expected_percentile_info = np.array(['(5.0)', '(45.0)', '(75.0)'])

        assert_array_equal(feature_grids[[0, 4, 7]], expected_feature_grids)
        assert_array_equal(percentile_info[[0, 4, 7]], expected_percentile_info)

    def test_grids_grid_type(self, titanic_data):
        # grid_type='equal'
        feature_grids, percentile_info = _get_grids(
            feature_values=titanic_data['Fare'].values, num_grid_points=10, grid_type='equal',
            percentile_range=None, grid_range=None)

        expected_feature_grids = np.array([0.0, 227.70186666, 398.47826666])
        expected_percentile_info = []

        assert_array_almost_equal(feature_grids[[0, 4, 7]], expected_feature_grids, decimal=8)
        assert percentile_info == expected_percentile_info

    def test_grids_grid_range_percentile(self, titanic_data):
        # grid_range=(0, 100), grid_type='percentile'
        feature_grids, percentile_info = _get_grids(
            feature_values=titanic_data['Fare'].values, num_grid_points=10, grid_type='percentile',
            percentile_range=None, grid_range=(0, 100))

        expected_feature_grids = np.array([0.0, 13.0, 35.11111111])
        expected_percentile_info = np.array(['(0.0)', '(44.44)', '(77.78)'])

        assert_array_almost_equal(feature_grids[[0, 4, 7]], expected_feature_grids, decimal=8)
        assert_array_equal(percentile_info[[0, 4, 7]], expected_percentile_info)

    def test_grids_grid_range_equal(self, titanic_data):
        # grid_range=(0, 100), grid_type='equal'
        feature_grids, percentile_info = _get_grids(
            feature_values=titanic_data['Fare'].values, num_grid_points=10, grid_type='equal',
            percentile_range=None, grid_range=(0, 100))

        expected_feature_grids = np.array([0.0, 44.44444444, 77.77777777])
        expected_percentile_info = []

        assert_array_almost_equal(feature_grids[[0, 4, 7]], expected_feature_grids, decimal=8)
        assert percentile_info == expected_percentile_info


class TestGetGridCombos(object):
    def test_grid_combo_binary_binary(self):
        grid_combos = _get_grid_combos(feature_grids=[[0, 1], [0, 1]], feature_types=['binary', 'binary'])
        assert_array_equal(grid_combos, np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))

    def test_grid_combo_binary_onehot(self):
        grid_combos = _get_grid_combos(feature_grids=[[0, 1], ['a', 'b', 'c']], feature_types=['binary', 'onehot'])
        assert_array_equal(grid_combos, np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                                                  [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]]))

    def test_grid_combo_binary_numeric(self):
        grid_combos = _get_grid_combos(feature_grids=[[0, 1], [1, 2, 3]], feature_types=['binary', 'numeric'])
        assert_array_equal(grid_combos, np.array([[0, 1], [0, 2], [0, 3],
                                                  [1, 1], [1, 2], [1, 3]]))

    def test_grid_combo_onehot_onehot(self):
        grid_combos = _get_grid_combos(feature_grids=[['one', 'two'], ['a', 'b', 'c']],
                                       feature_types=['onehot', 'onehot'])
        assert_array_equal(grid_combos, np.array([[1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1],
                                                  [0, 1, 1, 0, 0], [0, 1, 0, 1, 0], [0, 1, 0, 0, 1]]))

    def test_grid_combo_onehot_binary(self):
        grid_combos = _get_grid_combos(feature_grids=[['one', 'two'], [0, 1]], feature_types=['onehot', 'binary'])
        assert_array_equal(grid_combos, np.array([[1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1]]))

    def test_grid_combo_onehot_numeric(self):
        grid_combos = _get_grid_combos(feature_grids=[['one', 'two'], [1, 2, 3]], feature_types=['onehot', 'numeric'])
        assert_array_equal(grid_combos, np.array([[1, 0, 1], [1, 0, 2], [1, 0, 3],
                                                  [0, 1, 1], [0, 1, 2], [0, 1, 3]]))

    def test_grid_combo_numeric_numeric(self):
        grid_combos = _get_grid_combos(feature_grids=[[-1, -2], [1, 2, 3]], feature_types=['numeric', 'numeric'])
        assert_array_equal(grid_combos, np.array([[-1, 1], [-1, 2], [-1, 3],
                                                  [-2, 1], [-2, 2], [-2, 3]]))

    def test_grid_combo_numeric_binary(self):
        grid_combos = _get_grid_combos(feature_grids=[[-1, -2], [0, 1]], feature_types=['numeric', 'binary'])
        assert_array_equal(grid_combos, np.array([[-1, 0], [-1, 1], [-2, 0], [-2, 1]]))

    def test_grid_combo_numeric_onehot(self):
        grid_combos = _get_grid_combos(feature_grids=[[-1, -2], ['a', 'b', 'c']], feature_types=['numeric', 'onehot'])
        assert_array_equal(grid_combos, np.array([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1],
                                                  [-2, 1, 0, 0], [-2, 0, 1, 0], [-2, 0, 0, 1]]))


def test_sample_data():
    from pdpbox.utils import _sample_data

    fake_ice_lines = pd.DataFrame(np.arange(100), columns=['line'])

    ice_plot_data = _sample_data(ice_lines=fake_ice_lines, frac_to_plot=10)
    assert ice_plot_data.shape[0] == 10
    assert_array_equal(ice_plot_data.index.values, np.arange(ice_plot_data.shape[0]))

    ice_plot_data = _sample_data(ice_lines=fake_ice_lines, frac_to_plot=0.3)
    assert ice_plot_data.shape[0] == 30
    assert_array_equal(ice_plot_data.index.values, np.arange(ice_plot_data.shape[0]))

    ice_plot_data = _sample_data(ice_lines=fake_ice_lines, frac_to_plot=1)
    assert ice_plot_data.shape[0] == 100
    assert_array_equal(ice_plot_data.index.values, np.arange(ice_plot_data.shape[0]))


def test_find_onehot_actual():
    from pdpbox.utils import _find_onehot_actual

    assert _find_onehot_actual(x=[0, 1, 0]) == 1
    assert np.isnan(_find_onehot_actual(x=[0, 0, 0]))


def test_find_bucket():
    from pdpbox.utils import _find_bucket

    assert _find_bucket(x=1, feature_grids=[2, 3, 4], endpoint=True) == 0
    assert _find_bucket(x=2, feature_grids=[2, 3, 4], endpoint=True) == 1
    assert _find_bucket(x=3, feature_grids=[2, 3, 4], endpoint=True) == 2
    assert _find_bucket(x=4, feature_grids=[2, 3, 4], endpoint=True) == 2
    assert _find_bucket(x=4, feature_grids=[2, 3, 4], endpoint=False) == 3
    assert _find_bucket(x=5, feature_grids=[2, 3, 4], endpoint=True) == 3


def test_get_string():
    from pdpbox.utils import _get_string

    assert _get_string(x=1.0) == '1'
    assert _get_string(x=1.1) == '1.1'
    assert _get_string(x=1.12) == '1.12'
    assert _get_string(x=1.123) == '1.12'


class TestMakeBucketColumnNames(object):
    def test_bucket_default(self):
        # default settings
        column_names, bound_lows, bound_ups = _make_bucket_column_names(feature_grids=[1, 2, 3], endpoint=True)
        assert column_names == ['< 1', '[1, 2)', '[2, 3]', '> 3']
        assert bound_lows == [np.nan, 1, 2, 3]
        assert bound_ups == [1, 2, 3, np.nan]

    def test_bucket_no_endpoint(self):
        # without endpoint
        column_names, bound_lows, bound_ups = _make_bucket_column_names(feature_grids=[1, 2, 3], endpoint=False)
        assert column_names == ['< 1', '[1, 2)', '[2, 3)', '>= 3']
        assert bound_lows == [np.nan, 1, 2, 3]
        assert bound_ups == [1, 2, 3, np.nan]

    def test_bucket_float_0(self):
        # float grids type with 0 digit after decimal point
        column_names, bound_lows, bound_ups = _make_bucket_column_names(feature_grids=[1., 2., 3.], endpoint=True)
        assert column_names == ['< 1', '[1, 2)', '[2, 3]', '> 3']
        assert bound_lows == [np.nan, 1., 2., 3.]
        assert bound_ups == [1., 2., 3., np.nan]

    def test_bucket_float_1(self):
        # float grids type with 1 digit after decimal point
        column_names, bound_lows, bound_ups = _make_bucket_column_names(feature_grids=[1.2, 2.5, 3.5],
                                                                        endpoint=True)
        assert column_names == ['< 1.2', '[1.2, 2.5)', '[2.5, 3.5]', '> 3.5']
        assert bound_lows == [np.nan, 1.2, 2.5, 3.5]
        assert bound_ups == [1.2, 2.5, 3.5, np.nan]

    def test_bucket_float_more(self):
        # float grids type with more than 2 digits after decimal point
        # all round to 2
        column_names, bound_lows, bound_ups = _make_bucket_column_names(feature_grids=[1.234, 2.54322, 3.54332],
                                                                        endpoint=True)
        assert column_names == ['< 1.23', '[1.23, 2.54)', '[2.54, 3.54]', '> 3.54']
        assert bound_lows == [np.nan, 1.234, 2.54322, 3.54332]
        assert bound_ups == [1.234, 2.54322, 3.54332, np.nan]

    def test_bucket_value_close(self):
        # float grids type with 2 digit after decimal point
        # grid points very close in value
        column_names, bound_lows, bound_ups = _make_bucket_column_names(feature_grids=[1.234, 2.54322, 2.54332],
                                                                        endpoint=True)
        assert column_names == ['< 1.23', '[1.23, 2.54)', '[2.54, 2.54]', '> 2.54']
        assert bound_lows == [np.nan, 1.234, 2.54322, 2.54332]
        assert bound_ups == [1.234, 2.54322, 2.54332, np.nan]


class TestMakeBucketColumnNamesPercentile(object):
    def test_bucket_percentile_default(self):
        results = _make_bucket_column_names_percentile(
            percentile_info=np.array(['(0.0, 10.0)', '(28.57)', '(50.0, 60.0, 70.0)', '(78.57)']), endpoint=True)
        assert results[0] == ['< 0', '[0, 28.57)', '[28.57, 70)', '[70, 78.57]', '> 78.57']
        assert results[1] == [0, 0.0, 28.57, 70.0, 78.57]
        assert results[2] == [0.0, 28.57, 70.0, 78.57, 100]

    def test_bucket_percentile_no_endpoint(self):
        results = _make_bucket_column_names_percentile(
            percentile_info=np.array(['(0.0, 10.0)', '(28.57)', '(50.0, 60.0, 70.0)', '(78.57)']), endpoint=False)
        assert results[0] == ['< 0', '[0, 28.57)', '[28.57, 70)', '[70, 78.57)', '>= 78.57']
        assert results[1] == [0, 0.0, 28.57, 70.0, 78.57]
        assert results[2] == [0.0, 28.57, 70.0, 78.57, 100]


def test_calc_figsize():
    from pdpbox.utils import _calc_figsize

    assert (_calc_figsize(num_charts=1, ncols=2, title_height=2, unit_figsize=(7, 7))) == (7, 9, 1, 1)
    assert (_calc_figsize(num_charts=2, ncols=2, title_height=2, unit_figsize=(8, 7))) == (15, 9, 1, 2)
    assert (_calc_figsize(num_charts=3, ncols=2, title_height=2, unit_figsize=(8, 7))) == (15, 16, 2, 2)
    assert (_calc_figsize(num_charts=2, ncols=3, title_height=2, unit_figsize=(8, 7))) == (15, 9, 1, 2)


# no idea how to test plot function yet
# def _plot_title(title, subtitle, title_ax, plot_params)
# def _axes_modify(font_family, ax, top=False, right=False, grid=False)
