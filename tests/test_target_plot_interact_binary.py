from pdpbox.info_plots import target_plot_interact
import pandas as pd
import numpy as np
from numpy import nan
from pandas.testing import assert_frame_equal


def test_binary_onehot(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=['Sex', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
        feature_names=['Sex', 'Embarked'],
        target=titanic_target)

    expected = pd.DataFrame(
        {'x1': {0: 0, 2: 0, 5: 1},
         'x2': {0: 0, 2: 2, 5: 2},
         'display_column_1': {0: 'Sex_0', 2: 'Sex_0', 5: 'Sex_1'},
         'display_column_2': {0: 'Embarked_C', 2: 'Embarked_S', 5: 'Embarked_S'},
         'count': {0: 73, 2: 205, 5: 441},
         'Survived': {0: 0.8767123287671232, 2: 0.6926829268292682, 5: 0.1746031746031746}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 2, 5], :], check_like=True)
    assert summary_df['count'].sum() == len(titanic_data)


def test_onehot_numeric(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=['Fare', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
        feature_names=['Fare', 'Embarked'],
        target=titanic_target)

    expected = pd.DataFrame(
        {'x1': {0: 0, 13: 4, 26: 8},
         'x2': {0: 0, 13: 1, 26: 2},
         'display_column_1': {0: '[0, 7.73)', 13: '[13, 16.7)', 26: '[73.5, 512.33]'},
         'display_column_2': {0: 'Embarked_C', 13: 'Embarked_Q', 26: 'Embarked_S'},
         'value_lower_1': {0: 0.0, 13: 13.0, 26: 73.5},
         'value_upper_1': {0: 7.732844444444444, 13: 16.7, 26: 512.3292},
         'count': {0: 29.0, 13: 8.0, 26: 51.0},
         'Survived': {0: 0.2413793103448276, 13: 0.375, 26: 0.6862745098039216}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 13, 26], :], check_like=True)
    assert summary_df['count'].sum() == len(titanic_data)


def test_onehot_numeric_endpoint(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=['Fare', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
        feature_names=['Fare', 'Embarked'],
        target=titanic_target,
        endpoint=False)

    expected = pd.DataFrame(
        {'x1': {0: 0, 15: 5, 29: 9},
         'x2': {0: 0, 15: 0, 29: 2},
         'display_column_1': {0: '[0, 7.73)', 15: '[16.7, 26)', 29: '>= 512.33'},
         'display_column_2': {0: 'Embarked_C', 15: 'Embarked_C', 29: 'Embarked_S'},
         'value_lower_1': {0: 0.0, 15: 16.7, 29: 512.3292},
         'value_upper_1': {0: 7.732844444444444, 15: 26.0, 29: nan},
         'count': {0: 29.0, 15: 11.0, 29: 0.0},
         'Survived': {0: 0.2413793103448276, 15: 0.7272727272727273, 29: 0.0}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 15, 29], :], check_like=True)
    assert summary_df['count'].sum() == len(titanic_data)


def test_onehot_numeric_num_grid_points(titanic_data, titanic_target):
    fare_grid_points = 15
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=['Fare', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
        feature_names=['Fare', 'Embarked'],
        target=titanic_target,
        num_grid_points=[fare_grid_points, None])

    expected = pd.DataFrame(
        {'x1': {0: 0, 21: 7, 41: 13},
         'x2': {0: 0, 21: 0, 41: 2},
         'display_column_1': {0: '[0, 7.23)', 21: '[14.45, 19.26)', 41: '[86.5, 512.33]'},
         'display_column_2': {0: 'Embarked_C', 21: 'Embarked_C', 41: 'Embarked_S'},
         'value_lower_1': {0: 0.0, 21: 14.4542, 41: 86.5},
         'value_upper_1': {0: 7.2292000000000005, 21: 19.2583, 41: 512.3292},
         'count': {0: 29.0, 21: 21.0, 41: 31.0},
         'Survived': {0: 0.2413793103448276, 21: 0.3333333333333333, 41: 0.8064516129032258}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 21, 41], :], check_like=True)
    assert summary_df['count'].sum() == len(titanic_data)
    assert len(summary_df) == (fare_grid_points - 1) * 3


def test_onehot_numeric_gridtype_equal(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=['Fare', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
        feature_names=['Fare', 'Embarked'],
        target=titanic_target,
        grid_types=['equal', 'equal'])

    expected = pd.DataFrame(
        {'x1': {0: 0, 13: 4, 26: 8},
         'x2': {0: 0, 13: 1, 26: 2},
         'display_column_1': {0: '[0, 56.93)',
                              13: '[227.7, 284.63)',
                              26: '[455.4, 512.33]'},
         'display_column_2': {0: 'Embarked_C', 13: 'Embarked_Q', 26: 'Embarked_S'},
         'value_lower_1': {0: 0.0, 13: 227.70186666666666, 26: 455.4037333333333},
         'value_upper_1': {0: 56.925466666666665,
                           13: 284.62733333333335,
                           26: 512.3292},
         'count': {0: 108.0, 13: 0.0, 26: 0.0},
         'Survived': {0: 0.42592592592592593, 13: 0.0, 26: 0.0}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 13, 26], :], check_like=True)
    assert summary_df['count'].sum() == len(titanic_data)

    # all bins should have the same width (equal grid type)
    assert (summary_df.value_upper_1 - summary_df.value_lower_1).diff().sum() < 1e-9


def test_onehot_numeric_percentile(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=['Fare', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
        feature_names=['Fare', 'Embarked'],
        target=titanic_target,
        percentile_ranges=[(5, 95), None])

    expected = pd.DataFrame(
        {'x1': {0: 0, 13: 4, 26: 8},
         'x2': {0: 0, 13: 1, 26: 2},
         'display_column_1': {0: '[7.22, 7.75)', 13: '[13, 16.1)', 26: '[56.5, 112.08]'},
         'display_column_2': {0: 'Embarked_C', 13: 'Embarked_Q', 26: 'Embarked_S'},
         'value_lower_1': {0: 7.225, 13: 13.0, 26: 56.4958},
         'value_upper_1': {0: 7.75, 13: 16.1, 26: 112.07915},
         'count': {0: 27.0, 13: 8.0, 26: 50.0},
         'Survived': {0: 0.25925925925925924, 13: 0.375, 26: 0.56}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 13, 26], :], check_like=True)

    fare = titanic_data['Fare']
    inside_percentile = ((fare >= fare.quantile(0.05)) & (fare <= fare.quantile(0.95)))
    assert summary_df['count'].sum() == inside_percentile.sum()


def test_onehot_numeric_gridranges(titanic_data, titanic_target):
    """
    Grid type must be 'equal' for grid ranges to work
    TODO: maybe this should be automatic or at least warn the user when grid types not 'equal'
    """
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=['Fare', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
        feature_names=['Fare', 'Embarked'],
        target=titanic_target,
        grid_types=['equal', 'equal'],
        grid_ranges=[(0, 100), None])

    expected = pd.DataFrame(
        {'x1': {0: 0, 13: 4, 26: 8},
         'x2': {0: 0, 13: 1, 26: 2},
         'display_column_1': {0: '[0, 11.11)', 13: '[44.44, 55.56)', 26: '[88.89, 100]'},
         'display_column_2': {0: 'Embarked_C', 13: 'Embarked_Q', 26: 'Embarked_S'},
         'value_lower_1': {0: 0.0, 13: 44.44444444444444, 26: 88.88888888888889},
         'value_upper_1': {0: 11.11111111111111, 13: 55.55555555555556, 26: 100.0},
         'count': {0: 37.0, 13: 0.0, 26: 4.0},
         'Survived': {0: 0.24324324324324326, 13: 0.0, 26: 1.0}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 13, 26], :], check_like=True)

    # counts must total to the total observations inside the defined range
    inside_range = (titanic_data['Fare'] >= 0) & (titanic_data['Fare'] <= 100)
    assert summary_df['count'].sum() == inside_range.sum()

    # first and last values must be equal to the defined ranges
    assert summary_df.groupby('x2').first().value_lower_1.unique()[0] == 0.0
    assert summary_df.groupby('x2').last().value_upper_1.unique()[0] == 100.0


def test_onehot_numeric_gridranges_outliers(titanic_data, titanic_target):
    """
    'show_outliers' implies 'grid_ranges' or another custom grid definition
    TODO: again, this should be explicit to the user
    """
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=['Fare', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
        feature_names=['Fare', 'Embarked'],
        target=titanic_target,
        grid_types=['equal', 'equal'],
        grid_ranges=[(0, 100), None],
        show_outliers=True)

    expected = pd.DataFrame(
        {'x1': {0: 0, 15: 5, 29: 9},
         'x2': {0: 0, 15: 0, 29: 2},
         'display_column_1': {0: '[0, 11.11)', 15: '[55.56, 66.67)', 29: '> 100'},
         'display_column_2': {0: 'Embarked_C', 15: 'Embarked_C', 29: 'Embarked_S'},
         'value_lower_1': {0: 0.0, 15: 55.55555555555556, 29: 100.0},
         'value_upper_1': {0: 11.11111111111111, 15: 66.66666666666666, 29: nan},
         'count': {0: 37.0, 15: 8.0, 29: 24.0},
         'Survived': {0: 0.24324324324324326, 15: 0.75, 29: 0.75}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 15, 29], :], check_like=True)
    assert summary_df['count'].sum() == len(titanic_data)


def test_onehot_numeric_gridranges_outliers_endpoint(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=['Fare', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
        feature_names=['Fare', 'Embarked'],
        target=titanic_target,
        grid_types=['equal', 'equal'],
        grid_ranges=[(0, 100), None],
        show_outliers=True,
        endpoint=False)

    expected = pd.DataFrame(
        {'x1': {0: 0, 15: 5, 29: 9},
         'x2': {0: 0, 15: 0, 29: 2},
         'display_column_1': {0: '[0, 11.11)', 15: '[55.56, 66.67)', 29: '>= 100'},
         'display_column_2': {0: 'Embarked_C', 15: 'Embarked_C', 29: 'Embarked_S'},
         'value_lower_1': {0: 0.0, 15: 55.55555555555556, 29: 100.0},
         'value_upper_1': {0: 11.11111111111111, 15: 66.66666666666666, 29: nan},
         'count': {0: 37.0, 15: 8.0, 29: 24.0},
         'Survived': {0: 0.24324324324324326, 15: 0.75, 29: 0.75}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 15, 29], :], check_like=True)
    assert summary_df['count'].sum() == len(titanic_data)


def test_onehot_numeric_cust_grid_points(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=['Fare', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
        feature_names=['Fare', 'Embarked'],
        target=titanic_target,
        cust_grid_points=[range(0, 100, 10), None])

    expected = pd.DataFrame(
        {'x1': {0: 0, 13: 4, 26: 8},
         'x2': {0: 0, 13: 1, 26: 2},
         'display_column_1': {0: '[0, 10)', 13: '[40, 50)', 26: '[80, 90]'},
         'display_column_2': {0: 'Embarked_C', 13: 'Embarked_Q', 26: 'Embarked_S'},
         'value_lower_1': {0: 0.0, 13: 40.0, 26: 80.0},
         'value_upper_1': {0: 10.0, 13: 50.0, 26: 90.0},
         'count': {0: 37.0, 13: 0.0, 26: 10.0},
         'Survived': {0: 0.24324324324324326, 13: 0.0, 26: 0.9}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 13, 26], :], check_like=True)

    # counts must total to the total observations inside the defined range
    inside_range = (titanic_data['Fare'] >= 0) & (titanic_data['Fare'] <= 90)
    assert summary_df['count'].sum() == inside_range.sum()

    # lower and upper values must follow the prescribed grid
    assert (summary_df.groupby('x1').value_lower_1.mean().values == np.arange(0., 90, 10)).all()
    assert (summary_df.groupby('x1').value_upper_1.mean().values == np.arange(10., 100, 10)).all()


def test_onehot_numeric_gridpoints_outliers(titanic_data, titanic_target):
    """
    'show_outliers' implies 'custom_grid_points' or another custom grid definition
    TODO: again, this should be explicit to the user
    """
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=['Fare', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
        feature_names=['Fare', 'Embarked'],
        target=titanic_target,
        cust_grid_points=[range(0, 100, 10), None],
        show_outliers=True)

    expected = pd.DataFrame(
        {'x1': {0: 0, 15: 5, 29: 9},
         'x2': {0: 0, 15: 0, 29: 2},
         'display_column_1': {0: '[0, 10)', 15: '[50, 60)', 29: '> 90'},
         'display_column_2': {0: 'Embarked_C', 15: 'Embarked_C', 29: 'Embarked_S'},
         'value_lower_1': {0: 0.0, 15: 50.0, 29: 90.0},
         'value_upper_1': {0: 10.0, 15: 60.0, 29: nan},
         'count': {0: 37.0, 15: 6.0, 29: 26.0},
         'Survived': {0: 0.24324324324324326, 15: 1.0, 29: 0.7692307692307693}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 15, 29], :], check_like=True)
    assert summary_df['count'].sum() == len(titanic_data)


def test_onehot_numeric_gridpoints_outliers_endpoint(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=['Fare', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
        feature_names=['Fare', 'Embarked'],
        target=titanic_target,
        cust_grid_points=[range(0, 100, 10), None],
        show_outliers=True,
        endpoint=False)

    expected = pd.DataFrame(
        {'x1': {0: 0, 15: 5, 29: 9},
         'x2': {0: 0, 15: 0, 29: 2},
         'display_column_1': {0: '[0, 10)', 15: '[50, 60)', 29: '>= 90'},
         'display_column_2': {0: 'Embarked_C', 15: 'Embarked_C', 29: 'Embarked_S'},
         'value_lower_1': {0: 0.0, 15: 50.0, 29: 90.0},
         'value_upper_1': {0: 10.0, 15: 60.0, 29: nan},
         'count': {0: 37.0, 15: 6.0, 29: 28.0},
         'Survived': {0: 0.24324324324324326, 15: 1.0, 29: 0.7857142857142857}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 15, 29], :], check_like=True)
    assert summary_df['count'].sum() == len(titanic_data)


def test_onehot_numeric_show_percentiles(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=['Fare', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
        feature_names=['Fare', 'Embarked'],
        target=titanic_target,
        percentile_ranges=[(5, 95), None],
        show_percentile=True)

    expected = pd.DataFrame(
        {'x1': {0: 0, 13: 4, 26: 8},
         'x2': {0: 0, 13: 1, 26: 2},
         'display_column_1': {0: '[7.22, 7.75)', 13: '[13, 16.1)', 26: '[56.5, 112.08]'},
         'display_column_2': {0: 'Embarked_C', 13: 'Embarked_Q', 26: 'Embarked_S'},
         'value_lower_1': {0: 7.225, 13: 13.0, 26: 56.4958},
         'value_upper_1': {0: 7.75, 13: 16.1, 26: 112.07915},
         'percentile_column_1': {0: '[5, 15)', 13: '[45, 55)', 26: '[85, 95]'},
         'percentile_lower_1': {0: 5.0, 13: 45.0, 26: 85.0},
         'percentile_upper_1': {0: 15.0, 13: 55.0, 26: 95.0},
         'count': {0: 27.0, 13: 8.0, 26: 50.0},
         'Survived': {0: 0.25925925925925924, 13: 0.375, 26: 0.56}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 13, 26], :], check_like=True)

    fare = titanic_data['Fare']
    inside_percentile = ((fare >= fare.quantile(0.05)) & (fare <= fare.quantile(0.95)))
    assert summary_df['count'].sum() == inside_percentile.sum()


def test_onehot_numeric_show_outliers(titanic_data, titanic_target):
    """
    'show_outliers' implies 'percentile_ranges' or another custom grid definition
    TODO: again, this should be explicit to the user
    """
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=['Fare', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
        feature_names=['Fare', 'Embarked'],
        target=titanic_target,
        percentile_ranges=[(5, 95), None],
        show_percentile=True,
        show_outliers=True)

    expected = pd.DataFrame(
        {'x1': {0: 0, 16: 5, 32: 10},
         'x2': {0: 0, 16: 1, 32: 2},
         'display_column_1': {0: '< 7.22', 16: '[13, 16.1)', 32: '> 112.08'},
         'display_column_2': {0: 'Embarked_C', 16: 'Embarked_Q', 32: 'Embarked_S'},
         'value_lower_1': {0: nan, 16: 13.0, 32: 112.07915},
         'value_upper_1': {0: 7.225, 16: 16.1, 32: nan},
         'percentile_column_1': {0: '< 5', 16: '[45, 55)', 32: '> 95'},
         'percentile_lower_1': {0: 0.0, 16: 45.0, 32: 95.0},
         'percentile_upper_1': {0: 5.0, 16: 55.0, 32: 100.0},
         'count': {0: 2.0, 16: 8.0, 32: 24.0},
         'Survived': {0: 0.0, 16: 0.375, 32: 0.75}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 16, 32], :], check_like=True)

    assert summary_df['count'].sum() == len(titanic_data)


def test_onehot_numeric_show_outliers_endpoint(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=['Fare', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
        feature_names=['Fare', 'Embarked'],
        target=titanic_target,
        percentile_ranges=[(5, 95), None],
        show_percentile=True,
        show_outliers=True,
        endpoint=False)

    expected = pd.DataFrame(
        {'x1': {0: 0, 16: 5, 32: 10},
         'x2': {0: 0, 16: 1, 32: 2},
         'display_column_1': {0: '< 7.22', 16: '[13, 16.1)', 32: '>= 112.08'},
         'display_column_2': {0: 'Embarked_C', 16: 'Embarked_Q', 32: 'Embarked_S'},
         'value_lower_1': {0: nan, 16: 13.0, 32: 112.07915},
         'value_upper_1': {0: 7.225, 16: 16.1, 32: nan},
         'percentile_column_1': {0: '< 5', 16: '[45, 55)', 32: '>= 95'},
         'percentile_lower_1': {0: 0.0, 16: 45.0, 32: 95.0},
         'percentile_upper_1': {0: 5.0, 16: 55.0, 32: 100.0},
         'count': {0: 2.0, 16: 8.0, 32: 24.0},
         'Survived': {0: 0.0, 16: 0.375, 32: 0.75}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 16, 32], :], check_like=True)

    assert summary_df['count'].sum() == len(titanic_data)


def test_binary_numeric(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=['Fare', 'Sex'],
        feature_names=['Fare', 'Sex'],
        target=titanic_target,
        show_percentile=True,
        percentile_ranges=[(5, 95), None],
        show_outliers=True)

    expected = pd.DataFrame(
        {'x1': {0: 0, 11: 5, 21: 10},
         'x2': {0: 0, 11: 1, 21: 1},
         'display_column_1': {0: '< 7.22', 11: '[13, 16.1)', 21: '> 112.08'},
         'display_column_2': {0: 'Sex_0', 11: 'Sex_1', 21: 'Sex_1'},
         'value_lower_1': {0: nan, 11: 13.0, 21: 112.07915},
         'value_upper_1': {0: 7.225, 11: 16.1, 21: nan},
         'percentile_column_1': {0: '< 5', 11: '[45, 55)', 21: '> 95'},
         'percentile_lower_1': {0: 0.0, 11: 45.0, 21: 95.0},
         'percentile_upper_1': {0: 5.0, 11: 55.0, 21: 100.0},
         'count': {0: 1, 11: 59, 21: 15},
         'Survived': {0: 0.0, 11: 0.1864406779661017, 21: 0.4}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 11, 21], :], check_like=True)

    assert summary_df['count'].sum() == len(titanic_data)
