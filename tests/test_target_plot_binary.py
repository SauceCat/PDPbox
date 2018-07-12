from pdpbox.info_plots import target_plot
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal


def test_binary(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot(df=titanic_data,
                                        feature='Sex',
                                        feature_name='Sex',
                                        target=titanic_target)

    expected = pd.DataFrame(
        {'x': {0: 0, 1: 1},
         'display_column': {0: 'Sex_0', 1: 'Sex_1'},
         'count': {0: 314, 1: 577},
         'Survived': {0: 0.7420382165605095, 1: 0.18890814558058924}}
    )

    assert_frame_equal(expected, summary_df, check_like=True)


def test_onehot(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot(df=titanic_data,
                                        feature=['Embarked_C', 'Embarked_Q',
                                                 'Embarked_S'],
                                        feature_name='Embarked',
                                        target=titanic_target)

    expected = pd.DataFrame(
        {'x': {0: 0, 1: 1, 2: 2},
         'display_column': {0: 'Embarked_C', 1: 'Embarked_Q', 2: 'Embarked_S'},
         'count': {0: 168, 1: 77, 2: 646},
         'Survived': {0: 0.5535714285714286,
                      1: 0.38961038961038963,
                      2: 0.33900928792569657}}
    )

    assert_frame_equal(expected, summary_df, check_like=True)


def test_numeric(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot(df=titanic_data,
                                        feature='Fare',
                                        feature_name='Fare',
                                        target=titanic_target)

    expected = pd.DataFrame(
        {'x': {0: 0, 4: 4, 7: 7},
         'display_column': {0: '[0, 7.73)',
                            4: '[13, 16.7)',
                            7: '[35.11, 73.5)'},
         'value_lower': {0: 0.0, 4: 13.0, 7: 35.111111111111086},
         'value_upper': {0: 7.732844444444444, 4: 16.7, 7: 73.5},
         'count': {0: 99, 4: 108, 7: 96},
         'Survived': {0: 0.1414141414141414,
                      4: 0.37037037037037035,
                      7: 0.5104166666666666}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 4, 7], :], check_like=True)
    assert len(summary_df) == 9


def test_endpoint(titanic_data, titanic_target):
    """
    test endpoint==False (last point should't be included)
    """
    fig, axes, summary_df = target_plot(df=titanic_data,
                                        feature='Fare',
                                        feature_name='Fare',
                                        target=titanic_target,
                                        endpoint=False)

    expected = pd.DataFrame(
        {'x': {0: 0, 8: 8, 9: 9},
         'display_column': {0: '[0, 7.73)', 8: '[73.5, 512.33)',
                            9: '>= 512.33'},
         'value_lower': {0: 0.0, 8: 73.5, 9: 512.3292},
         'value_upper': {0: 7.732844444444444, 8: 512.3292, 9: np.nan},
         'count': {0: 99, 8: 99, 9: 3},
         'Survived': {0: 0.1414141414141414, 8: 0.7171717171717171, 9: 1.0}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 8, 9], :], check_like=True)
    assert len(summary_df) == 10


def test_num_grid_points(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot(df=titanic_data,
                                        feature='Fare',
                                        feature_name='Fare',
                                        target=titanic_target,
                                        num_grid_points=20)

    expected = pd.DataFrame(
        {'x': {0: 0, 9: 9, 18: 18},
         'display_column': {0: '[0, 7.22)',
                            9: '[13, 15.5)',
                            18: '[110.88, 512.33]'},
         'value_lower': {0: 0.0, 9: 13.0, 18: 110.8833},
         'value_upper': {0: 7.225, 9: 15.5, 18: 512.3292},
         'count': {0: 43, 9: 80, 18: 49},
         'Survived': {0: 0.06976744186046512, 9: 0.3375,
                      18: 0.7551020408163265}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 9, 18], :], check_like=True)
    assert len(summary_df) == 19


def test_grid_type(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot(df=titanic_data,
                                        feature='Fare',
                                        feature_name='Fare',
                                        target=titanic_target,
                                        grid_type='equal')

    expected = pd.DataFrame(
        {'x': {1: 1, 6: 6, 8: 8},
         'display_column': {1: '[56.93, 113.85)',
                            6: '[341.55, 398.48)',
                            8: '[455.4, 512.33]'},
         'value_lower': {1: 56.925466666666665, 6: 341.5528,
                         8: 455.4037333333333},
         'value_upper': {1: 113.85093333333333, 6: 398.4782666666666,
                         8: 512.3292},
         'count': {1: 87.0, 6: 0.0, 8: 3.0},
         'Survived': {1: 0.6551724137931034, 6: np.nan, 8: 1.0}}
    )

    assert_frame_equal(expected, summary_df.loc[[1, 6, 8], :], check_like=True)
    assert len(summary_df) == 9


def test_grid_range(titanic_data, titanic_target):
    """
    grid_range, need to set grid_type='equal'
    """
    fig, axes, summary_df = target_plot(df=titanic_data,
                                        feature='Fare',
                                        feature_name='Fare',
                                        target=titanic_target,
                                        grid_type='equal',
                                        grid_range=(5, 100))

    expected = pd.DataFrame(
        {'x': {0: 0, 4: 4, 8: 8},
         'display_column': {0: '[5, 15.56)',
                            4: '[47.22, 57.78)',
                            8: '[89.44, 100]'},
         'value_lower': {0: 5.0, 4: 47.22222222222222, 8: 89.44444444444444},
         'value_upper': {0: 15.555555555555555, 4: 57.77777777777778, 8: 100.0},
         'count': {0: 459, 4: 39, 8: 8},
         'Survived': {0: 0.25925925925925924, 4: 0.6666666666666666, 8: 0.875}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 4, 8], :], check_like=True)
    assert len(summary_df) == 9


def test_grid_range_outliers(titanic_data, titanic_target):
    """
    show_outliers with grid_range defined
    grid_range, need to set grid_type='equal'
    """
    fig, axes, summary_df = target_plot(df=titanic_data,
                                        feature='Fare',
                                        feature_name='Fare',
                                        target=titanic_target,
                                        grid_range=(0, 100),
                                        grid_type='equal',
                                        show_outliers=True)

    expected = pd.DataFrame(
        {'x': {0: 0, 8: 8, 9: 9},
         'display_column': {0: '[0, 11.11)', 8: '[88.89, 100]',
                            9: '> 100'},
         'value_lower': {0: 0.0, 8: 88.88888888888889, 9: 100.0},
         'value_upper': {0: 11.11111111111111, 8: 100.0, 9: np.nan},
         'count': {0: 364, 8: 10, 9: 53},
         'Survived': {0: 0.2087912087912088, 8: 0.9, 9: 0.7358490566037735}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 8, 9], :], check_like=True)
    assert len(summary_df) == 10


def test_grid_range_outliers_endpoint(titanic_data, titanic_target):
    """
    show_outliers with grid_range defined and endpoint==False
    grid_range, need to set grid_type='equal'
    """
    fig, axes, summary_df = target_plot(df=titanic_data,
                                        feature='Fare',
                                        feature_name='Fare',
                                        target=titanic_target,
                                        grid_range=(0, 100),
                                        grid_type='equal',
                                        show_outliers=True,
                                        endpoint=False)

    expected = pd.DataFrame(
        {'x': {0: 0, 8: 8, 9: 9},
         'display_column': {0: '[0, 11.11)', 8: '[88.89, 100)',
                            9: '>= 100'},
         'value_lower': {0: 0.0, 8: 88.88888888888889, 9: 100.0},
         'value_upper': {0: 11.11111111111111, 8: 100.0, 9: np.nan},
         'count': {0: 364, 8: 10, 9: 53},
         'Survived': {0: 0.2087912087912088, 8: 0.9, 9: 0.7358490566037735}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 8, 9], :], check_like=True)
    assert len(summary_df) == 10


def test_cust_grid_points(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot(df=titanic_data,
                                        feature='Fare',
                                        feature_name='Fare',
                                        target=titanic_target,
                                        cust_grid_points=range(0, 100, 10))

    expected = pd.DataFrame(
        {'x': {0: 0, 4: 4, 8: 8},
         'display_column': {0: '[0, 10)',
                            4: '[40, 50)',
                            8: '[80, 90]'},
         'value_lower': {0: 0.0, 4: 40.0, 8: 80.0},
         'value_upper': {0: 10.0, 4: 50.0, 8: 90.0},
         'count': {0: 336, 4: 15, 8: 19},
         'Survived': {0: 0.19940476190476192,
                      4: 0.26666666666666666,
                      8: 0.8421052631578947}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 4, 8], :], check_like=True)
    assert len(summary_df) == 9


def test_cust_grid_outliers(titanic_data, titanic_target):
    """
    show_outliers with custom_grid_points
    """
    fig, axes, summary_df = target_plot(df=titanic_data,
                                        feature='Fare',
                                        feature_name='Fare',
                                        target=titanic_target,
                                        cust_grid_points=range(0, 100, 10),
                                        show_outliers=True)

    expected = pd.DataFrame(
        {'x': {0: 0, 8: 8, 9: 9},
         'display_column': {0: '[0, 10)', 8: '[80, 90]',
                            9: '> 90'},
         'value_lower': {0: 0.0, 8: 80.0, 9: 90.0},
         'value_upper': {0: 10.0, 8: 90.0, 9: np.nan},
         'count': {0: 336, 8: 19, 9: 57},
         'Survived': {0: 0.19940476190476192,
                      8: 0.8421052631578947,
                      9: 0.7543859649122807}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 8, 9], :], check_like=True)
    assert len(summary_df) == 10


def test_cust_grid_outliers_endpoint(titanic_data, titanic_target):
    """
    show_outliers with custom_grid_points and endpoint==False
    """
    fig, axes, summary_df = target_plot(df=titanic_data,
                                        feature='Fare',
                                        feature_name='Fare',
                                        target=titanic_target,
                                        cust_grid_points=range(0, 100, 10),
                                        show_outliers=True,
                                        endpoint=False)

    expected = pd.DataFrame(
        {'x': {0: 0, 8: 8, 9: 9},
         'display_column': {0: '[0, 10)', 8: '[80, 90)',
                            9: '>= 90'},
         'value_lower': {0: 0.0, 8: 80.0, 9: 90.0},
         'value_upper': {0: 10.0, 8: 90.0, 9: np.nan},
         'count': {0: 336, 8: 15, 9: 61},
         'Survived': {0: 0.19940476190476192,
                      8: 0.8666666666666667,
                      9: 0.7540983606557377}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 8, 9], :], check_like=True)
    assert len(summary_df) == 10


def test_show_percentile(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot(df=titanic_data,
                                        feature='Fare',
                                        feature_name='Fare',
                                        target=titanic_target,
                                        show_percentile=True)

    expected = pd.Series(
        {0: '[0, 11.11)',
         1: '[11.11, 22.22)',
         2: '[22.22, 33.33)',
         3: '[33.33, 44.44)',
         4: '[44.44, 55.56)',
         5: '[55.56, 66.67)',
         6: '[66.67, 77.78)',
         7: '[77.78, 88.89)',
         8: '[88.89, 100]'}
    )

    assert_series_equal(expected, summary_df['percentile_column'],
                        check_names=False)
    assert len(summary_df) == 9


def test_percentile_range(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot(df=titanic_data,
                                        feature='Fare',
                                        feature_name='Fare',
                                        target=titanic_target,
                                        percentile_range=(5, 95))

    expected = pd.DataFrame(
        {'x': {0: 0, 4: 4, 8: 8},
         'display_column': {0: '[7.22, 7.75)',
                            4: '[13, 16.1)',
                            8: '[56.5, 112.08]'},
         'value_lower': {0: 7.225, 4: 13.0, 8: 56.4958},
         'value_upper': {0: 7.75, 4: 16.1, 8: 112.07915},
         'count': {0: 63, 4: 99, 8: 91},
         'Survived': {0: 0.2222222222222222,
                      4: 0.3838383838383838,
                      8: 0.6593406593406593}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 4, 8], :], check_like=True)
    assert len(summary_df) == 9


def test_percentile_range_outliers(titanic_data, titanic_target):
    """
    show_outliers with percentile_range defined
    """
    fig, axes, summary_df = target_plot(df=titanic_data,
                                        feature='Fare',
                                        feature_name='Fare',
                                        target=titanic_target,
                                        percentile_range=(5, 95),
                                        show_outliers=True)

    expected = pd.DataFrame(
        {'x': {0: 0, 5: 5, 10: 10},
         'display_column': {0: '< 7.22', 5: '[13, 16.1)', 10: '> 112.08'},
         'value_lower': {0: np.nan, 5: 13.0, 10: 112.07915},
         'value_upper': {0: 7.225, 5: 16.1, 10: np.nan},
         'count': {0: 43, 5: 99, 10: 45},
         'Survived': {0: 0.06976744186046512,
                      5: 0.3838383838383838,
                      10: 0.7555555555555555}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 5, 10], :], check_like=True)
    assert len(summary_df) == 11


def test_percentile_range_outliers_endpoint(titanic_data, titanic_target):
    """
    show_outliers with percentile_range defined and endpoint==False
    """
    fig, axes, summary_df = target_plot(df=titanic_data,
                                        feature='Fare',
                                        feature_name='Fare',
                                        target=titanic_target,
                                        percentile_range=(5, 95),
                                        show_outliers=True,
                                        endpoint=False)

    expected = pd.DataFrame(
        {'x': {0: 0, 9: 9, 10: 10},
         'display_column': {0: '< 7.22', 9: '[56.5, 112.08)', 10: '>= 112.08'},
         'value_lower': {0: np.nan, 9: 56.4958, 10: 112.07915},
         'value_upper': {0: 7.225, 9: 112.07915, 10: np.nan},
         'count': {0: 43, 9: 91, 10: 45},
         'Survived': {0: 0.06976744186046512,
                      9: 0.6593406593406593,
                      10: 0.7555555555555555}}
    )

    assert_frame_equal(expected, summary_df.loc[[0, 9, 10], :], check_like=True)
    assert len(summary_df) == 11
