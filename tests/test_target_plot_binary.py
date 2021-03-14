from pdpbox.info_plots import target_plot
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal


def test_binary(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot(
        df=titanic_data, feature="Sex", feature_name="Sex", target=titanic_target
    )


def test_onehot(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot(
        df=titanic_data,
        feature=["Embarked_C", "Embarked_Q", "Embarked_S"],
        feature_name="Embarked",
        target=titanic_target,
    )


def test_numeric(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot(
        df=titanic_data, feature="Fare", feature_name="Fare", target=titanic_target
    )
    assert len(summary_df) == 9


def test_endpoint(titanic_data, titanic_target):
    """
    test endpoint==False (last point should't be included)
    """
    fig, axes, summary_df = target_plot(
        df=titanic_data,
        feature="Fare",
        feature_name="Fare",
        target=titanic_target,
        endpoint=False,
    )
    assert len(summary_df) == 10


def test_num_grid_points(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot(
        df=titanic_data,
        feature="Fare",
        feature_name="Fare",
        target=titanic_target,
        num_grid_points=20,
    )
    assert len(summary_df) == 19


def test_grid_type(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot(
        df=titanic_data,
        feature="Fare",
        feature_name="Fare",
        target=titanic_target,
        grid_type="equal",
    )
    assert len(summary_df) == 9


def test_grid_range(titanic_data, titanic_target):
    """
    grid_range, need to set grid_type='equal'
    """
    fig, axes, summary_df = target_plot(
        df=titanic_data,
        feature="Fare",
        feature_name="Fare",
        target=titanic_target,
        grid_type="equal",
        grid_range=(5, 100),
    )
    assert len(summary_df) == 9


def test_grid_range_outliers(titanic_data, titanic_target):
    """
    show_outliers with grid_range defined
    grid_range, need to set grid_type='equal'
    """
    fig, axes, summary_df = target_plot(
        df=titanic_data,
        feature="Fare",
        feature_name="Fare",
        target=titanic_target,
        grid_range=(0, 100),
        grid_type="equal",
        show_outliers=True,
    )
    assert len(summary_df) == 10


def test_grid_range_outliers_endpoint(titanic_data, titanic_target):
    """
    show_outliers with grid_range defined and endpoint==False
    grid_range, need to set grid_type='equal'
    """
    fig, axes, summary_df = target_plot(
        df=titanic_data,
        feature="Fare",
        feature_name="Fare",
        target=titanic_target,
        grid_range=(0, 100),
        grid_type="equal",
        show_outliers=True,
        endpoint=False,
    )
    assert len(summary_df) == 10


def test_cust_grid_points(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot(
        df=titanic_data,
        feature="Fare",
        feature_name="Fare",
        target=titanic_target,
        cust_grid_points=range(0, 100, 10),
    )
    assert len(summary_df) == 9


def test_cust_grid_outliers(titanic_data, titanic_target):
    """
    show_outliers with custom_grid_points
    """
    fig, axes, summary_df = target_plot(
        df=titanic_data,
        feature="Fare",
        feature_name="Fare",
        target=titanic_target,
        cust_grid_points=range(0, 100, 10),
        show_outliers=True,
    )
    assert len(summary_df) == 10


def test_cust_grid_outliers_endpoint(titanic_data, titanic_target):
    """
    show_outliers with custom_grid_points and endpoint==False
    """
    fig, axes, summary_df = target_plot(
        df=titanic_data,
        feature="Fare",
        feature_name="Fare",
        target=titanic_target,
        cust_grid_points=range(0, 100, 10),
        show_outliers=True,
        endpoint=False,
    )
    assert len(summary_df) == 10


def test_show_percentile(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot(
        df=titanic_data,
        feature="Fare",
        feature_name="Fare",
        target=titanic_target,
        show_percentile=True,
    )
    assert len(summary_df) == 9


def test_percentile_range(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot(
        df=titanic_data,
        feature="Fare",
        feature_name="Fare",
        target=titanic_target,
        percentile_range=(5, 95),
    )
    assert len(summary_df) == 9


def test_percentile_range_outliers(titanic_data, titanic_target):
    """
    show_outliers with percentile_range defined
    """
    fig, axes, summary_df = target_plot(
        df=titanic_data,
        feature="Fare",
        feature_name="Fare",
        target=titanic_target,
        percentile_range=(5, 95),
        show_outliers=True,
    )
    assert len(summary_df) == 11


def test_percentile_range_outliers_endpoint(titanic_data, titanic_target):
    """
    show_outliers with percentile_range defined and endpoint==False
    """
    fig, axes, summary_df = target_plot(
        df=titanic_data,
        feature="Fare",
        feature_name="Fare",
        target=titanic_target,
        percentile_range=(5, 95),
        show_outliers=True,
        endpoint=False,
    )
    assert len(summary_df) == 11
