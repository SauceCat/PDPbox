from pdpbox.info_plots import target_plot_interact
import pandas as pd
import numpy as np
from numpy import nan
from pandas.testing import assert_frame_equal


def test_binary_onehot(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=["Sex", ["Embarked_C", "Embarked_Q", "Embarked_S"]],
        feature_names=["Sex", "Embarked"],
        target=titanic_target,
    )
    assert summary_df["count"].sum() == len(titanic_data)


def test_onehot_numeric(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=["Fare", ["Embarked_C", "Embarked_Q", "Embarked_S"]],
        feature_names=["Fare", "Embarked"],
        target=titanic_target,
    )
    assert summary_df["count"].sum() == len(titanic_data)


def test_onehot_numeric_endpoint(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=["Fare", ["Embarked_C", "Embarked_Q", "Embarked_S"]],
        feature_names=["Fare", "Embarked"],
        target=titanic_target,
        endpoint=False,
    )
    assert summary_df["count"].sum() == len(titanic_data)


def test_onehot_numeric_num_grid_points(titanic_data, titanic_target):
    fare_grid_points = 15
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=["Fare", ["Embarked_C", "Embarked_Q", "Embarked_S"]],
        feature_names=["Fare", "Embarked"],
        target=titanic_target,
        num_grid_points=[fare_grid_points, None],
    )
    assert summary_df["count"].sum() == len(titanic_data)
    assert len(summary_df) == (fare_grid_points - 1) * 3


def test_onehot_numeric_gridtype_equal(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=["Fare", ["Embarked_C", "Embarked_Q", "Embarked_S"]],
        feature_names=["Fare", "Embarked"],
        target=titanic_target,
        grid_types=["equal", "equal"],
    )
    assert summary_df["count"].sum() == len(titanic_data)

    # all bins should have the same width (equal grid type)
    assert (summary_df.value_upper_1 - summary_df.value_lower_1).diff().sum() < 1e-9


def test_onehot_numeric_percentile(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=["Fare", ["Embarked_C", "Embarked_Q", "Embarked_S"]],
        feature_names=["Fare", "Embarked"],
        target=titanic_target,
        percentile_ranges=[(5, 95), None],
    )

    fare = titanic_data["Fare"]
    inside_percentile = (fare >= fare.quantile(0.05)) & (fare <= fare.quantile(0.95))
    assert summary_df["count"].sum() == inside_percentile.sum()


def test_onehot_numeric_gridranges(titanic_data, titanic_target):
    """
    Grid type must be 'equal' for grid ranges to work
    TODO: maybe this should be automatic or at least warn the user when grid types not 'equal'
    """
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=["Fare", ["Embarked_C", "Embarked_Q", "Embarked_S"]],
        feature_names=["Fare", "Embarked"],
        target=titanic_target,
        grid_types=["equal", "equal"],
        grid_ranges=[(0, 100), None],
    )

    # counts must total to the total observations inside the defined range
    inside_range = (titanic_data["Fare"] >= 0) & (titanic_data["Fare"] <= 100)
    assert summary_df["count"].sum() == inside_range.sum()

    # first and last values must be equal to the defined ranges
    assert summary_df.groupby("x2").first().value_lower_1.unique()[0] == 0.0
    assert summary_df.groupby("x2").last().value_upper_1.unique()[0] == 100.0


def test_onehot_numeric_gridranges_outliers(titanic_data, titanic_target):
    """
    'show_outliers' implies 'grid_ranges' or another custom grid definition
    TODO: again, this should be explicit to the user
    """
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=["Fare", ["Embarked_C", "Embarked_Q", "Embarked_S"]],
        feature_names=["Fare", "Embarked"],
        target=titanic_target,
        grid_types=["equal", "equal"],
        grid_ranges=[(0, 100), None],
        show_outliers=True,
    )
    assert summary_df["count"].sum() == len(titanic_data)


def test_onehot_numeric_gridranges_outliers_endpoint(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=["Fare", ["Embarked_C", "Embarked_Q", "Embarked_S"]],
        feature_names=["Fare", "Embarked"],
        target=titanic_target,
        grid_types=["equal", "equal"],
        grid_ranges=[(0, 100), None],
        show_outliers=True,
        endpoint=False,
    )
    assert summary_df["count"].sum() == len(titanic_data)


def test_onehot_numeric_cust_grid_points(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=["Fare", ["Embarked_C", "Embarked_Q", "Embarked_S"]],
        feature_names=["Fare", "Embarked"],
        target=titanic_target,
        cust_grid_points=[range(0, 100, 10), None],
    )

    # counts must total to the total observations inside the defined range
    inside_range = (titanic_data["Fare"] >= 0) & (titanic_data["Fare"] <= 90)
    assert summary_df["count"].sum() == inside_range.sum()

    # lower and upper values must follow the prescribed grid
    assert (
        summary_df.groupby("x1").value_lower_1.mean().values == np.arange(0.0, 90, 10)
    ).all()
    assert (
        summary_df.groupby("x1").value_upper_1.mean().values == np.arange(10.0, 100, 10)
    ).all()


def test_onehot_numeric_gridpoints_outliers(titanic_data, titanic_target):
    """
    'show_outliers' implies 'custom_grid_points' or another custom grid definition
    TODO: again, this should be explicit to the user
    """
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=["Fare", ["Embarked_C", "Embarked_Q", "Embarked_S"]],
        feature_names=["Fare", "Embarked"],
        target=titanic_target,
        cust_grid_points=[range(0, 100, 10), None],
        show_outliers=True,
    )
    assert summary_df["count"].sum() == len(titanic_data)


def test_onehot_numeric_gridpoints_outliers_endpoint(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=["Fare", ["Embarked_C", "Embarked_Q", "Embarked_S"]],
        feature_names=["Fare", "Embarked"],
        target=titanic_target,
        cust_grid_points=[range(0, 100, 10), None],
        show_outliers=True,
        endpoint=False,
    )
    assert summary_df["count"].sum() == len(titanic_data)


def test_onehot_numeric_show_percentiles(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=["Fare", ["Embarked_C", "Embarked_Q", "Embarked_S"]],
        feature_names=["Fare", "Embarked"],
        target=titanic_target,
        percentile_ranges=[(5, 95), None],
        show_percentile=True,
    )

    fare = titanic_data["Fare"]
    inside_percentile = (fare >= fare.quantile(0.05)) & (fare <= fare.quantile(0.95))
    assert summary_df["count"].sum() == inside_percentile.sum()


def test_onehot_numeric_show_outliers(titanic_data, titanic_target):
    """
    'show_outliers' implies 'percentile_ranges' or another custom grid definition
    TODO: again, this should be explicit to the user
    """
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=["Fare", ["Embarked_C", "Embarked_Q", "Embarked_S"]],
        feature_names=["Fare", "Embarked"],
        target=titanic_target,
        percentile_ranges=[(5, 95), None],
        show_percentile=True,
        show_outliers=True,
    )
    assert summary_df["count"].sum() == len(titanic_data)


def test_onehot_numeric_show_outliers_endpoint(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=["Fare", ["Embarked_C", "Embarked_Q", "Embarked_S"]],
        feature_names=["Fare", "Embarked"],
        target=titanic_target,
        percentile_ranges=[(5, 95), None],
        show_percentile=True,
        show_outliers=True,
        endpoint=False,
    )
    assert summary_df["count"].sum() == len(titanic_data)


def test_binary_numeric(titanic_data, titanic_target):
    fig, axes, summary_df = target_plot_interact(
        df=titanic_data,
        features=["Fare", "Sex"],
        feature_names=["Fare", "Sex"],
        target=titanic_target,
        show_percentile=True,
        percentile_ranges=[(5, 95), None],
        show_outliers=True,
    )
    assert summary_df["count"].sum() == len(titanic_data)
