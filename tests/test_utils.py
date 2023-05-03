# for local debug use
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pdpbox.utils import (
    FeatureInfo,
    _to_rgba,
    _check_col,
    _check_dataset,
    _make_list,
    _check_model,
    _check_classes,
    _check_memory_limit,
    _check_frac_to_plot,
    _calc_n_jobs,
    _get_string,
    _expand_params_for_interact,
    _calc_preds,
)
import pytest
import pandas as pd
import numpy as np
import copy


class FeatureInfoTestCases:
    grid_types = [
        ["percentile", True],
        ["equal", True],
        ["wrong", False],
    ]

    percentile_ranges = [
        # Correct inputs
        [(0, 100), True],
        [(25, 75), True],
        [(1.5, 98.5), True],
        # Wrong inputs
        # Incorrect type
        [[0, 100], False],
        ["0, 100", False],
        [100, False],
        # Incorrect tuple length
        [(0, 100, 200), False],
        [(0,), False],
        # Tuple elements not in order
        [(100, 0), False],
        [(75, 25), False],
        [(98.5, 1.5), False],
        # Tuple elements out of range (0-100)
        [(-10, 110), False],
        [(50, 120), False],
        [(-5, 50), False],
    ]

    grid_ranges = [
        # Correct inputs
        [(0, 100), True],
        [(-50, 50), True],
        [(10.5, 20.5), True],
        # Wrong inputs
        # Incorrect type
        [[0, 100], False],
        ["0, 100", False],
        [100, False],
        # Incorrect tuple length
        [(0, 100, 200), False],
        [(0,), False],
        # Tuple elements not in order
        [(100, 0), False],
        [(50, -50), False],
        [(20.5, 10.5), False],
    ]

    binary_feature_info_params = [
        {
            "feature": "Sex",
            "feature_name": "gender",
        },
    ]

    onehot_feature_info_params = [
        {
            "feature": ["Embarked_C", "Embarked_S", "Embarked_Q"],
            "feature_name": "embarked",
        },
    ]

    numeric_feature_info_params = [
        {
            "feature": "Fare",
            "feature_name": "fare",
            "cust_grid_points": None,
            "grid_type": "percentile",
            "num_grid_points": 10,
            "percentile_range": None,
            "grid_range": None,
            "show_outliers": False,
            "endpoint": True,
        },
        {
            "feature": "Fare",
            "feature_name": "fare",
            "cust_grid_points": None,
            "grid_type": "equal",
            "num_grid_points": 7,
            "percentile_range": None,
            "grid_range": None,
            "show_outliers": False,
            "endpoint": True,
        },
        {
            "feature": "Fare",
            "feature_name": "fare",
            "cust_grid_points": [5, 10, 20, 50, 100],
            "grid_type": "percentile",
            "num_grid_points": None,
            "percentile_range": None,
            "grid_range": None,
            "show_outliers": False,
            "endpoint": True,
        },
        {
            "feature": "Fare",
            "feature_name": "fare",
            "cust_grid_points": None,
            "grid_type": "percentile",
            "num_grid_points": 8,
            "percentile_range": (5, 95),
            "grid_range": None,
            "show_outliers": True,
            "endpoint": True,
        },
        {
            "feature": "Fare",
            "feature_name": "fare",
            "cust_grid_points": None,
            "grid_type": "equal",
            "num_grid_points": 10,
            "percentile_range": None,
            "grid_range": (0, 200),
            "show_outliers": False,
            "endpoint": False,
        },
    ]

    numeric_feature_buckets = [
        [
            {
                "endpoint": True,
                "cust_grid_points": None,
                "grid_type": "percentile",
                "show_outliers": False,
            },
            {
                "x": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8]),
                "display_columns": [
                    "[1.0, 2.0)",
                    "[2.0, 3.0)",
                    "[3.0, 4.0)",
                    "[4.0, 5.0)",
                    "[5.0, 6.0)",
                    "[6.0, 7.0)",
                    "[7.0, 8.0)",
                    "[8.0, 9.0)",
                    "[9.0, 10.0]",
                ],
                "percentile_columns": [
                    "[0.0, 11.11)",
                    "[11.11, 22.22)",
                    "[22.22, 33.33)",
                    "[33.33, 44.44)",
                    "[44.44, 55.56)",
                    "[55.56, 66.67)",
                    "[66.67, 77.78)",
                    "[77.78, 88.89)",
                    "[88.89, 100.0]",
                ],
            },
        ],
        [
            {
                "endpoint": False,
                "cust_grid_points": None,
                "grid_type": "percentile",
                "show_outliers": False,
            },
            {
                "x": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
                "display_columns": [
                    "[1.0, 2.0)",
                    "[2.0, 3.0)",
                    "[3.0, 4.0)",
                    "[4.0, 5.0)",
                    "[5.0, 6.0)",
                    "[6.0, 7.0)",
                    "[7.0, 8.0)",
                    "[8.0, 9.0)",
                    "[9.0, 10.0)",
                ],
                "percentile_columns": [
                    "[0.0, 11.11)",
                    "[11.11, 22.22)",
                    "[22.22, 33.33)",
                    "[33.33, 44.44)",
                    "[44.44, 55.56)",
                    "[55.56, 66.67)",
                    "[66.67, 77.78)",
                    "[77.78, 88.89)",
                    "[88.89, 100.0)",
                ],
            },
        ],
        [
            {
                "endpoint": False,
                "cust_grid_points": None,
                "grid_type": "percentile",
                "percentile_range": (0, 90),
                "show_outliers": True,
            },
            {
                "x": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                "display_columns": [
                    "[1.0, 1.9)",
                    "[1.9, 2.8)",
                    "[2.8, 3.7)",
                    "[3.7, 4.6)",
                    "[4.6, 5.5)",
                    "[5.5, 6.4)",
                    "[6.4, 7.3)",
                    "[7.3, 8.2)",
                    "[8.2, 9.1)",
                    ">=9.1",
                ],
                "percentile_columns": [
                    "[0.0, 10.0)",
                    "[10.0, 20.0)",
                    "[20.0, 30.0)",
                    "[30.0, 40.0)",
                    "[40.0, 50.0)",
                    "[50.0, 60.0)",
                    "[60.0, 70.0)",
                    "[70.0, 80.0)",
                    "[80.0, 90.0)",
                    ">=90.0",
                ],
            },
        ],
        [
            {
                "endpoint": True,
                "cust_grid_points": None,
                "grid_type": "percentile",
                "percentile_range": (0, 90),
                "show_outliers": True,
            },
            {
                "x": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                "display_columns": [
                    "[1.0, 1.9)",
                    "[1.9, 2.8)",
                    "[2.8, 3.7)",
                    "[3.7, 4.6)",
                    "[4.6, 5.5)",
                    "[5.5, 6.4)",
                    "[6.4, 7.3)",
                    "[7.3, 8.2)",
                    "[8.2, 9.1]",
                    ">9.1",
                ],
                "percentile_columns": [
                    "[0.0, 10.0)",
                    "[10.0, 20.0)",
                    "[20.0, 30.0)",
                    "[30.0, 40.0)",
                    "[40.0, 50.0)",
                    "[50.0, 60.0)",
                    "[60.0, 70.0)",
                    "[70.0, 80.0)",
                    "[80.0, 90.0]",
                    ">90.0",
                ],
            },
        ],
        [
            {
                "endpoint": True,
                "cust_grid_points": None,
                "grid_type": "percentile",
                "percentile_range": (10, 100),
                "show_outliers": True,
            },
            {
                "x": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                "display_columns": [
                    "<1.9",
                    "[1.9, 2.8)",
                    "[2.8, 3.7)",
                    "[3.7, 4.6)",
                    "[4.6, 5.5)",
                    "[5.5, 6.4)",
                    "[6.4, 7.3)",
                    "[7.3, 8.2)",
                    "[8.2, 9.1)",
                    "[9.1, 10.0]",
                ],
                "percentile_columns": [
                    "<10.0",
                    "[10.0, 20.0)",
                    "[20.0, 30.0)",
                    "[30.0, 40.0)",
                    "[40.0, 50.0)",
                    "[50.0, 60.0)",
                    "[60.0, 70.0)",
                    "[70.0, 80.0)",
                    "[80.0, 90.0)",
                    "[90.0, 100.0]",
                ],
            },
        ],
        [
            {
                "endpoint": True,
                "cust_grid_points": None,
                "grid_type": "percentile",
                "percentile_range": (10, 90),
                "show_outliers": True,
            },
            {
                "x": np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10]),
                "display_columns": [
                    "<1.9",
                    "[1.9, 2.7)",
                    "[2.7, 3.5)",
                    "[3.5, 4.3)",
                    "[4.3, 5.1)",
                    "[5.1, 5.9)",
                    "[5.9, 6.7)",
                    "[6.7, 7.5)",
                    "[7.5, 8.3)",
                    "[8.3, 9.1]",
                    ">9.1",
                ],
                "percentile_columns": [
                    "<10.0",
                    "[10.0, 18.89)",
                    "[18.89, 27.78)",
                    "[27.78, 36.67)",
                    "[36.67, 45.56)",
                    "[45.56, 54.44)",
                    "[54.44, 63.33)",
                    "[63.33, 72.22)",
                    "[72.22, 81.11)",
                    "[81.11, 90.0]",
                    ">90.0",
                ],
            },
        ],
        [
            {
                "endpoint": True,
                "cust_grid_points": None,
                "grid_type": "equal",
                "grid_range": (2, 9),
                "show_outliers": True,
            },
            {
                "x": np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10]),
                "display_columns": [
                    "<2.0",
                    "[2.0, 2.78)",
                    "[2.78, 3.56)",
                    "[3.56, 4.33)",
                    "[4.33, 5.11)",
                    "[5.11, 5.89)",
                    "[5.89, 6.67)",
                    "[6.67, 7.44)",
                    "[7.44, 8.22)",
                    "[8.22, 9.0]",
                    ">9.0",
                ],
                "percentile_columns": [],
            },
        ],
        [
            {
                "endpoint": True,
                "cust_grid_points": [2, 4, 6, 8],
                "show_outliers": True,
            },
            {
                "x": np.array([0, 1, 1, 2, 2, 3, 3, 3, 4, 4]),
                "display_columns": ["<2", "[2, 4)", "[4, 6)", "[6, 8]", ">8"],
                "percentile_columns": [],
            },
        ],
    ]


class TestFeatureInfo:
    @pytest.fixture(scope="function")
    def sample_data(self):
        return pd.DataFrame({"feature": np.arange(1, 11)})

    @pytest.fixture(scope="function")
    def feature_info_sample(self, sample_data):
        feature_info = FeatureInfo(
            feature="feature",
            feature_name="Feature",
            df=sample_data,
            grid_type="percentile",
            num_grid_points=5,
        )
        return feature_info

    def get_feature_info(self, titanic_data, **kwargs):
        return FeatureInfo(df=titanic_data, **kwargs)

    def get_random_numeric_params(self):
        return np.random.choice(FeatureInfoTestCases.numeric_feature_info_params, 1)[
            0
        ].copy()

    def check_valid(self, titanic_data, params, is_valid):
        if is_valid:
            self.get_feature_info(titanic_data, **params)
        else:
            with pytest.raises(AssertionError):
                self.get_feature_info(titanic_data, **params)

    @pytest.mark.parametrize("grid_type, is_valid", FeatureInfoTestCases.grid_types)
    def test_grid_type(self, titanic_data, grid_type, is_valid):
        params = self.get_random_numeric_params()
        params["grid_type"] = grid_type
        self.check_valid(titanic_data, params, is_valid)

    @pytest.mark.parametrize("grid_range, is_valid", FeatureInfoTestCases.grid_ranges)
    def test_check_grid_range(self, titanic_data, grid_range, is_valid):
        params = self.get_random_numeric_params()
        params["grid_range"] = grid_range
        self.check_valid(titanic_data, params, is_valid)

    @pytest.mark.parametrize(
        "percentile_range, is_valid", FeatureInfoTestCases.percentile_ranges
    )
    def test_check_percentile_range(self, titanic_data, percentile_range, is_valid):
        params = self.get_random_numeric_params()
        params["percentile_range"] = percentile_range
        self.check_valid(titanic_data, params, is_valid)

    @pytest.mark.parametrize("params", FeatureInfoTestCases.binary_feature_info_params)
    def test_binary_feature(self, params, titanic_data):
        binary_feature_info = self.get_feature_info(titanic_data, **params)
        assert binary_feature_info.type == "binary"

        binary_feature_info.prepare(titanic_data)
        assert np.array_equal(binary_feature_info.grids, [0, 1])
        assert binary_feature_info.percentiles is None
        assert binary_feature_info.num_bins == 2

    @pytest.mark.parametrize("params", FeatureInfoTestCases.onehot_feature_info_params)
    def test_onehot_feature(self, params, titanic_data):
        onehot_feature_info = self.get_feature_info(titanic_data, **params)
        assert onehot_feature_info.type == "onehot"

        onehot_feature_info.prepare(titanic_data)
        assert np.array_equal(onehot_feature_info.grids, onehot_feature_info.col_name)
        assert onehot_feature_info.percentiles is None
        assert onehot_feature_info.num_bins == len(onehot_feature_info.col_name)

    @pytest.mark.parametrize("params", FeatureInfoTestCases.numeric_feature_info_params)
    def test_numeric_feature(self, params, titanic_data):
        numeric_feature_info = self.get_feature_info(titanic_data, **params)
        assert numeric_feature_info.type == "numeric"

        numeric_feature_info.prepare(titanic_data)
        if params["cust_grid_points"] is not None:
            assert np.array_equal(
                numeric_feature_info.grids, params["cust_grid_points"]
            )
        else:
            assert len(numeric_feature_info.grids) == params["num_grid_points"]
            if params["grid_type"] == "percentile":
                assert numeric_feature_info.percentiles is not None
            else:
                assert numeric_feature_info.percentiles is None
        assert numeric_feature_info.num_bins == len(numeric_feature_info.grids) - 1

    def test_get_numeric_grids_percentile(self, feature_info_sample, sample_data):
        grids, percentiles = feature_info_sample._get_numeric_grids(
            sample_data["feature"].values
        )

        assert len(grids) == feature_info_sample.num_grid_points
        assert len(percentiles) == feature_info_sample.num_grid_points
        assert np.all(np.diff(grids) > 0), "Grids should be sorted in ascending order"

        expected_percentiles = [0, 25, 50, 75, 100]
        for i, p in enumerate(percentiles):
            assert len(p) == 1 or np.all(
                np.diff(p) > 0
            ), "Percentiles should be sorted in ascending order"
            assert round(p[0], 2) == expected_percentiles[i]

    def test_get_numeric_grids_equal(self, feature_info_sample, sample_data):
        feature_info_sample.grid_type = "equal"
        feature_info_sample.num_grid_points = 4
        grids, percentiles = feature_info_sample._get_numeric_grids(
            sample_data["feature"].values
        )

        assert len(grids) == feature_info_sample.num_grid_points
        assert percentiles is None
        assert np.all(np.diff(grids) > 0), "Grids should be sorted in ascending order"
        assert np.array_equal(
            grids, np.linspace(1, 10, feature_info_sample.num_grid_points)
        )

    def test_get_numeric_grids_percentile_range(self, feature_info_sample, sample_data):
        feature_info_sample.grid_type = "percentile"
        feature_info_sample.percentile_range = (25, 75)
        grids, percentiles = feature_info_sample._get_numeric_grids(
            sample_data["feature"].values
        )

        assert len(grids) == feature_info_sample.num_grid_points
        assert len(percentiles) == feature_info_sample.num_grid_points
        assert np.all(np.diff(grids) > 0), "Grids should be sorted in ascending order"

        expected_percentiles = [25, 37.5, 50, 62.5, 75]
        for i, p in enumerate(percentiles):
            assert len(p) == 1 or np.all(
                np.diff(p) > 0
            ), "Percentiles should be sorted in ascending order"
            assert round(p[0], 2) == expected_percentiles[i]

    def test_get_numeric_grids_grid_range(self, feature_info_sample, sample_data):
        feature_info_sample.grid_type = "equal"
        feature_info_sample.grid_range = (2, 8)
        grids, percentiles = feature_info_sample._get_numeric_grids(
            sample_data["feature"].values
        )

        assert len(grids) == feature_info_sample.num_grid_points
        assert percentiles is None
        assert np.all(np.diff(grids) > 0), "Grids should be sorted in ascending order"
        assert np.array_equal(grids, np.linspace(2, 8, 5))

    def test_binary_feature_buckets(self):
        feature_values = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        df = pd.DataFrame(
            {
                "Sex": feature_values,
            }
        )
        feature_info = FeatureInfo(feature="Sex", feature_name="gender", df=df)
        feature_info._get_grids(df)
        mapped_df = feature_info._map_values_to_buckets(df)

        assert np.array_equal(mapped_df["x"].values, feature_values)
        assert feature_info.display_columns == ["Sex_0", "Sex_1"]
        assert feature_info.percentile_columns == []

    def test_onehot_feature_buckets(self):
        df = pd.DataFrame(
            {
                "Embarked_C": [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                "Embarked_S": [0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                "Embarked_Q": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            }
        )
        feature_values = np.array([0, 1, 1, 1, 0, 2, 2, 1, 1, 1])
        one_hot_columns = ["Embarked_C", "Embarked_S", "Embarked_Q"]
        feature_info = FeatureInfo(
            feature=one_hot_columns, feature_name="embarked", df=df
        )
        feature_info._get_grids(df)
        mapped_df = feature_info._map_values_to_buckets(df)

        assert np.array_equal(mapped_df["x"].values, feature_values)
        assert feature_info.display_columns == one_hot_columns
        assert feature_info.percentile_columns == []

    @pytest.mark.parametrize(
        "params, expected", FeatureInfoTestCases.numeric_feature_buckets
    )
    def test_numeric_feature_buckets(self, params, expected, sample_data):
        feature_info = FeatureInfo(
            feature="feature",
            feature_name="Feature",
            df=sample_data,
            num_grid_points=10,
            **params,
        )
        feature_info._get_grids(sample_data)
        mapped_df = feature_info._map_values_to_buckets(sample_data)

        assert np.array_equal(mapped_df["x"].values, expected["x"])
        assert feature_info.display_columns == expected["display_columns"]
        assert feature_info.percentile_columns == expected["percentile_columns"]


def test_to_rgba():
    color = (0.5, 0.25, 0.75)
    opacity = 0.8
    expected_output = "rgba(127,63,191,0.8)"

    assert _to_rgba(color) == "rgba(127,63,191,1.0)"
    assert _to_rgba(color, opacity) == expected_output

    # Test edge cases
    assert _to_rgba((0, 0, 0)) == "rgba(0,0,0,1.0)"
    assert _to_rgba((1, 1, 1)) == "rgba(255,255,255,1.0)"
    assert _to_rgba((0.333, 0.666, 1.0), 0) == "rgba(84,169,255,0)"


def test_check_target_col():
    # Create sample DataFrames for testing
    df_binary = pd.DataFrame({"target": [0, 1, 0, 1, 1]})
    df_multi_class = pd.DataFrame(
        {"class1": [1, 0, 0], "class2": [0, 1, 0], "class3": [0, 0, 1]}
    )
    df_regression = pd.DataFrame({"target": [1.5, 2.3, 3.1, 4.6, 5.9]})

    assert _check_col("target", df_binary) == "binary"
    assert _check_col(["class1", "class2", "class3"], df_multi_class) == "multi-class"
    assert _check_col("target", df_regression) == "regression"


def test_check_dataset():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    _check_dataset(df)

    not_dataframe = {"A": [1, 2], "B": [3, 4]}
    with pytest.raises(AssertionError, match="only accept pandas DataFrame"):
        _check_dataset(not_dataframe)

    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    features = ["A", "B", "C"]
    with pytest.raises(
        ValueError, match="df doesn't contain all model features, missing: {'C'}"
    ):
        _check_dataset(df, features)

    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    features = ["A", "B", "C"]
    _check_dataset(df, features)


def test_make_list():
    input_list = [1, 2, 3]
    output_list = _make_list(input_list)
    assert output_list == input_list

    input_not_list = "example"
    output_list = _make_list(input_not_list)
    assert output_list == [input_not_list]

    input_list = []
    output_list = _make_list(input_list)
    assert output_list == input_list

    input_value = 42
    output_list = _make_list(input_value)
    assert output_list == [input_value]


class TestCheckModel:
    def get_dummy_model(self):
        class DummyModel:
            pass

        return DummyModel()

    def get_dummy_func(self):
        def dummy_func():
            pass

        return dummy_func

    def test_n_classes_from_model(self, titanic_model):
        n_classes, pred_func, from_model = _check_model(titanic_model, None, None)
        assert n_classes == 2
        assert callable(pred_func)
        assert from_model

    def test_n_classes_from_input(self):
        model = self.get_dummy_model()
        model.predict_proba = self.get_dummy_func()
        dummy_n_classes = 3

        n_classes, pred_func, from_model = _check_model(model, dummy_n_classes, None)
        assert n_classes == dummy_n_classes
        assert callable(pred_func)
        assert from_model

    def test_pred_func_from_input(self):
        model = self.get_dummy_model()
        model.n_classes_ = 2
        pred_func_input = self.get_dummy_func()
        n_classes, pred_func, from_model = _check_model(model, None, pred_func_input)
        assert n_classes == model.n_classes_
        assert callable(pred_func)
        assert pred_func == pred_func_input
        assert not from_model

    def test_raises_n_classes_error(self):
        model = self.get_dummy_model()
        model.predict = self.get_dummy_func()

        with pytest.raises(AssertionError, match="n_classes is required"):
            _check_model(model, None, None)

    def test_raises_pred_func_error(self):
        model = self.get_dummy_model()
        model.n_classes_ = 3

        with pytest.raises(AssertionError, match="pred_func is required"):
            _check_model(model, None, None)


def test_check_classes():
    for which_classes in [None, []]:
        which_classes = _check_classes(which_classes, 4)
        assert which_classes == [0, 1, 2, 3]

    for n_classes in range(3):
        which_classes = _check_classes(None, n_classes)
        assert which_classes == [0]

    which_classes = _check_classes([0, 2], 4)
    assert which_classes == [0, 2]

    with pytest.raises(ValueError, match="class index should be >= 0"):
        _check_classes([-1, 0, 1], 3)

    with pytest.raises(ValueError, match="class index should be < n_classes"):
        _check_classes([0, 3], 3)


def test_check_memory_limit():
    valid_values = [0.5, 0.1, 0.99]
    for limit in valid_values:
        _check_memory_limit(limit)

    invalid_values = [0, 1, -0.1, 1.1]
    for limit in invalid_values:
        with pytest.raises(ValueError):
            _check_memory_limit(limit)


def test_check_frac_to_plot():
    valid_values = [0.5, 0.1, 1, 10]
    for frac in valid_values:
        _check_frac_to_plot(frac)

    with pytest.raises(
        ValueError, match="frac_to_plot: should in range\\(0, 1\\) when it is a float"
    ):
        _check_frac_to_plot(0.0)
    with pytest.raises(
        ValueError, match="frac_to_plot: should in range\\(0, 1\\) when it is a float"
    ):
        _check_frac_to_plot(-0.1)
    with pytest.raises(ValueError, match="frac_to_plot: should be larger than 0."):
        _check_frac_to_plot(0)
    with pytest.raises(ValueError, match="frac_to_plot: should be float or integer"):
        _check_frac_to_plot("invalid")


def test_calc_n_jobs():
    df = pd.DataFrame(np.random.random((100, 4)), columns=["A", "B", "C", "D"])
    n_grids = 10
    memory_limit = 0.5
    n_jobs = 4

    true_n_jobs = _calc_n_jobs(df, n_grids, memory_limit, n_jobs)

    assert isinstance(true_n_jobs, int)
    assert true_n_jobs >= 1


def test_get_string():
    assert _get_string(5.0) == "5"
    assert _get_string(5.123) == "5.12"
    assert _get_string(5.1) == "5.1"
    assert _get_string(3) == "3"
    assert _get_string(-3.14) == "-3.14"
    assert _get_string(0.0) == "0"


def test_expand_params_for_interact():
    test_params = {
        "param1": [1, 2],
        "param2": 3,
    }

    expanded_params = _expand_params_for_interact(test_params)

    assert expanded_params["param1"] == [1, 2]
    assert expanded_params["param2"] == [3, 3]

    with pytest.raises(AssertionError):
        _expand_params_for_interact({"param3": [1, 2, 3]})


def test_calc_preds():
    class DummyModel:
        def predict(self, X):
            return X * 2

    model = DummyModel()
    X = np.array([1, 2, 3, 4, 5])
    pred_func = model.predict
    from_model = True
    predict_kwds = {}

    preds = _calc_preds(model, X, pred_func, from_model, predict_kwds)
    assert np.array_equal(preds, X * 2)

    chunk_size = 2
    preds = _calc_preds(model, X, pred_func, from_model, predict_kwds, chunk_size)
    assert np.array_equal(preds, X * 2)
