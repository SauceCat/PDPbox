# for local debug use
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pdpbox.utils import *
import pytest
import pandas as pd
import numpy as np


binary_feature_info_params = [
    {
        "feature": "Sex",
        "feature_name": "gender",
        "df": None,
    },
]

onehot_feature_info_params = [
    {
        "feature": ["Embarked_C", "Embarked_S", "Embarked_Q"],
        "feature_name": "embarked",
        "df": None,
    },
]

numeric_feature_info_params = [
    {
        "feature": "Fare",
        "feature_name": "fare",
        "df": None,
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
        "df": None,
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
        "df": None,
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
        "df": None,
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
        "df": None,
        "cust_grid_points": None,
        "grid_type": "equal",
        "num_grid_points": 10,
        "percentile_range": None,
        "grid_range": (0, 200),
        "show_outliers": False,
        "endpoint": False,
    },
]


class TestFeatureInfo:
    def get_feature_info(self, params, titanic_data):
        params["df"] = titanic_data
        return FeatureInfo(**params)

    @pytest.mark.parametrize("params", binary_feature_info_params)
    def test_binary_feature(self, params, titanic_data):
        binary_feature_info = self.get_feature_info(params, titanic_data)
        assert binary_feature_info.type == "binary"

        binary_feature_info.prepare(titanic_data)
        assert np.array_equal(binary_feature_info.grids, [0, 1])
        assert binary_feature_info.percentiles is None
        assert binary_feature_info.num_bins == 2

    @pytest.mark.parametrize("params", onehot_feature_info_params)
    def test_onehot_feature(self, params, titanic_data):
        onehot_feature_info = self.get_feature_info(params, titanic_data)
        assert onehot_feature_info.type == "onehot"

        onehot_feature_info.prepare(titanic_data)
        assert np.array_equal(onehot_feature_info.grids, onehot_feature_info.col_name)
        assert onehot_feature_info.percentiles is None
        assert onehot_feature_info.num_bins == len(onehot_feature_info.col_name)

    @pytest.mark.parametrize("params", numeric_feature_info_params)
    def test_numeric_feature(self, params, titanic_data):
        numeric_feature_info = self.get_feature_info(params, titanic_data)
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
