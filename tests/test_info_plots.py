# for local debug use
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pdpbox.info_plots import (
    TargetPlot,
    PredictPlot,
    InterectTargetPlot,
    InterectPredictPlot,
)
from pdpbox.utils import FeatureInfo, _make_list
import pytest
import pandas as pd
import numpy as np
import copy
from conftest import DummyModel, PlotTestBase

plot_params = [
    # Test default parameters
    {},
    # Test show_percentile
    {"show_percentile": True},
    # Test custom which_classes
    {"which_classes": [0]},
    {"which_classes": [0, 2]},
    # Test custom figsize
    {"figsize": (900, 500)},
    {"figsize": (12, 8), "engine": "matplotlib"},
    # Test custom ncols
    {"ncols": 1},
    {"ncols": 3},
    # Test custom plot_params
    {"plot_params": {"line": {"width": 2, "colors": ["red", "green"]}}},
    # Test custom engine (matplotlib)
    {"engine": "matplotlib"},
    # Test custom dpi
    {"dpi": 100, "engine": "matplotlib"},
    # Test custom template (plotly_dark)
    {"template": "plotly_dark"},
]


class _TestInfoPlot(PlotTestBase):
    def _test_plot_obj(self, target_or_model_type):
        for plot_obj in self.get_plot_objs(target_or_model_type):
            self.check_common(plot_obj, target_or_model_type)

    def _test_plot(self, params):
        # randomly choose a model_type
        target_or_model_type = np.random.choice(self.model_types)
        if params.get("which_classes", None) is not None:
            target_or_model_type = "multi-class"

        for plot_obj in self.get_plot_objs(target_or_model_type):
            plot_obj.plot(**params)
            self.close_plt(params)
            break


class TestTargetPlot(_TestInfoPlot):
    def get_plot_objs(self, target_type):
        for df, features, feat_types in self.data_gen.get_dummy_dfs(target_type):
            yield TargetPlot(
                df,
                features[0],
                "Feature 1",
                target=[f"target_{i}" for i in range(self.data_gen.DUMMY_MULTI_CLASSES)]
                if target_type == "multi-class"
                else "target",
            )

    def check_common(self, plot_obj, target_type):
        assert plot_obj.plot_type == "target"
        assert isinstance(plot_obj.feature_info, FeatureInfo)

        if target_type == "multi-class":
            assert plot_obj.target == [f"target_{i}" for i in range(plot_obj.n_classes)]
            assert len(plot_obj.target_lines) == plot_obj.n_classes
            for i in range(plot_obj.n_classes):
                assert set(plot_obj.target_lines[i].columns) == set(
                    ["x", f"target_{i}"]
                )
        else:
            assert plot_obj.target == ["target"]
            assert len(plot_obj.target_lines) == 1
            assert set(plot_obj.target_lines[0].columns) == set(["x", "target"])

        assert set(plot_obj.df.columns) == set(
            plot_obj.feature_cols + ["x", "count"] + plot_obj.target
        )
        assert set(plot_obj.count_df.columns) == set(["x", "count", "count_norm"])

        summary_cols = ["x", "value", "count"] + plot_obj.target
        if plot_obj.feature_info.percentiles is not None:
            summary_cols.append("percentile")
        assert set(plot_obj.summary_df.columns) == set(summary_cols)

    @pytest.mark.parametrize("target_type", ["regression", "binary", "multi-class"])
    def test_plot_obj(self, target_type):
        self._test_plot_obj(target_type)

    @pytest.mark.parametrize("params", plot_params)
    def test_plot(self, params):
        self._test_plot(params)


class TestPredictPlot(_TestInfoPlot):
    def get_plot_objs(self, model_type):
        for df, features, feat_types in self.data_gen.get_dummy_dfs():
            model = DummyModel(model_type, features[0], feat_types[0])
            yield PredictPlot(
                df,
                features[0],
                "Feature 1",
                model=model,
                model_features=_make_list(features[0]) + _make_list(features[1]),
                n_classes=0 if model_type == "regression" else None,
            )

    def check_common(self, plot_obj, model_type):
        assert plot_obj.plot_type == "predict"
        assert isinstance(plot_obj.feature_info, FeatureInfo)

        summary_cols = ["x", "value", "count"]
        if model_type == "multi-class":
            assert plot_obj.target == [f"pred_{i}" for i in range(plot_obj.n_classes)]
            assert len(plot_obj.target_lines) == plot_obj.n_classes
            for i in range(plot_obj.n_classes):
                target_cols = [f"pred_{i}_q1", f"pred_{i}_q2", f"pred_{i}_q3"]
                assert set(plot_obj.target_lines[i].columns) == set(["x"] + target_cols)
                summary_cols += target_cols
        else:
            assert plot_obj.target == ["pred"]
            assert len(plot_obj.target_lines) == 1
            target_cols = ["pred_q1", "pred_q2", "pred_q3"]
            assert set(plot_obj.target_lines[0].columns) == set(["x"] + target_cols)
            summary_cols += target_cols

        assert set(plot_obj.df.columns) == set(
            plot_obj.feature_cols + ["x", "count"] + plot_obj.target
        )
        assert set(plot_obj.count_df.columns) == set(["x", "count", "count_norm"])
        if plot_obj.feature_info.percentiles is not None:
            summary_cols.append("percentile")
        assert set(plot_obj.summary_df.columns) == set(summary_cols)

    @pytest.mark.parametrize("model_type", ["regression", "binary", "multi-class"])
    def test_plot_obj(self, model_type):
        self._test_plot_obj(model_type)

    @pytest.mark.parametrize("params", plot_params)
    def test_plot(self, params):
        self._test_plot(params)


class TestInterectTargetPlot(_TestInfoPlot):
    def get_plot_objs(self, target_type):
        for df, features, feat_types in self.data_gen.get_dummy_dfs(target_type):
            yield InterectTargetPlot(
                df,
                features,
                ["Feature 1", "Feature 2"],
                target=[f"target_{i}" for i in range(self.data_gen.DUMMY_MULTI_CLASSES)]
                if target_type == "multi-class"
                else "target",
            )

    def check_common(self, plot_obj, target_type):
        assert plot_obj.plot_type == "interact_target"
        assert all(isinstance(info, FeatureInfo) for info in plot_obj.feature_infos)
        assert len(plot_obj.feature_infos) == 2

        if target_type == "multi-class":
            assert plot_obj.target == [f"target_{i}" for i in range(plot_obj.n_classes)]
        else:
            assert plot_obj.target == ["target"]
        assert set(plot_obj.df.columns) == set(
            plot_obj.feature_cols + ["x1", "x2", "count"] + plot_obj.target
        )

        summary_cols = ["x1", "x2", "value_1", "value_2", "count"] + plot_obj.target
        for i in range(2):
            if plot_obj.feature_infos[i].percentiles is not None:
                summary_cols.append(f"percentile_{i+1}")
        assert set(plot_obj.summary_df.columns) == set(summary_cols)
        assert set(plot_obj.plot_df.columns) == set(
            ["x1", "x2", "count"] + plot_obj.target
        )

    @pytest.mark.parametrize("target_type", ["regression", "binary", "multi-class"])
    def test_plot_obj(self, target_type):
        self._test_plot_obj(target_type)

    @pytest.mark.parametrize("params", plot_params)
    def test_plot(self, params):
        self._test_plot(params)


class TestInterectPredictPlot(_TestInfoPlot):
    def get_plot_objs(self, model_type):
        for df, features, feat_types in self.data_gen.get_dummy_dfs():
            model = DummyModel(model_type, features, feat_types, interact=True)
            yield InterectPredictPlot(
                df,
                features,
                ["Feature 1", "Feature 2"],
                model=model,
                model_features=_make_list(features[0]) + _make_list(features[1]),
                n_classes=0 if model_type == "regression" else None,
            )

    def check_common(self, plot_obj, model_type):
        assert plot_obj.plot_type == "interact_predict"
        assert all(isinstance(info, FeatureInfo) for info in plot_obj.feature_infos)
        assert len(plot_obj.feature_infos) == 2

        if model_type == "multi-class":
            assert plot_obj.target == [f"pred_{i}" for i in range(plot_obj.n_classes)]
        else:
            assert plot_obj.target == ["pred"]
        assert set(plot_obj.df.columns) == set(
            plot_obj.feature_cols + ["x1", "x2", "count"] + plot_obj.target
        )

        summary_cols = ["x1", "x2", "value_1", "value_2", "count"] + plot_obj.target
        for i in range(2):
            if plot_obj.feature_infos[i].percentiles is not None:
                summary_cols.append(f"percentile_{i+1}")
        assert set(plot_obj.summary_df.columns) == set(summary_cols)
        assert set(plot_obj.plot_df.columns) == set(
            ["x1", "x2", "count"] + plot_obj.target
        )

    @pytest.mark.parametrize("model_type", ["regression", "binary", "multi-class"])
    def test_plot_obj(self, model_type):
        self._test_plot_obj(model_type)

    @pytest.mark.parametrize("params", plot_params)
    def test_plot(self, params):
        self._test_plot(params)
