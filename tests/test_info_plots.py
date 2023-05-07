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
import matplotlib


dummy_features = {
    "numeric": np.random.randn(100),
    "onehot": np.array([np.eye(4)[i] for i in np.random.randint(0, 4, 100)]),
    "binary": np.random.randint(0, 2, 100),
}
feature_types = dummy_features.keys()

plot_params = [
    # Test default parameters
    {},
    # Test custom which_classes
    {
        "which_classes": [0],
    },
    {
        "which_classes": [0, 2],
    },
    # Test show_percentile
    {
        "show_percentile": True,
    },
    # Test custom figsize
    {
        "figsize": (900, 500),
    },
    {
        "figsize": (12, 8),
        "engine": "matplotlib",
    },
    # Test custom ncols
    {
        "ncols": 1,
    },
    {
        "ncols": 3,
    },
    # Test custom plot_params
    {
        "plot_params": {"line": {"width": 2, "colors": ["red", "green"]}},
    },
    # Test custom engine (matplotlib)
    {
        "engine": "matplotlib",
    },
    # Test custom dpi
    {
        "dpi": 100,
        "engine": "matplotlib",
    },
    # Test custom template (plotly_dark)
    {
        "template": "plotly_dark",
    },
]


def get_dummy_dfs(target_type=None):
    dfs = []
    for t1 in feature_types:
        for t2 in feature_types:
            features = []
            df = {}
            for i, t in enumerate([t1, t2], start=1):
                if t == "onehot":
                    features.append([])
                    for j in range(dummy_features[t].shape[1]):
                        col = f"feature_{i}_{j}"
                        df[col] = dummy_features[t][:, j]
                        features[-1].append(col)
                else:
                    col = f"feature_{i}"
                    df[col] = dummy_features[t]
                    features.append(col)
            df = pd.DataFrame(df)
            if target_type == "regression":
                df["target"] = np.random.randn(100)
            elif target_type == "binary":
                df["target"] = np.random.randint(0, 2, 100)
            elif target_type == "multi-class":
                for i in range(3):
                    df[f"target_{i}"] = np.random.randint(0, 2, 100)
            dfs.append([df, features])

    return dfs


def get_dummy_model(model_type):
    class DummyModel:
        pass

    model = DummyModel()

    if model_type == "regression":
        model.predict = lambda x: np.random.randn(len(x))
    elif model_type == "binary":
        model.n_classes_ = 2
        model.predict_proba = lambda x: np.random.randn(len(x), 2)
    elif model_type == "multi-class":
        model.n_classes_ = 3
        model.predict_proba = lambda x: np.random.randn(len(x), 3)
    return model


class TestInfoPlot:
    def close_plt(self, params):
        if params.get("engine", "plotly") == "matplotlib":
            matplotlib.pyplot.close()

    def _test_regression(self, params):
        for info_plot in self.get_plot_objs("regression"):
            assert info_plot.n_classes == 0
            self.check_common(info_plot, "regression")
            if params.get("which_classes", None) is None:
                info_plot.plot(**params)
                self.close_plt(params)

    def _test_binary(self, params):
        for info_plot in self.get_plot_objs("binary"):
            assert info_plot.n_classes == 2
            self.check_common(info_plot, "binary")
            if params.get("which_classes", None) is None:
                info_plot.plot(**params)
                self.close_plt(params)

    def _test_multi_class(self, params):
        for info_plot in self.get_plot_objs("multi-class"):
            assert info_plot.n_classes == 3
            self.check_common(info_plot, "multi-class")
            info_plot.plot(**params)
            self.close_plt(params)


class TestTargetPlot(TestInfoPlot):
    def get_plot_objs(self, target_type):
        dfs = get_dummy_dfs(target_type)
        info_plots = [
            TargetPlot(
                df,
                features[0],
                "Feature 1",
                target=[f"target_{i}" for i in range(3)]
                if target_type == "multi-class"
                else "target",
            )
            for (df, features) in dfs
        ]
        return info_plots

    def check_common(self, info_plot, target_type):
        assert info_plot.plot_type == "target"
        assert isinstance(info_plot.feature_info, FeatureInfo)

        if target_type == "multi-class":
            assert info_plot.target == [
                f"target_{i}" for i in range(info_plot.n_classes)
            ]
            assert len(info_plot.target_lines) == info_plot.n_classes
            for i in range(info_plot.n_classes):
                assert set(info_plot.target_lines[i].columns) == set(
                    ["x", f"target_{i}"]
                )
        else:
            assert info_plot.target == ["target"]
            assert len(info_plot.target_lines) == 1
            assert set(info_plot.target_lines[0].columns) == set(["x", "target"])

        assert set(info_plot.df.columns) == set(
            info_plot.feature_cols + ["x", "count"] + info_plot.target
        )
        assert set(info_plot.count_df.columns) == set(["x", "count", "count_norm"])

        summary_cols = ["x", "value", "count"] + info_plot.target
        if info_plot.feature_info.percentiles is not None:
            summary_cols.append("percentile")
        assert set(info_plot.summary_df.columns) == set(summary_cols)

    @pytest.mark.parametrize("params", plot_params)
    def test_regression(self, params):
        self._test_regression(params)

    @pytest.mark.parametrize("params", plot_params)
    def test_binary(self, params):
        self._test_binary(params)

    @pytest.mark.parametrize("params", plot_params)
    def test_multi_class(self, params):
        self._test_multi_class(params)


class TestPredictPlot(TestInfoPlot):
    def get_plot_objs(self, model_type):
        dfs = get_dummy_dfs()
        model = get_dummy_model(model_type)
        info_plots = [
            PredictPlot(
                df,
                features[0],
                "Feature 1",
                model=model,
                model_features=_make_list(features[0]) + _make_list(features[1]),
                n_classes=0 if model_type == "regression" else None,
            )
            for (df, features) in dfs
        ]
        return info_plots

    def check_common(self, info_plot, model_type):
        assert info_plot.plot_type == "predict"
        assert isinstance(info_plot.feature_info, FeatureInfo)

        summary_cols = ["x", "value", "count"]
        if model_type == "multi-class":
            assert info_plot.target == [f"pred_{i}" for i in range(info_plot.n_classes)]
            assert len(info_plot.target_lines) == info_plot.n_classes
            for i in range(info_plot.n_classes):
                target_cols = [f"pred_{i}_q1", f"pred_{i}_q2", f"pred_{i}_q3"]
                assert set(info_plot.target_lines[i].columns) == set(
                    ["x"] + target_cols
                )
                summary_cols += target_cols
        else:
            assert info_plot.target == ["pred"]
            assert len(info_plot.target_lines) == 1
            target_cols = ["pred_q1", "pred_q2", "pred_q3"]
            assert set(info_plot.target_lines[0].columns) == set(["x"] + target_cols)
            summary_cols += target_cols

        assert set(info_plot.df.columns) == set(
            info_plot.feature_cols + ["x", "count"] + info_plot.target
        )
        assert set(info_plot.count_df.columns) == set(["x", "count", "count_norm"])
        if info_plot.feature_info.percentiles is not None:
            summary_cols.append("percentile")
        assert set(info_plot.summary_df.columns) == set(summary_cols)

    @pytest.mark.parametrize("params", plot_params)
    def test_regression(self, params):
        self._test_regression(params)

    @pytest.mark.parametrize("params", plot_params)
    def test_binary(self, params):
        self._test_binary(params)

    @pytest.mark.parametrize("params", plot_params)
    def test_multi_class(self, params):
        self._test_multi_class(params)


class TestInterectTargetPlot(TestInfoPlot):
    def get_plot_objs(self, target_type):
        dfs = get_dummy_dfs(target_type)
        info_plots = [
            InterectTargetPlot(
                df,
                features,
                ["Feature 1", "Feature 2"],
                target=[f"target_{i}" for i in range(3)]
                if target_type == "multi-class"
                else "target",
            )
            for (df, features) in dfs
        ]
        return info_plots

    def check_common(self, info_plot, target_type):
        assert info_plot.plot_type == "interact_target"
        assert all(isinstance(info, FeatureInfo) for info in info_plot.feature_infos)
        assert len(info_plot.feature_infos) == 2

        if target_type == "multi-class":
            assert info_plot.target == [
                f"target_{i}" for i in range(info_plot.n_classes)
            ]
        else:
            assert info_plot.target == ["target"]
        assert set(info_plot.df.columns) == set(
            info_plot.feature_cols + ["x1", "x2", "count"] + info_plot.target
        )

        summary_cols = ["x1", "x2", "value_1", "value_2", "count"] + info_plot.target
        for i in range(2):
            if info_plot.feature_infos[i].percentiles is not None:
                summary_cols.append(f"percentile_{i+1}")
        assert set(info_plot.summary_df.columns) == set(summary_cols)
        assert set(info_plot.plot_df.columns) == set(
            ["x1", "x2", "count"] + info_plot.target
        )

    @pytest.mark.parametrize("params", plot_params)
    def test_regression(self, params):
        self._test_regression(params)

    @pytest.mark.parametrize("params", plot_params)
    def test_binary(self, params):
        self._test_binary(params)

    @pytest.mark.parametrize("params", plot_params)
    def test_multi_class(self, params):
        self._test_multi_class(params)


class TestInterectPredictPlot(TestInfoPlot):
    def get_plot_objs(self, model_type):
        dfs = get_dummy_dfs()
        model = get_dummy_model(model_type)
        info_plots = [
            InterectPredictPlot(
                df,
                features,
                ["Feature 1", "Feature 2"],
                model=model,
                model_features=_make_list(features[0]) + _make_list(features[1]),
                n_classes=0 if model_type == "regression" else None,
            )
            for (df, features) in dfs
        ]
        return info_plots

    def check_common(self, info_plot, model_type):
        assert info_plot.plot_type == "interact_predict"
        assert all(isinstance(info, FeatureInfo) for info in info_plot.feature_infos)
        assert len(info_plot.feature_infos) == 2

        if model_type == "multi-class":
            assert info_plot.target == [f"pred_{i}" for i in range(info_plot.n_classes)]
        else:
            assert info_plot.target == ["pred"]
        assert set(info_plot.df.columns) == set(
            info_plot.feature_cols + ["x1", "x2", "count"] + info_plot.target
        )

        summary_cols = ["x1", "x2", "value_1", "value_2", "count"] + info_plot.target
        for i in range(2):
            if info_plot.feature_infos[i].percentiles is not None:
                summary_cols.append(f"percentile_{i+1}")
        assert set(info_plot.summary_df.columns) == set(summary_cols)
        assert set(info_plot.plot_df.columns) == set(
            ["x1", "x2", "count"] + info_plot.target
        )

    @pytest.mark.parametrize("params", plot_params)
    def test_regression(self, params):
        self._test_regression(params)

    @pytest.mark.parametrize("params", plot_params)
    def test_binary(self, params):
        self._test_binary(params)

    @pytest.mark.parametrize("params", plot_params)
    def test_multi_class(self, params):
        self._test_multi_class(params)
