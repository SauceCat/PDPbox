import pytest
import numpy as np

from pdpbox.info_plots import (
    TargetPlot,
    PredictPlot,
    InteractTargetPlot,
    InteractPredictPlot,
)
from pdpbox.utils import FeatureInfo, _make_list
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


def titanic_predict_proba(model, X, predict_kwds=None):
    if predict_kwds is None:
        predict_kwds = {}
    return model.predict_proba(X, **predict_kwds) + 0.05


info_binary_params = [
    # binary feature
    [
        {"feature": "Sex", "feature_name": "gender"},
        [
            {"engine": "plotly"},
            {"engine": "matplotlib"},
            {"engine": "matplotlib", "dpi": 200},
        ],
    ],
    # onehot feature
    [
        {
            "feature": ["Embarked_C", "Embarked_S", "Embarked_Q"],
            "feature_name": "embarked",
        },
        [
            {"engine": "plotly"},
            {"engine": "matplotlib"},
        ],
    ],
    # numeric feature
    [
        {
            "feature": "Fare",
            "feature_name": "fare",
        },
        [
            {"engine": "plotly"},
            {"engine": "matplotlib"},
        ],
    ],
]

predict_binary_params = [
    [
        {
            "feature": "Sex",
            "feature_name": "gender",
            "pred_func": titanic_predict_proba,
        },
        [
            {"engine": "plotly"},
        ],
    ],
]

info_multiclass_params = [
    # numeric feature
    [
        {
            "feature": "feat_67",
            "feature_name": "feat_67",
        },
        [
            {
                "figsize": (1200, 400),
                "plot_params": {"gaps": {"outer_y": 0.05}},
            },
            {
                "which_classes": [1, 2],
                "show_percentile": True,
                "figsize": (1200, 400),
                "plot_params": {"gaps": {"outer_y": 0.05, "top": 0.1}},
            },
            {
                "show_percentile": True,
                "figsize": (16, 6),
                "plot_params": {
                    "gaps": {"outer_y": 0.4},
                    "title": {"subplot_title": {"fontsize": 11}},
                },
                "engine": "matplotlib",
            },
            {
                "which_classes": [1, 3, 5, 8],
                "show_percentile": True,
                "plot_params": {
                    "title": {"subplot_title": {"fontsize": 10}},
                    "gaps": {"outer_y": 0.25},
                },
                "engine": "matplotlib",
            },
        ],
    ]
]

info_regression_params = [
    # binary feature
    [
        {
            "feature": "SchoolHoliday",
            "feature_name": "school holiday",
        },
        [
            {"engine": "plotly"},
            {"engine": "matplotlib"},
            {"engine": "plotly", "plot_params": {"gaps": {"inner_y": 0.05}}},
        ],
    ],
    # onehot feature
    [
        {
            "feature": ["StoreType_a", "StoreType_b", "StoreType_c", "StoreType_d"],
            "feature_name": "store type",
        },
        [
            {"engine": "plotly"},
            {"engine": "matplotlib"},
        ],
    ],
    # numeric feature
    [
        {
            "feature": "weekofyear",
            "feature_name": "week of year",
        },
        [
            {"engine": "plotly"},
            {"engine": "matplotlib"},
        ],
    ],
]

info_interact_binary_params = [
    # numeric, numeric
    [
        {
            "features": ["Age", "Fare"],
            "feature_names": ["age", "fare"],
        },
        [
            {
                "engine": "plotly",
                "figsize": (1200, 800),
                "plot_params": {"subplot_ratio": {"y": [10, 1]}},
                "annotate": True,
            },
            {"engine": "matplotlib", "figsize": (16, 10)},
        ],
    ],
    # numeric, binary
    [
        {"features": ["Age", "Sex"], "feature_names": ["age", "gender"]},
        [
            {"engine": "plotly", "figsize": (1200, 700)},
            {"engine": "matplotlib", "annotate": True},
        ],
    ],
    # numeric, onehot
    [
        {
            "features": ["Age", ["Embarked_C", "Embarked_S", "Embarked_Q"]],
            "feature_names": ["age", "embarked"],
        },
        [
            {"engine": "plotly", "annotate": True},
            {"engine": "matplotlib", "annotate": True},
        ],
    ],
    # binary, onehot
    [
        {
            "features": ["Sex", ["Embarked_C", "Embarked_S", "Embarked_Q"]],
            "feature_names": ["gender", "embarked"],
        },
        [
            {"engine": "plotly", "template": "seaborn"},
            {"engine": "matplotlib"},
        ],
    ],
]

info_interact_multiclass_params = [
    # numeric, numeric
    [
        {
            "features": ["feat_67", "feat_25"],
            "feature_names": ["feat_67", "feat_25"],
        },
        [
            {
                "which_classes": [0, 1, 2, 3],
                "plot_params": {"gaps": {"inner_y": 0.06}},
                "engine": "plotly",
            },
            {
                "which_classes": [0, 2, 3],
                "plot_params": {"title": {"subplot_title": {"fontsize": 10}}},
                "engine": "matplotlib",
            },
            {
                "which_classes": [1, 3, 5, 6],
                "plot_params": {"gaps": {"inner_y": 0.06}},
                "engine": "plotly",
                "show_percentile": True,
            },
        ],
    ],
]

info_interact_regression_params = [
    # numeric, onehot
    [
        {
            "features": [
                "weekofyear",
                ["StoreType_a", "StoreType_b", "StoreType_c", "StoreType_d"],
            ],
            "feature_names": ["weekofyear", "storetype"],
        },
        [
            {
                "engine": "plotly",
                "plot_params": {
                    "subplot_ratio": {"y": [7, 0.8]},
                    "gaps": {"inner_y": 0.2},
                },
                "annotate": True,
                "show_percentile": True,
            },
            {
                "engine": "matplotlib",
                "show_percentile": True,
                "annotate": True,
            },
        ],
    ],
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

    def _test_real_plot(self, plot_obj, plot_params_list):
        for plot_params in plot_params_list:
            fig, axes, _ = plot_obj.plot(**plot_params)
            self.close_plt(plot_params)
            assert fig is not None
            if plot_params.get("engine", "plotly") == "matplotlib":
                assert axes is not None
            else:
                assert axes is None


class TestTargetPlot(_TestInfoPlot):
    def get_plot_objs(self, target_type):
        for df, features, _ in self.data_gen.get_dummy_dfs(target_type):
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

    @pytest.mark.parametrize("params, plot_params_list", info_binary_params)
    def test_real_binary_model(self, params, plot_params_list, titanic):
        plot_obj = TargetPlot(df=titanic["data"], target=titanic["target"], **params)
        self._test_real_plot(plot_obj, plot_params_list)

    @pytest.mark.parametrize("params, plot_params_list", info_multiclass_params)
    def test_real_multiclass_model(self, params, plot_params_list, otto):
        plot_obj = TargetPlot(df=otto["data"], target=otto["target"][1:], **params)
        self._test_real_plot(plot_obj, plot_params_list)

    @pytest.mark.parametrize("params, plot_params_list", info_regression_params)
    def test_real_regression_model(self, params, plot_params_list, ross):
        plot_obj = TargetPlot(df=ross["data"], target=ross["target"], **params)
        self._test_real_plot(plot_obj, plot_params_list)


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

    @pytest.mark.parametrize(
        "params, plot_params_list", info_binary_params + predict_binary_params
    )
    def test_real_binary_model(self, params, plot_params_list, titanic):
        plot_obj = PredictPlot(
            model=titanic["model"],
            df=titanic["data"],
            model_features=titanic["features"],
            **params,
        )
        self._test_real_plot(plot_obj, plot_params_list)

    @pytest.mark.parametrize("params, plot_params_list", info_multiclass_params)
    def test_real_multiclass_model(self, params, plot_params_list, otto):
        plot_obj = PredictPlot(
            model=otto["model"],
            df=otto["data"],
            model_features=otto["features"],
            **params,
        )
        self._test_real_plot(plot_obj, plot_params_list)

    @pytest.mark.parametrize("params, plot_params_list", info_regression_params)
    def test_real_regression_model(self, params, plot_params_list, ross):
        plot_obj = PredictPlot(
            model=ross["model"],
            df=ross["data"],
            model_features=ross["features"],
            n_classes=0,
            **params,
        )
        self._test_real_plot(plot_obj, plot_params_list)


class TestInteractTargetPlot(_TestInfoPlot):
    def get_plot_objs(self, target_type):
        for df, features, _ in self.data_gen.get_dummy_dfs(target_type):
            yield InteractTargetPlot(
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

    @pytest.mark.parametrize("params, plot_params_list", info_interact_binary_params)
    def test_real_binary_model(self, params, plot_params_list, titanic):
        plot_obj = InteractTargetPlot(
            df=titanic["data"], target=titanic["target"], **params
        )
        self._test_real_plot(plot_obj, plot_params_list)

    @pytest.mark.parametrize(
        "params, plot_params_list", info_interact_multiclass_params
    )
    def test_real_multiclass_model(self, params, plot_params_list, otto):
        plot_obj = InteractTargetPlot(
            df=otto["data"], target=otto["target"][1:], **params
        )
        self._test_real_plot(plot_obj, plot_params_list)

    @pytest.mark.parametrize(
        "params, plot_params_list", info_interact_regression_params
    )
    def test_real_regression_model(self, params, plot_params_list, ross):
        plot_obj = InteractTargetPlot(df=ross["data"], target=ross["target"], **params)
        self._test_real_plot(plot_obj, plot_params_list)


class TestInteractPredictPlot(_TestInfoPlot):
    def get_plot_objs(self, model_type):
        for df, features, feat_types in self.data_gen.get_dummy_dfs():
            model = DummyModel(model_type, features, feat_types, interact=True)
            yield InteractPredictPlot(
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

    @pytest.mark.parametrize("params, plot_params_list", info_interact_binary_params)
    def test_real_binary_model(self, params, plot_params_list, titanic):
        plot_obj = InteractPredictPlot(
            model=titanic["model"],
            df=titanic["data"],
            model_features=titanic["features"],
            **params,
        )
        self._test_real_plot(plot_obj, plot_params_list)

    @pytest.mark.parametrize(
        "params, plot_params_list", info_interact_multiclass_params
    )
    def test_real_multiclass_model(self, params, plot_params_list, otto):
        plot_obj = InteractPredictPlot(
            model=otto["model"],
            df=otto["data"],
            model_features=otto["features"],
            **params,
        )
        self._test_real_plot(plot_obj, plot_params_list)

    @pytest.mark.parametrize(
        "params, plot_params_list", info_interact_regression_params
    )
    def test_real_regression_model(self, params, plot_params_list, ross):
        plot_obj = InteractPredictPlot(
            model=ross["model"],
            df=ross["data"],
            model_features=ross["features"],
            n_classes=0,
            **params,
        )
        self._test_real_plot(plot_obj, plot_params_list)
