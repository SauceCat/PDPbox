import numpy as np
import pytest
import copy
from abc import abstractmethod

from pdpbox.pdp import PDPResult, PDPIsolate, PDPInteract
from pdpbox.utils import FeatureInfo, _make_list
from conftest import DummyModel, PlotTestBase

plot_params = [
    {"show_percentile": True},
    {"which_classes": [0]},
    {"which_classes": [0, 2]},
    {"figsize": (900, 500)},
    {"figsize": (12, 8), "engine": "matplotlib"},
    {"ncols": 1},
    {"ncols": 3},
    {"plot_params": {"line": {"width": 2, "markersize": 2, "fill_alpha": 0.5}}},
    {"engine": "matplotlib"},
    {"dpi": 100, "engine": "matplotlib"},
    {"template": "plotly_dark"},
]

pdp_plot_params = [
    {},
    {"center": False},
    {"plot_lines": True, "frac_to_plot": 0.1},
    {"plot_lines": True, "frac_to_plot": 100},
    {"cluster": True, "n_cluster_centers": 5},
    {"cluster": True, "n_cluster_centers": 10},
    {"cluster": True, "n_cluster_centers": 5, "cluster_method": "approx"},
    {"plot_pts_dist": True},
    {"plot_pts_dist": True, "to_bins": True},
] + plot_params

pdp_interact_plot_params = [
    {},
    {"plot_type": "contour", "plot_pdp": False, "to_bins": False},
    {"plot_type": "contour", "plot_pdp": True, "to_bins": True},
    {"plot_type": "contour", "plot_pdp": True, "to_bins": False},
    {"plot_type": "grid", "plot_pdp": False, "to_bins": True},
    {"plot_type": "grid", "plot_pdp": False, "to_bins": False},
    {"plot_type": "grid", "plot_pdp": True, "to_bins": True},
    {"plot_type": "grid", "plot_pdp": True, "to_bins": False},
] + plot_params

pdp_binary_params = [
    # binary feature
    [
        {"feature": "Sex", "feature_name": "gender"},
        {
            "center": False,
            "plot_lines": True,
            "frac_to_plot": 100,
            "plot_pts_dist": True,
            "plot_params": {"pdp_hl": True},
        },
        [
            {"engine": "plotly"},
            {"engine": "matplotlib", "dpi": 100},
        ],
    ],
    # onehot feature
    [
        {
            "feature": ["Embarked_C", "Embarked_S", "Embarked_Q"],
            "feature_name": "embarked",
        },
        {
            "plot_lines": True,
            "cluster": True,
            "n_cluster_centers": 50,
            "plot_pts_dist": True,
            "plot_params": {"pdp_hl": True, "line": {"hl_color": "#f46d43"}},
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
        {
            "plot_lines": True,
            "frac_to_plot": 0.1,
            "plot_pts_dist": True,
            "to_bins": True,
            "show_percentile": True,
            "plot_params": {"pdp_hl": True},
        },
        [
            {"engine": "plotly"},
            {"engine": "matplotlib"},
        ],
    ],
]

pdp_multiclass_params = [
    # numeric feature
    [
        {
            "feature": "feat_67",
            "feature_name": "feat_67",
            "chunk_size": 10000,
        },
        {
            "plot_lines": True,
            "frac_to_plot": 0.1,
            "plot_pts_dist": True,
            "show_percentile": True,
            "plot_params": {
                "pdp_hl": True,
                "gaps": {"outer_x": 0.06, "inner_y": 0.02, "outer_y": 0.1},
            },
        },
        [
            {"engine": "plotly", "figsize": (1200, 400), "ncols": 3},
            {"engine": "plotly", "which_classes": [1, 2]},
            {"engine": "matplotlib", "figsize": (16, 6.5), "which_classes": [0, 2]},
        ],
    ]
]

pdp_regression_params = [
    # binary feature
    [
        {
            "feature": "SchoolHoliday",
            "feature_name": "SchoolHoliday",
        },
        {
            "plot_lines": True,
            "frac_to_plot": 100,
            "show_percentile": True,
            "show_percentile": True,
            "plot_params": {
                "title": {
                    "title": {"text": "I am a title"},
                    "subtitle": {"text": "I am a subtitle"},
                }
            },
        },
        [
            {"engine": "plotly", "template": "plotly_dark"},
            {"engine": "matplotlib", "figsize": (12, 8)},
        ],
    ],
    # onehot feature
    [
        {
            "feature": ["StoreType_a", "StoreType_b", "StoreType_c", "StoreType_d"],
            "feature_name": "StoreType",
        },
        {
            "plot_lines": True,
            "frac_to_plot": 500,
            "plot_params": {"pdp_hl": True},
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
            "feature_name": "weekofyear",
        },
        {
            "plot_lines": True,
            "frac_to_plot": 100,
            "plot_pts_dist": True,
            "to_bins": True,
            "show_percentile": True,
            "plot_params": {"pdp_hl": True},
        },
        [
            {"engine": "plotly"},
            {"engine": "matplotlib"},
        ],
    ],
]

pdp_interact_binary_params = [
    # numeric, numeric
    [
        {
            "features": ["Age", "Fare"],
            "feature_names": ["age", "fare"],
        },
        {"plot_pdp": True, "show_percentile": True},
        [
            {"engine": "plotly"},
            {"engine": "plotly", "plot_type": "grid"},
            {"engine": "matplotlib"},
        ],
    ],
    # numeric, binary
    [
        {"features": ["Age", "Sex"], "feature_names": ["age", "gender"]},
        {"plot_pdp": True, "show_percentile": True},
        [
            {"engine": "plotly"},
            {"engine": "matplotlib"},
            {"engine": "matplotlib", "plot_type": "grid"},
        ],
    ],
    # numeric, onehot
    [
        {
            "features": ["Age", ["Embarked_C", "Embarked_S", "Embarked_Q"]],
            "feature_names": ["age", "embarked"],
        },
        {
            "plot_pdp": True,
            "show_percentile": True,
        },
        [
            {"engine": "plotly"},
            {"engine": "matplotlib"},
            {"engine": "matplotlib", "plot_type": "grid"},
        ],
    ],
    # binary, onehot
    [
        {
            "features": ["Sex", ["Embarked_C", "Embarked_S", "Embarked_Q"]],
            "feature_names": ["gender", "embarked"],
        },
        {
            "plot_pdp": True,
            "plot_type": "grid",
        },
        [
            {"engine": "plotly"},
            {"engine": "plotly", "plot_type": "contour"},
            {"engine": "matplotlib"},
        ],
    ],
]

pdp_interact_multiclass_params = [
    # numeric, numeric
    [
        {
            "features": ["feat_67", "feat_25"],
            "feature_names": ["feat_67", "feat_25"],
        },
        {
            "plot_pdp": False,
            "show_percentile": True,
        },
        [
            {"engine": "plotly", "plot_type": "contour"},
            {"engine": "plotly", "plot_type": "contour", "which_classes": [2]},
            {
                "engine": "matplotlib",
                "which_classes": [2, 4, 6, 8],
                "plot_params": {"gaps": {"outer_y": 0.2}},
                "plot_type": "grid",
            },
        ],
    ],
]

pdp_interact_regression_params = [
    # numeric, onehot
    [
        {
            "features": [
                "weekofyear",
                ["StoreType_a", "StoreType_b", "StoreType_c", "StoreType_d"],
            ],
            "feature_names": ["weekofyear", "storetype"],
        },
        {
            "plot_type": "contour",
            "plot_pdp": True,
        },
        [
            {"engine": "plotly"},
            {"engine": "matplotlib"},
        ],
    ],
]


class _TestPDPBase(PlotTestBase):
    @abstractmethod
    def check_ice_pdp(self, plot_obj, model_type):
        pass

    def _test_plot_obj(self, model_type):
        # n_jobs > 1 is too slow so far, skip testing
        for plot_obj in self.get_plot_objs(model_type):
            self.check_common(plot_obj, model_type)
            self.check_ice_pdp(plot_obj, model_type)

    def _test_plot(self, params):
        # randomly choose a model_type
        model_type = np.random.choice(self.model_types)
        if params.get("which_classes", None) is not None:
            model_type = "multi-class"

        for plot_obj in self.get_plot_objs(model_type):
            plot_obj.plot(**params)
            self.close_plt(params)
            break

    def _test_real_plot(self, plot_obj, plot_params, extra_plot_settings):
        for extra_params in extra_plot_settings:
            curr_plot_params = copy.deepcopy(plot_params)
            curr_plot_params.update(extra_params)
            fig, axes = plot_obj.plot(**curr_plot_params)
            self.close_plt(curr_plot_params)
            assert fig is not None
            if curr_plot_params.get("engine", "plotly") == "matplotlib":
                assert axes is not None
            else:
                assert axes is None


class TestPDPIsolate(_TestPDPBase):
    def get_plot_objs(self, model_type, n_jobs=1):
        for df, features, feat_types in self.data_gen.get_dummy_dfs():
            model = DummyModel(model_type, features[0], feat_types[0])
            for n_jobs_ in _make_list(n_jobs):
                yield PDPIsolate(
                    model,
                    df,
                    model_features=_make_list(features[0]) + _make_list(features[1]),
                    feature=features[0],
                    feature_name="Feature 1",
                    n_classes=0 if model_type == "regression" else None,
                    n_jobs=n_jobs_,
                )

    def check_common(self, plot_obj, model_type):
        assert plot_obj.plot_type == "pdp_isolate"
        assert isinstance(plot_obj.feature_info, FeatureInfo)
        assert set(plot_obj.count_df.columns) == {"x", "count", "count_norm"}
        assert len(plot_obj.dist_df) <= plot_obj.dist_num_samples

        target_len = plot_obj.n_classes if model_type == "multi-class" else 1
        assert plot_obj.target == list(range(target_len))
        assert len(plot_obj.results) == target_len

        for cls_idx, res in enumerate(plot_obj.results):
            assert isinstance(res, PDPResult)
            assert (
                res.class_id == cls_idx
                if model_type == "multi-class"
                else res.class_id is None
            )
            assert res.ice_lines.shape == (
                self.data_gen.DUMMY_DF_LENGTH,
                plot_obj.n_grids,
            )
            assert res.pdp.shape == (plot_obj.n_grids,)

    def calculate_target_value(self, model_type, i, grid, feat_type, cls_idx):
        if model_type == "regression":
            target_value = i if feat_type == "onehot" else grid
        elif model_type == "binary":
            target_value = (
                (i + 1) / self.data_gen.DUMMY_ONEHOT_LENGTH
                if feat_type == "onehot"
                else 1
            )
        elif model_type == "multi-class":
            if feat_type == "onehot":
                pos_value = (i + 1) / self.data_gen.DUMMY_ONEHOT_LENGTH
                target_value = (
                    pos_value
                    if cls_idx == 0
                    else (1 - pos_value) / (self.data_gen.DUMMY_MULTI_CLASSES - 1)
                )
            else:
                target_value = 1 if cls_idx == 0 else 0
        return target_value

    def check_ice_pdp(self, plot_obj, model_type):
        feat_type = plot_obj.feature_info.type
        for cls_idx, res in enumerate(plot_obj.results):
            ice_lines = res.ice_lines.to_numpy()
            grids = plot_obj.feature_info.grids

            for i, grid in enumerate(grids):
                ice_uni_values = np.unique(ice_lines[:, i])
                assert len(ice_uni_values) == 1

                target_value = self.calculate_target_value(
                    model_type, i, grid, feat_type, cls_idx
                )
                assert np.isclose(
                    res.pdp[i], target_value, rtol=1e-05, atol=1e-08, equal_nan=False
                )
                assert np.isclose(
                    ice_uni_values[0],
                    target_value,
                    rtol=1e-05,
                    atol=1e-08,
                    equal_nan=False,
                )

    @pytest.mark.parametrize("model_type", ["regression", "binary", "multi-class"])
    def test_plot_obj(self, model_type):
        self._test_plot_obj(model_type)

    @pytest.mark.parametrize("params", pdp_plot_params)
    def test_plot(self, params):
        self._test_plot(params)

    @pytest.mark.parametrize(
        "params, plot_params, extra_plot_settings", pdp_binary_params
    )
    def test_real_binary_model(self, params, plot_params, extra_plot_settings, titanic):
        plot_obj = PDPIsolate(
            model=titanic["model"],
            df=titanic["data"],
            model_features=titanic["features"],
            **params
        )
        self._test_real_plot(plot_obj, plot_params, extra_plot_settings)

    @pytest.mark.parametrize(
        "params, plot_params, extra_plot_settings", pdp_multiclass_params
    )
    def test_real_multiclass_model(
        self, params, plot_params, extra_plot_settings, otto
    ):
        plot_obj = PDPIsolate(
            model=otto["model"],
            df=otto["data"],
            model_features=otto["features"],
            **params
        )
        self._test_real_plot(plot_obj, plot_params, extra_plot_settings)

    @pytest.mark.parametrize(
        "params, plot_params, extra_plot_settings", pdp_regression_params
    )
    def test_real_regression_model(
        self, params, plot_params, extra_plot_settings, ross
    ):
        plot_obj = PDPIsolate(
            model=ross["model"],
            df=ross["data"],
            model_features=ross["features"],
            n_classes=0,
            **params
        )
        self._test_real_plot(plot_obj, plot_params, extra_plot_settings)


class TestPDPInteract(_TestPDPBase):
    def get_plot_objs(self, model_type, n_jobs=1):
        for df, features, feat_types in self.data_gen.get_dummy_dfs():
            model = DummyModel(model_type, features, feat_types, interact=True)
            for n_jobs_ in _make_list(n_jobs):
                yield PDPInteract(
                    model,
                    df,
                    model_features=_make_list(features[0]) + _make_list(features[1]),
                    features=features,
                    feature_names=["Feature 1", "Feature 2"],
                    n_classes=0 if model_type == "regression" else None,
                    n_jobs=n_jobs_,
                )

    def check_common(self, plot_obj, model_type):
        assert plot_obj.plot_type == "pdp_interact"
        n_grids = []
        for obj in plot_obj.pdp_isolate_objs:
            assert isinstance(obj, PDPIsolate)
            assert isinstance(obj.feature_info, FeatureInfo)
            n_grids.append(obj.n_grids)

        target_len = plot_obj.n_classes if model_type == "multi-class" else 1
        assert plot_obj.target == list(range(target_len))
        assert len(plot_obj.results) == target_len
        assert plot_obj.n_grids == np.prod(n_grids)
        assert len(plot_obj.feature_grid_combos) == plot_obj.n_grids

        for cls_idx, res in enumerate(plot_obj.results):
            assert isinstance(res, PDPResult)
            assert (
                res.class_id == cls_idx
                if model_type == "multi-class"
                else res.class_id is None
            )
            assert res.ice_lines.shape == (
                self.data_gen.DUMMY_DF_LENGTH,
                plot_obj.n_grids,
            )
            assert res.pdp.shape == (plot_obj.n_grids,)

    def calculate_target_value(self, model_type, grid, feat_types, cls_idx):
        def get_onehot_value(j):
            return np.argmax(
                grid[: self.data_gen.DUMMY_ONEHOT_LENGTH]
                if j == 0
                else grid[-self.data_gen.DUMMY_ONEHOT_LENGTH :]
            )

        def get_value(j, feat_type, is_regression=True):
            if is_regression:
                if feat_type == "onehot":
                    return get_onehot_value(j)
                else:
                    return grid[j] if j == 0 else grid[-j]
            else:
                if feat_type == "onehot":
                    return (get_onehot_value(j) + 1) / self.data_gen.DUMMY_ONEHOT_LENGTH
                else:
                    return 1

        if model_type == "regression":
            target_value = sum(
                [get_value(j, feat_type) for j, feat_type in enumerate(feat_types)]
            )
        else:
            pos_value = np.prod(
                [
                    get_value(j, feat_type, is_regression=False)
                    for j, feat_type in enumerate(feat_types)
                ]
            )
            if model_type == "binary":
                target_value = pos_value
            else:
                target_value = (
                    pos_value
                    if cls_idx == 0
                    else (1 - pos_value) / (self.data_gen.DUMMY_MULTI_CLASSES - 1)
                )

        return target_value

    def check_ice_pdp(self, plot_obj, model_type):
        feat_types = [obj.feature_info.type for obj in plot_obj.pdp_isolate_objs]
        grid_combos = plot_obj.feature_grid_combos

        for cls_idx, res in enumerate(plot_obj.results):
            ice_lines = res.ice_lines.to_numpy()

            for i, grid in enumerate(grid_combos):
                ice_uni_values = np.unique(ice_lines[:, i])
                assert len(ice_uni_values) == 1

                target_value = self.calculate_target_value(
                    model_type, grid, feat_types, cls_idx
                )
                assert np.isclose(
                    res.pdp[i], target_value, rtol=1e-05, atol=1e-08, equal_nan=False
                )
                assert np.isclose(
                    ice_uni_values[0],
                    target_value,
                    rtol=1e-05,
                    atol=1e-08,
                    equal_nan=False,
                )

    @pytest.mark.parametrize("model_type", ["regression", "binary", "multi-class"])
    def test_plot_obj(self, model_type):
        self._test_plot_obj(model_type)

    @pytest.mark.parametrize("params", pdp_interact_plot_params)
    def test_plot(self, params):
        self._test_plot(params)

    @pytest.mark.parametrize(
        "params, plot_params, extra_plot_settings", pdp_interact_binary_params
    )
    def test_real_binary_model(self, params, plot_params, extra_plot_settings, titanic):
        plot_obj = PDPInteract(
            model=titanic["model"],
            df=titanic["data"],
            model_features=titanic["features"],
            **params
        )
        self._test_real_plot(plot_obj, plot_params, extra_plot_settings)

    @pytest.mark.parametrize(
        "params, plot_params, extra_plot_settings", pdp_interact_multiclass_params
    )
    def test_real_multiclass_model(
        self, params, plot_params, extra_plot_settings, otto
    ):
        plot_obj = PDPInteract(
            model=otto["model"],
            df=otto["data"],
            model_features=otto["features"],
            **params
        )
        self._test_real_plot(plot_obj, plot_params, extra_plot_settings)

    @pytest.mark.parametrize(
        "params, plot_params, extra_plot_settings", pdp_interact_regression_params
    )
    def test_real_regression_model(
        self, params, plot_params, extra_plot_settings, ross
    ):
        plot_obj = PDPInteract(
            model=ross["model"],
            df=ross["data"],
            model_features=ross["features"],
            n_classes=0,
            **params
        )
        self._test_real_plot(plot_obj, plot_params, extra_plot_settings)
