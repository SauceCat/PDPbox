import os
import sys
from itertools import product
import numpy as np
import pandas as pd
import pytest
import matplotlib
from abc import ABC, abstractmethod

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pdpbox.pdp import PDPResult, PDPIsolate, PDPInteract
from pdpbox.utils import FeatureInfo, _make_list

DUMMY_ONEHOT_LENGTH = 4
DUMMY_DF_LENGTH = 100
DUMMY_MULTI_CLASSES = 3
FEATURE_TYPES = ["numeric", "onehot", "binary"]

dummy_features = {
    "numeric": np.random.randn(DUMMY_DF_LENGTH),
    "onehot": np.array(
        [
            np.eye(DUMMY_ONEHOT_LENGTH)[i]
            for i in np.random.randint(0, DUMMY_ONEHOT_LENGTH, DUMMY_DF_LENGTH)
        ]
    ),
    "binary": np.random.randint(0, 2, DUMMY_DF_LENGTH),
}

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


def shuffle_array(array):
    shuffled = np.array(array, copy=True)
    np.random.shuffle(shuffled)
    return shuffled


def get_dummy_dfs():
    feat_types_1 = shuffle_array(FEATURE_TYPES)
    feat_types_2 = shuffle_array(FEATURE_TYPES)

    for t1, t2 in product(feat_types_1, feat_types_2):
        df = {}
        features = []
        for i, t in enumerate((t1, t2), start=1):
            if t == "onehot":
                for j in range(DUMMY_ONEHOT_LENGTH):
                    df[f"feature_{i}_{j}"] = dummy_features[t][:, j]
                features.append(
                    [f"feature_{i}_{j}" for j in range(DUMMY_ONEHOT_LENGTH)]
                )
            else:
                df[f"feature_{i}"] = dummy_features[t]
                features.append(f"feature_{i}")
        yield pd.DataFrame(df), features, (t1, t2)


class DummyModel:
    def __init__(self, model_type, feat, feat_type, interact=False):
        self.feat = feat
        self.feat_type = feat_type
        self.interact = interact

        if model_type == "regression":
            self.predict = self.dummy_regression_predict
        elif model_type == "binary":
            self.predict_proba = self.dummy_binary_predict_proba
            self.n_classes_ = 2
        elif model_type == "multi-class":
            self.predict_proba = self.dummy_multi_class_predict_proba
            self.n_classes_ = DUMMY_MULTI_CLASSES

    def _compute_feature(self, x, feat, feat_type):
        if feat_type == "onehot":
            return np.argmax(x[feat].to_numpy(), axis=1)
        else:
            return x[feat].values

    def _compute_score(self, x, feat, feat_type):
        if feat_type == "onehot":
            return (np.argmax(x[feat].to_numpy(), axis=1) + 1) / len(feat)
        else:
            return (
                np.ones(len(x))
                if x[feat].max() == 0
                else x[feat].values / x[feat].max()
            )

    def dummy_regression_predict(self, x):
        if self.interact:
            return sum(
                self._compute_feature(x, f, t)
                for f, t in zip(self.feat, self.feat_type)
            )
        else:
            return self._compute_feature(x, self.feat, self.feat_type)

    def predict_proba_func(self, x):
        if self.interact:
            values = [
                self._compute_score(x, f, t) for f, t in zip(self.feat, self.feat_type)
            ]
            return values[0] * values[1]
        else:
            return self._compute_score(x, self.feat, self.feat_type)

    def dummy_binary_predict_proba(self, x):
        scores = self.predict_proba_func(x)
        return np.stack((1 - scores, scores)).T

    def dummy_multi_class_predict_proba(self, x):
        scores = self.predict_proba_func(x)
        each_scores = (1 - scores) / (self.n_classes_ - 1)
        proba = np.zeros((len(x), self.n_classes_))
        proba[:, 0] = scores
        proba[:, 1:] = each_scores[:, np.newaxis]
        return proba


class _TestPDPBase(ABC):
    def setup(self):
        self.model_types = ["regression", "binary", "multi-class"]

    @staticmethod
    def close_plt(params):
        if params.get("engine", "plotly") == "matplotlib":
            matplotlib.pyplot.close()

    @abstractmethod
    def get_plot_objs(self, model_type):
        pass

    @abstractmethod
    def check_common(self, plot_obj, model_type):
        pass

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


class TestPDPIsolate(_TestPDPBase):
    @staticmethod
    def get_plot_objs(model_type, n_jobs=1):
        for df, features, feat_types in get_dummy_dfs():
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

    @staticmethod
    def check_common(plot_obj, model_type):
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
            assert res.ice_lines.shape == (DUMMY_DF_LENGTH, plot_obj.n_grids)
            assert res.pdp.shape == (plot_obj.n_grids,)

    @staticmethod
    def calculate_target_value(model_type, i, grid, feat_type, cls_idx):
        if model_type == "regression":
            target_value = i if feat_type == "onehot" else grid
        elif model_type == "binary":
            target_value = (i + 1) / DUMMY_ONEHOT_LENGTH if feat_type == "onehot" else 1
        elif model_type == "multi-class":
            if feat_type == "onehot":
                pos_value = (i + 1) / DUMMY_ONEHOT_LENGTH
                target_value = (
                    pos_value
                    if cls_idx == 0
                    else (1 - pos_value) / (DUMMY_MULTI_CLASSES - 1)
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


class TestPDPInteract(_TestPDPBase):
    @staticmethod
    def get_plot_objs(model_type, n_jobs=1):
        for df, features, feat_types in get_dummy_dfs():
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

    @staticmethod
    def check_common(plot_obj, model_type):
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
            assert res.ice_lines.shape == (DUMMY_DF_LENGTH, plot_obj.n_grids)
            assert res.pdp.shape == (plot_obj.n_grids,)

    @staticmethod
    def calculate_target_value(model_type, grid, feat_types, cls_idx):
        def get_onehot_value(j):
            return np.argmax(
                grid[:DUMMY_ONEHOT_LENGTH] if j == 0 else grid[-DUMMY_ONEHOT_LENGTH:]
            )

        def get_value(j, feat_type, is_regression=True):
            if is_regression:
                if feat_type == "onehot":
                    return get_onehot_value(j)
                else:
                    return grid[j] if j == 0 else grid[-j]
            else:
                if feat_type == "onehot":
                    return (get_onehot_value(j) + 1) / DUMMY_ONEHOT_LENGTH
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
                    else (1 - pos_value) / (DUMMY_MULTI_CLASSES - 1)
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
        if model_type == "binary":

            self._test_plot_obj(model_type)

    @pytest.mark.parametrize("params", pdp_interact_plot_params)
    def test_plot(self, params):
        self._test_plot(params)
